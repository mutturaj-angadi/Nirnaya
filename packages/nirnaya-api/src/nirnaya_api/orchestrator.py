"""
The single orchestration path used by the Python API (:mod:`nirnaya_api.model`),
the CLI (:mod:`nirnaya_api.cli`), and the REST API (:mod:`nirnaya_api.server`).

Having exactly one function that drives "validate -> select backend ->
presolve -> solve -> build Solution" is what keeps the three interfaces
behaviorally identical - none of them re-implement orchestration.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from pydantic import ValidationError as PydanticValidationError

from . import _integration as integ
from .backends import select_backend
from .errors import ValidationError, SolveTimeoutError, DeviceUnavailableError
from .options import SolveOptions
from .schema import ModelSchema
from .security import (
    DEFAULT_LIMITS,
    SecurityLimits,
    check_model_size,
    clamp_time_limit,
    run_with_wall_clock_limit,
)
from .solution import NumericalDiagnostics, Solution, SolveStatus


def parse_and_validate_model(model_dict: dict) -> ModelSchema:
    """Strict schema-level validation of the JSON model format.

    Does not require nirnaya-core to be installed - this is pure format
    validation (shape, types, cross-references, bound sanity), independent
    of the upstream solving stack.
    """
    try:
        return ModelSchema.model_validate(_normalize_model_payload(model_dict))
    except PydanticValidationError as exc:
        issues = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise ValidationError(
            "Model failed schema validation.", details={"issues": issues}
        )


def _normalize_model_payload(payload: dict) -> dict:
    """Accept both the API's mapping schema and the supplied UI array schema."""
    if not isinstance(payload, dict) or not isinstance(payload.get("variables"), list):
        return payload
    variables = {}
    for item in payload["variables"]:
        if not isinstance(item, dict) or "name" not in item:
            raise ValidationError("Each variable entry must contain a name.")
        variables[item["name"]] = {
            "lb": 0.0 if item.get("lower", 0.0) is None else item.get("lower", 0.0),
            "ub": float("inf") if item.get("upper") is None else item.get("upper"),
            "kind": item.get("type", "continuous"),
        }
    constraints = {}
    for item in payload.get("constraints", []):
        if not isinstance(item, dict) or "name" not in item:
            raise ValidationError("Each constraint entry must contain a name.")
        sense = "=" if item.get("sense") in ("=", "==") else item.get("sense")
        constraints[item["name"]] = {"coefficients": item.get("coefficients", {}),
                                      "sense": sense, "rhs": item.get("rhs")}
    objective = dict(payload.get("objective", {}))
    objective["sense"] = {"minimize": "min", "maximize": "max"}.get(
        payload.get("sense", "minimize"), "min")
    objective.setdefault("offset", 0.0)
    return {"format_version": payload.get("format_version", "1.0"),
            "name": payload.get("name"), "variables": variables,
            "constraints": constraints, "objective": objective}


def validate_model(model_dict: dict, limits: SecurityLimits = DEFAULT_LIMITS) -> Dict[str, Any]:
    """Full validation used by ``nirnaya validate`` and ``POST /validate``.

    Returns a dict suitable for :class:`nirnaya_api.schema.ValidateResponseSchema`.
    Never raises for *ordinary* invalid models - it reports issues instead -
    but still raises :class:`ValidationError`/`PayloadTooLargeError` for
    structurally unparseable input, since there is nothing meaningful to
    report on.
    """
    issues: List[str] = []
    try:
        parsed = parse_and_validate_model(model_dict)
    except ValidationError as exc:
        issues = exc.details.get("issues", [exc.message])
        # model_dict may not even be a mapping (e.g. a JSON array or scalar
        # was submitted) - fall back to 0 rather than raising here, since
        # this branch's whole job is to report issues, not add new ones.
        safe = model_dict if isinstance(model_dict, dict) else {}
        return {
            "valid": False,
            "num_variables": len(safe.get("variables", {}) or {}),
            "num_constraints": len(safe.get("constraints", {}) or {}),
            "issues": issues,
        }

    try:
        check_model_size(parsed.to_plain_dict(), limits)
    except Exception as exc:  # ResourceLimitError
        issues.append(str(exc))

    return {
        "valid": len(issues) == 0,
        "num_variables": len(parsed.variables),
        "num_constraints": len(parsed.constraints),
        "issues": issues,
    }


def solve_model(
    model_dict: dict,
    options: SolveOptions,
    limits: SecurityLimits = DEFAULT_LIMITS,
) -> Solution:
    """Run the full pipeline: validate -> select backend -> presolve ->
    solve -> Solution. This is what every interface ultimately calls.
    """
    options.validate()

    parsed = parse_and_validate_model(model_dict)
    plain = parsed.to_plain_dict()
    check_model_size(plain, limits)

    backend_impl = integ.get_backend()
    resolved_backend, resolved_device, warnings = select_backend(options.backend, options.device)
    # Device discovery describes hardware primitives, not the active LP
    # algorithm. The current revised simplex is CPU-only, so never label an
    # LP solve as GPU work merely because CUDA primitives are present.
    if resolved_backend == "gpu" and isinstance(backend_impl, integ.RealSolverBackend):
        if options.backend == "gpu" or options.device is not None:
            raise DeviceUnavailableError("The current LP solver supports CPU execution only.")
        resolved_backend, resolved_device = "cpu", "cpu:0"
        warnings.append("The LP solver is CPU-only; GPU primitives are not used by simplex.")

    enforced_time_limit = clamp_time_limit(options.time_limit, limits)

    core_model = backend_impl.build_model(plain)

    presolve_time_s = 0.0
    presolve_info: Dict[str, Any] = {}
    if options.presolve:
        core_model, presolve_info = backend_impl.presolve(
            core_model,
            {
                "feasibility_tol": options.feasibility_tol,
            },
        )
        presolve_time_s = float(presolve_info.get("presolve_time_s", 0.0))

    solver_options = {
        "solver": options.solver,
        "feasibility_tol": options.feasibility_tol,
        "optimality_tol": options.optimality_tol,
        "time_limit": enforced_time_limit,
        "iteration_limit": options.iteration_limit,
        "verbose": options.verbose,
        "seed": options.seed,
    }

    def _do_solve():
        return backend_impl.solve(core_model, solver_options, resolved_backend, resolved_device)

    try:
        raw = run_with_wall_clock_limit(_do_solve, limits.hard_wall_clock_s)
    except SolveTimeoutError:
        return Solution(
            status=SolveStatus.TIME_LIMIT,
            objective_value=None,
            variable_values={},
            solve_time_s=limits.hard_wall_clock_s,
            presolve_time_s=presolve_time_s,
            iterations=0,
            backend=resolved_backend,
            device=resolved_device,
            solver=options.solver,
            diagnostics=NumericalDiagnostics(),
            presolve_info=presolve_info,
            warnings=warnings + ["Hard wall-clock limit exceeded; solve aborted."],
        )

    diag_raw = raw.get("diagnostics", {}) or {}
    diagnostics = NumericalDiagnostics(
        max_infeasibility=diag_raw.get("max_infeasibility"),
        duality_gap=diag_raw.get("duality_gap"),
        conditioning=diag_raw.get("conditioning"),
        extra=diag_raw.get("extra", {}) if options.verbose else {},
    )

    try:
        status = SolveStatus(raw["status"])
    except ValueError:
        status = SolveStatus.ERROR

    return Solution(
        status=status,
        objective_value=raw.get("objective_value"),
        variable_values=raw.get("variable_values", {}) or {},
        solve_time_s=float(raw.get("solve_time_s", 0.0)),
        presolve_time_s=presolve_time_s,
        iterations=int(raw.get("iterations", 0)),
        backend=resolved_backend,
        device=resolved_device,
        solver=options.solver,
        diagnostics=diagnostics,
        presolve_info=presolve_info,
        constraint_residuals=raw.get("constraint_residuals", {}) or {},
        warnings=warnings,
    )
