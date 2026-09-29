"""
Test support for nirnaya-api.

Contains :class:`FakeSolverBackend`: a minimal stand-in for the real
nirnaya-core/presolve/solver/gpu stack, used *only* by nirnaya-api's own
test suite so the orchestration layer (validation, security limits,
options plumbing, observability fields, CLI/REST behavior) can be tested
without requiring those separate packages to be installed.

This is **not** a second solver:

  * It is not reachable through any public option, CLI flag, or REST
    parameter - the only way to install it is
    :func:`nirnaya_api._integration.set_backend`, an explicit, private
    dependency-injection hook.
  * It supports only tiny, closed-form linear problems (a single variable,
    or two variables with a simple sum constraint) sufficient to exercise
    plumbing; anything else is rejected outright rather than "solved"
    approximately.
  * It is documented and named as a fake throughout.

Use it via the :func:`fake_backend` context manager in tests.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

from . import _integration as integ
from .errors import ValidationError


class FakeCoreModel:
    """Trivial stand-in for a nirnaya-core Model object."""

    def __init__(self, model_dict: Dict[str, Any]):
        self.model_dict = model_dict


class FakeSolverBackend:
    """See module docstring. Test-only."""

    def __init__(
        self,
        gpu_devices: List[Dict[str, Any]] | None = None,
        fail_gpu_listing: str | None = None,
        force_status: str | None = None,
        artificial_delay_s: float = 0.0,
    ):
        self._gpu_devices = gpu_devices if gpu_devices is not None else []
        self._fail_gpu_listing = fail_gpu_listing
        self._force_status = force_status
        self._artificial_delay_s = artificial_delay_s

    def list_gpu_devices(self) -> List[Dict[str, Any]]:
        if self._fail_gpu_listing:
            from .errors import DependencyUnavailableError

            raise DependencyUnavailableError(self._fail_gpu_listing)
        return self._gpu_devices

    def build_model(self, model_dict: Dict[str, Any]) -> Any:
        variables = model_dict.get("variables", {})
        constraints = model_dict.get("constraints", {})
        objective = model_dict.get("objective")
        if not variables:
            raise ValidationError("model.variables must be non-empty")
        if objective is None:
            raise ValidationError("model.objective is required")
        for cname, c in constraints.items():
            for vname in c.get("coefficients", {}):
                if vname not in variables:
                    raise ValidationError(
                        f"constraint {cname!r} references unknown variable {vname!r}"
                    )
        return FakeCoreModel(model_dict)

    def presolve(self, core_model: Any, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        t0 = time.perf_counter()
        # "Presolve" here just tags the model; no reductions.
        elapsed = time.perf_counter() - t0
        return core_model, {"presolve_time_s": elapsed, "reductions": 0}

    def solve(
        self,
        core_model: Any,
        options: Dict[str, Any],
        backend: str,
        device: str,
    ) -> Dict[str, Any]:
        if self._artificial_delay_s:
            time.sleep(self._artificial_delay_s)

        model_dict = core_model.model_dict
        variables = model_dict["variables"]
        objective = model_dict["objective"]
        sense = objective.get("sense", "min")
        coeffs = objective.get("coefficients", {})

        if self._force_status:
            return {
                "status": self._force_status,
                "objective_value": None,
                "variable_values": {v: 0.0 for v in variables},
                "iterations": 0,
                "solve_time_s": self._artificial_delay_s,
                "diagnostics": {},
            }

        # Extremely small closed-form "solve": push every variable to the
        # bound that improves the (linear, separable) objective. This is
        # sufficient to exercise the plumbing (status/objective/values are
        # all populated and internally consistent) without implementing
        # anything resembling a real LP/MIP solver.
        values: Dict[str, float] = {}
        for name, spec in variables.items():
            lb = spec.get("lb", 0.0)
            ub = spec.get("ub", 1.0)
            c = coeffs.get(name, 0.0)
            if sense == "max":
                values[name] = ub if c >= 0 else lb
            else:
                values[name] = lb if c >= 0 else ub

        # Check constraints; if violated, report infeasible rather than
        # pretending to have solved it.
        for cname, c in model_dict.get("constraints", {}).items():
            lhs = sum(values.get(v, 0.0) * coef for v, coef in c["coefficients"].items())
            sense_c = c["sense"]
            rhs = c["rhs"]
            ok = {
                "<=": lhs <= rhs + 1e-6,
                ">=": lhs >= rhs - 1e-6,
                "=": abs(lhs - rhs) <= 1e-6,
            }[sense_c]
            if not ok:
                return {
                    "status": "infeasible",
                    "objective_value": None,
                    "variable_values": {},
                    "iterations": 1,
                    "solve_time_s": self._artificial_delay_s,
                    "diagnostics": {"violated_constraint": cname},
                }

        obj_value = sum(values.get(v, 0.0) * c for v, c in coeffs.items())
        return {
            "status": "optimal",
            "objective_value": obj_value,
            "variable_values": values,
            "iterations": 1,
            "solve_time_s": self._artificial_delay_s,
            "diagnostics": {"max_infeasibility": 0.0},
        }


@contextmanager
def fake_backend(**kwargs) -> Iterator[FakeSolverBackend]:
    """Context manager that installs a :class:`FakeSolverBackend` for the
    duration of the ``with`` block and restores the real backend after."""
    backend = FakeSolverBackend(**kwargs)
    integ.set_backend(backend)
    try:
        yield backend
    finally:
        integ.reset_backend()
