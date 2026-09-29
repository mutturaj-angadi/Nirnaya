"""Translation boundary from the API JSON schema to the real Nirnaya stack."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np

from .errors import DependencyUnavailableError, ValidationError


class SolverBackend(Protocol):
    def list_gpu_devices(self) -> List[Dict[str, Any]]: ...
    def build_model(self, model_dict: Dict[str, Any]) -> Any: ...
    def presolve(self, core_model: Any, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]: ...
    def solve(self, core_model: Any, options: Dict[str, Any], backend: str, device: str) -> Dict[str, Any]: ...


class RealSolverBackend:
    """Uses only core, presolve, solver, and the public GPU registry APIs."""

    def list_gpu_devices(self) -> List[Dict[str, Any]]:
        try:
            import nirnaya_gpu
            report = nirnaya_gpu.discover_all()
            return [{"id": f"{d.backend_name}:{d.device_id}", "name": d.model_name, "available": True,
                     "detail": {"device_type": d.device_type.value}}
                    for d in report.devices if d.is_gpu]
        except ImportError:
            return []

    def build_model(self, payload: Dict[str, Any]) -> Any:
        try:
            from nirnaya_core import ConstraintSense, Model, ObjectiveSense, VariableType
            model = Model(name=payload.get("name") or "api_model")
            for name, spec in payload["variables"].items():
                model.add_variable(name, lower=spec.get("lb", 0.0),
                    upper=spec.get("ub", float("inf")), vtype=VariableType(spec.get("kind", "continuous")))
            senses = {"<=": ConstraintSense.LE, ">=": ConstraintSense.GE,
                      "=": ConstraintSense.EQ, "==": ConstraintSense.EQ}
            for name, spec in payload.get("constraints", {}).items():
                model.add_constraint(name, spec["coefficients"], senses[spec["sense"]], spec["rhs"])
            obj = payload["objective"]
            sense = ObjectiveSense.MINIMIZE if obj["sense"] == "min" else ObjectiveSense.MAXIMIZE
            model.set_objective(sense, obj.get("coefficients", {}), constant=obj.get("offset", 0.0))
            return model
        except (ValueError, KeyError, TypeError) as exc:
            raise ValidationError(f"Invalid model: {exc}") from exc
        except ImportError as exc:
            raise DependencyUnavailableError("nirnaya-core is not installed") from exc

    def presolve(self, model: Any, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        try:
            from nirnaya_presolve import Presolver, PresolveTolerances
        except ImportError as exc:
            raise DependencyUnavailableError("nirnaya-presolve is not installed") from exc
        started = time.perf_counter()
        result = Presolver(tolerances=PresolveTolerances(
            feasibility_tol=options.get("feasibility_tol", 1e-8))).run(model)
        original_data = model.build()
        statistics = result.statistics.as_dict()
        original_nnz = original_data.a_eq.nnz + original_data.a_ub.nnz
        if result.reduced_model is None:
            reduced_nnz = None
        elif not result.reduced_model.variables:
            # A valid presolve result may fix/eliminate every original
            # variable. Model.build() intentionally rejects user-created
            # empty models, so do not revalidate this internal constant LP
            # merely to count its (necessarily zero) matrix entries.
            reduced_nnz = 0
        else:
            reduced_data = result.reduced_model.build()
            reduced_nnz = reduced_data.a_eq.nnz + reduced_data.a_ub.nnz
        statistics.update({"original_num_nonzeros": original_nnz,
            "reduced_num_nonzeros": reduced_nnz,
            "nonzero_reduction_fraction": ((original_nnz - reduced_nnz) / original_nnz
                if original_nnz else 0.0) if reduced_nnz is not None else None})
        return (model, result), {"presolve_time_s": time.perf_counter()-started,
            "infeasible": result.infeasible, "unbounded": result.unbounded,
            "summary": result.summary(), "statistics": statistics}

    def solve(self, prepared: Any, options: Dict[str, Any], backend: str, device: str) -> Dict[str, Any]:
        try:
            from nirnaya_core import SolverStatus
            from nirnaya_solver import solve as solve_lp
        except ImportError as exc:
            raise DependencyUnavailableError("nirnaya-solver is not installed") from exc
        model, presolve_result = prepared if isinstance(prepared, tuple) else (prepared, None)
        if options.get("solver", "auto") not in ("auto", "simplex"):
            raise ValidationError("This release implements only the CPU simplex algorithm.")
        if presolve_result is not None and presolve_result.infeasible:
            return {"status": "infeasible", "iterations": 0, "solve_time_s": 0.0,
                    "diagnostics": {"extra": {"presolve": presolve_result.infeasible_reason}}}
        if presolve_result is not None and presolve_result.unbounded:
            return {"status": "unbounded", "iterations": 0, "solve_time_s": 0.0,
                    "diagnostics": {"extra": {"presolve": presolve_result.unbounded_reason}}}
        target = presolve_result.reduced_model if presolve_result is not None else model
        target.config = target.config.with_overrides(
            feasibility_tol=options.get("feasibility_tol", target.config.feasibility_tol),
            optimality_tol=options.get("optimality_tol", target.config.optimality_tol),
            max_iterations=options.get("iteration_limit") or target.config.max_iterations,
            time_limit_seconds=options.get("time_limit"),
        )
        if backend == "gpu":
            raise DependencyUnavailableError("The current LP algorithm is CPU-only; GPU kernels are primitives only.")
        result = solve_lp(target, backend="cpu", device=device,
            time_limit=options.get("time_limit"), iteration_limit=options.get("iteration_limit"))
        verified_residuals = {}
        if presolve_result is not None and result.status is SolverStatus.OPTIMAL:
            result = presolve_result.recover_solution(result)
            # Presolve recovery is checked independently in original model
            # coordinates before the API reports success.
            original = model.build()
            values_by_name = dict(zip(result.variable_names, result.primal or ()))
            x = np.asarray([values_by_name[n] for n in original.variable_names], dtype=original.config.dtype)
            eq_error = np.max(np.abs(original.a_eq.matvec(x) - original.b_eq), initial=0.0)
            ub_error = np.max(np.maximum(original.a_ub.matvec(x) - original.b_ub, 0), initial=0.0)
            lo_error = np.max(np.maximum(original.lower - x, 0), initial=0.0)
            hi_error = np.max(np.maximum(x - original.upper, 0), initial=0.0)
            original_error = float(max(eq_error, ub_error, lo_error, hi_error))
            constraint_error = float(max(eq_error, ub_error))
            bound_error = float(max(lo_error, hi_error))
            if (not np.isfinite(original_error)
                    or constraint_error > original.config.feasibility_tol
                    or bound_error > original.config.bound_tol):
                from dataclasses import replace
                result = replace(result, status=SolverStatus.NUMERICAL_ERROR, primal=None,
                    statistics=replace(result.statistics, primal_objective=None,
                        max_primal_infeasibility=original_error),
                    message="Original-space feasibility verification after presolve recovery failed.")
            else:
                residuals = np.concatenate((original.a_eq.matvec(x) - original.b_eq,
                                            original.a_ub.matvec(x) - original.b_ub))
                residual_names = original.constraint_names_eq + original.constraint_names_ub
                verified_residuals = {name: float(value) for name, value in zip(residual_names, residuals)}
        elif result.status is SolverStatus.OPTIMAL and result.constraint_residuals is not None:
            verified_residuals = {name: float(value) for name, value in
                                  zip(result.constraint_names, result.constraint_residuals)}
        values = dict(zip(result.variable_names, result.primal or ()))
        stats = result.statistics
        return {"status": result.status.value, "objective_value": stats.primal_objective,
            "variable_values": values, "iterations": stats.iterations or 0,
            "solve_time_s": stats.solve_time_seconds or 0.0,
            "diagnostics": {"max_infeasibility": stats.max_primal_infeasibility,
                "duality_gap": stats.duality_gap,
                "extra": {"solver": stats.solver_name, **dict(stats.extra)}},
            "constraint_residuals": verified_residuals}


_active_backend: SolverBackend = RealSolverBackend()


def get_backend() -> SolverBackend:
    return _active_backend


def set_backend(backend: SolverBackend) -> None:
    global _active_backend
    _active_backend = backend


def reset_backend() -> None:
    set_backend(RealSolverBackend())


def try_list_gpu_devices():
    try:
        return get_backend().list_gpu_devices(), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
