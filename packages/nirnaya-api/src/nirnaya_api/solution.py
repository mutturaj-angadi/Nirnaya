"""Solution and observability types.

``Solution`` is the single structured result object returned by the Python
API, serialized by the REST API, and printed by the CLI. It carries both
the numerical answer and the observability fields required by the spec:
solve time, presolve time, iterations, backend, device, objective, status,
and numerical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    NUMERICAL_ERROR = "numerical_error"
    ERROR = "error"


@dataclass
class NumericalDiagnostics:
    #: Largest constraint/bound violation observed in the returned point.
    max_infeasibility: Optional[float] = None
    #: Duality gap, if applicable to the solver family used.
    duality_gap: Optional[float] = None
    #: Condition-number style indicator, if the solver reports one.
    conditioning: Optional[float] = None
    #: Free-form extra diagnostics reported by nirnaya-solver (only
    #: populated when ``verbose=True``).
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "max_infeasibility": self.max_infeasibility,
            "duality_gap": self.duality_gap,
            "conditioning": self.conditioning,
            "extra": self.extra,
        }


@dataclass
class Solution:
    status: SolveStatus
    objective_value: Optional[float]
    variable_values: Dict[str, float]

    # --- observability -----------------------------------------------
    solve_time_s: float
    presolve_time_s: float
    iterations: int
    backend: str
    device: str
    solver: str
    diagnostics: NumericalDiagnostics = field(default_factory=NumericalDiagnostics)
    presolve_info: Dict[str, Any] = field(default_factory=dict)
    constraint_residuals: Dict[str, float] = field(default_factory=dict)

    #: Non-fatal notices (e.g. "fell back to CPU: no GPU device found").
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "status": self.status.value,
            "objective_value": self.objective_value,
            "variable_values": self.variable_values,
            "observability": {
                "solve_time_s": self.solve_time_s,
                "presolve_time_s": self.presolve_time_s,
                "iterations": self.iterations,
                "backend": self.backend,
                "device": self.device,
                "solver": self.solver,
            },
            "diagnostics": self.diagnostics.to_dict(),
            "warnings": self.warnings,
            "presolve": self.presolve_info,
            "constraint_residuals": self.constraint_residuals,
        }
        # Compatibility fields consumed by the originally supplied dashboard.
        payload.update({
            "objective": self.objective_value,
            "variables": self.variable_values,
            "iterations": self.iterations,
            "solve_time_ms": self.solve_time_s * 1000.0,
            "presolve_time_ms": self.presolve_time_s * 1000.0,
            "numerical": {"max_residual": self.diagnostics.max_infeasibility,
                          "duality_gap": self.diagnostics.duality_gap,
                          "condition_estimate": self.diagnostics.conditioning},
        })
        return payload

    @property
    def is_optimal(self) -> bool:
        return self.status == SolveStatus.OPTIMAL
