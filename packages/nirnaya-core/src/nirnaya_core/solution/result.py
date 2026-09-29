"""
Solver output representation.

``SolverResult`` is a pure data container: Part 1 defines its shape so that
Parts 2-6 (the actual LP engine and any GPU/native backend) all produce and
consume exactly the same structure, and so that reporting/serialization
code can be written once against a stable schema. Nothing in this module
computes a solution; constructing a "fake" optimal result from this class
without an actual solve is a misuse of the API, not something the class
does on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from nirnaya_core.status.status import SolverStatus


@dataclass(frozen=True)
class OptimizationStatistics:
    """Diagnostic statistics about a solve attempt.

    All fields are optional (``None`` when not applicable/not measured) so
    that a partial or early-terminated solve can still report a result
    without inventing values.

    Attributes:
        iterations: Number of solver iterations performed.
        solve_time_seconds: Wall-clock solve time, in seconds.
        primal_objective: Objective value at the returned primal point,
            evaluated in the *original* (not internal standard-form) sense.
        dual_objective: Objective value of the dual solution, when
            available. For an optimal LP this should match
            ``primal_objective`` up to :attr:`~nirnaya_core.numeric.NumericalConfig.optimality_tol`.
        duality_gap: ``|primal_objective - dual_objective|`` when both are
            available.
        max_primal_infeasibility: Largest constraint/bound violation at the
            returned primal point.
        max_dual_infeasibility: Largest violation of dual feasibility
            (e.g. a reduced cost with the wrong sign) at the returned dual
            point.
        solver_name: Identifier of the backend that produced this result
            (e.g. ``"nirnaya-simplex-cpu"``), for provenance/debugging.
        extra: Backend-specific diagnostics that don't fit the fields
            above. Values must be JSON-serializable.
    """

    iterations: Optional[int] = None
    solve_time_seconds: Optional[float] = None
    primal_objective: Optional[float] = None
    dual_objective: Optional[float] = None
    duality_gap: Optional[float] = None
    max_primal_infeasibility: Optional[float] = None
    max_dual_infeasibility: Optional[float] = None
    solver_name: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr_name in (
            "solve_time_seconds",
            "primal_objective",
            "dual_objective",
            "duality_gap",
            "max_primal_infeasibility",
            "max_dual_infeasibility",
        ):
            value = getattr(self, attr_name)
            if value is not None:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"OptimizationStatistics.{attr_name} must be a real number or None")
                if math.isnan(value):
                    raise ValueError(f"OptimizationStatistics.{attr_name} must not be NaN")
        if self.iterations is not None:
            if not isinstance(self.iterations, int) or isinstance(self.iterations, bool) or self.iterations < 0:
                raise ValueError("OptimizationStatistics.iterations must be a non-negative int or None")
        object.__setattr__(self, "extra", dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "solve_time_seconds": self.solve_time_seconds,
            "primal_objective": self.primal_objective,
            "dual_objective": self.dual_objective,
            "duality_gap": self.duality_gap,
            "max_primal_infeasibility": self.max_primal_infeasibility,
            "max_dual_infeasibility": self.max_dual_infeasibility,
            "solver_name": self.solver_name,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizationStatistics":
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SolverResult:
    """The complete output of a solve attempt against a single :class:`~nirnaya_core.model.Model`.

    Attributes:
        status: The terminal :class:`~nirnaya_core.status.SolverStatus`.
        variable_names: Ordered variable names, matching the index order
            used by :attr:`primal` / :attr:`reduced_costs` (i.e.
            ``variable_names[i]`` corresponds to ``primal[i]``). Empty if
            the model had no variables or the solve never reached a point
            where an ordering was fixed.
        constraint_names: Ordered constraint names, matching the index
            order used by :attr:`dual_values` / :attr:`constraint_residuals`.
        primal: Primal solution vector, one entry per variable, in the
            order given by :attr:`variable_names`. ``None`` if no primal
            point is available (e.g. status is :data:`~nirnaya_core.status.SolverStatus.INFEASIBLE`
            with no certificate returned).
        dual_values: Dual (shadow price) values, one per constraint, in the
            order given by :attr:`constraint_names`. ``None`` if unavailable.
        reduced_costs: Reduced cost per variable, in the order given by
            :attr:`variable_names`. ``None`` if unavailable.
        constraint_residuals: ``a^T x - rhs`` per constraint at
            :attr:`primal`, in the order given by :attr:`constraint_names`.
            ``None`` if :attr:`primal` is ``None``.
        statistics: Diagnostic :class:`OptimizationStatistics`.
        message: Optional human-readable explanation (e.g. which bound was
            violated for an infeasibility certificate). Not machine-parsed.
    """

    status: SolverStatus
    variable_names: Sequence[str] = field(default_factory=tuple)
    constraint_names: Sequence[str] = field(default_factory=tuple)
    primal: Optional[Sequence[float]] = None
    dual_values: Optional[Sequence[float]] = None
    reduced_costs: Optional[Sequence[float]] = None
    constraint_residuals: Optional[Sequence[float]] = None
    statistics: OptimizationStatistics = field(default_factory=OptimizationStatistics)
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SolverStatus):
            raise ValueError(f"SolverResult.status must be a SolverStatus, got {self.status!r}")

        object.__setattr__(self, "variable_names", tuple(self.variable_names))
        object.__setattr__(self, "constraint_names", tuple(self.constraint_names))

        if self.primal is not None:
            primal_t = tuple(float(v) for v in self.primal)
            if len(primal_t) != len(self.variable_names):
                raise ValueError(
                    f"SolverResult.primal has length {len(primal_t)} but "
                    f"variable_names has length {len(self.variable_names)}"
                )
            object.__setattr__(self, "primal", primal_t)

        if self.dual_values is not None:
            dual_t = tuple(float(v) for v in self.dual_values)
            if len(dual_t) != len(self.constraint_names):
                raise ValueError(
                    f"SolverResult.dual_values has length {len(dual_t)} but "
                    f"constraint_names has length {len(self.constraint_names)}"
                )
            object.__setattr__(self, "dual_values", dual_t)

        if self.reduced_costs is not None:
            rc_t = tuple(float(v) for v in self.reduced_costs)
            if len(rc_t) != len(self.variable_names):
                raise ValueError(
                    f"SolverResult.reduced_costs has length {len(rc_t)} but "
                    f"variable_names has length {len(self.variable_names)}"
                )
            object.__setattr__(self, "reduced_costs", rc_t)

        if self.constraint_residuals is not None:
            res_t = tuple(float(v) for v in self.constraint_residuals)
            if len(res_t) != len(self.constraint_names):
                raise ValueError(
                    f"SolverResult.constraint_residuals has length {len(res_t)} but "
                    f"constraint_names has length {len(self.constraint_names)}"
                )
            object.__setattr__(self, "constraint_residuals", res_t)

    def is_optimal(self) -> bool:
        return self.status.is_terminal_success

    def primal_value(self, variable_name: str) -> float:
        """Look up the primal value of a named variable.

        Raises:
            ValueError: if no primal solution is present or the name is unknown.
        """
        if self.primal is None:
            raise ValueError("SolverResult has no primal solution")
        try:
            idx = self.variable_names.index(variable_name)
        except ValueError as exc:
            raise ValueError(f"Unknown variable name {variable_name!r}") from exc
        return self.primal[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "variable_names": list(self.variable_names),
            "constraint_names": list(self.constraint_names),
            "primal": list(self.primal) if self.primal is not None else None,
            "dual_values": list(self.dual_values) if self.dual_values is not None else None,
            "reduced_costs": list(self.reduced_costs) if self.reduced_costs is not None else None,
            "constraint_residuals": (
                list(self.constraint_residuals) if self.constraint_residuals is not None else None
            ),
            "statistics": self.statistics.to_dict(),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SolverResult":
        return cls(
            status=SolverStatus(payload["status"]),
            variable_names=tuple(payload.get("variable_names", ())),
            constraint_names=tuple(payload.get("constraint_names", ())),
            primal=payload.get("primal"),
            dual_values=payload.get("dual_values"),
            reduced_costs=payload.get("reduced_costs"),
            constraint_residuals=payload.get("constraint_residuals"),
            statistics=OptimizationStatistics.from_dict(payload.get("statistics", {})),
            message=payload.get("message"),
        )


__all__ = ["SolverResult", "OptimizationStatistics"]
