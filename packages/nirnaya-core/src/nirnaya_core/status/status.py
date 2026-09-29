"""Solver status enumeration.

This is a pure data/vocabulary module: Part 1 ships no solver, so nothing
here ever *produces* a status. It exists so that Parts 2-6 (the actual
simplex/interior-point/GPU solver, and any reporting layer built on top of
it) agree on one canonical, stable vocabulary for "what happened", instead
of each solver backend inventing its own strings.
"""

from __future__ import annotations

import enum


class SolverStatus(str, enum.Enum):
    """Canonical outcome of a solve attempt.

    Members:
        NOT_SOLVED: No solve has been attempted yet (the default status of
            a freshly constructed :class:`~nirnaya_core.solution.SolverResult`).
        OPTIMAL: A primal-dual optimal solution was found within tolerance.
        INFEASIBLE: The problem was proven to have no feasible point.
        UNBOUNDED: The problem was proven to have unbounded objective value
            in the direction of optimization.
        ITERATION_LIMIT: The solver stopped because it hit
            :attr:`~nirnaya_core.numeric.NumericalConfig.max_iterations` before
            reaching a conclusive status.
        TIME_LIMIT: The solver stopped because it hit
            :attr:`~nirnaya_core.numeric.NumericalConfig.time_limit_seconds`
            before reaching a conclusive status.
        NUMERICAL_ERROR: The solver stopped due to a numerical failure
            (e.g. singular basis, non-finite iterate) it could not recover
            from.
        INFEASIBLE_OR_UNBOUNDED: The solver detected that the problem is
            either infeasible or unbounded but could not (cheaply)
            distinguish which; callers should re-solve with a feasibility
            check if the distinction matters.
        UNKNOWN: A terminal state was reached that does not map to any of
            the above (reserved for forward compatibility).
    """

    NOT_SOLVED = "not_solved"
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    NUMERICAL_ERROR = "numerical_error"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    UNKNOWN = "unknown"

    @property
    def is_terminal_success(self) -> bool:
        """True only for :data:`OPTIMAL`."""
        return self is SolverStatus.OPTIMAL

    @property
    def is_conclusive(self) -> bool:
        """True if this status represents a proven mathematical conclusion
        (optimal, infeasible, or unbounded) rather than a resource limit or
        error.
        """
        return self in (
            SolverStatus.OPTIMAL,
            SolverStatus.INFEASIBLE,
            SolverStatus.UNBOUNDED,
        )


__all__ = ["SolverStatus"]
