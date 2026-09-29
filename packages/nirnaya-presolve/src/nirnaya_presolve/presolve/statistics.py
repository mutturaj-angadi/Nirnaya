"""Aggregate presolve statistics."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class PresolveStatistics:
    original_num_variables: int = 0
    original_num_constraints: int = 0
    reduced_num_variables: int = 0
    reduced_num_constraints: int = 0

    rounds_executed: int = 0

    variables_fixed: int = 0
    variables_eliminated_singleton_column: int = 0
    variables_eliminated_aggregation: int = 0
    variables_eliminated_empty_column: int = 0

    rows_removed_empty: int = 0
    rows_removed_singleton: int = 0
    rows_removed_redundant: int = 0
    rows_removed_duplicate: int = 0
    rows_removed_aggregation: int = 0

    bounds_tightened: int = 0
    coefficients_dropped: int = 0

    infeasible: bool = False
    infeasible_reason: str = ""
    unbounded: bool = False
    unbounded_reason: str = ""

    wall_time_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def total_variables_removed(self) -> int:
        return (
            self.variables_fixed
            + self.variables_eliminated_singleton_column
            + self.variables_eliminated_aggregation
            + self.variables_eliminated_empty_column
        )

    @property
    def total_rows_removed(self) -> int:
        return (
            self.rows_removed_empty
            + self.rows_removed_singleton
            + self.rows_removed_redundant
            + self.rows_removed_duplicate
            + self.rows_removed_aggregation
        )

    def summary(self) -> str:
        return (
            f"variables: {self.original_num_variables} -> {self.reduced_num_variables} "
            f"({self.total_variables_removed} removed) | "
            f"constraints: {self.original_num_constraints} -> {self.reduced_num_constraints} "
            f"({self.total_rows_removed} removed) | "
            f"rounds: {self.rounds_executed} | "
            f"infeasible: {self.infeasible} | unbounded: {self.unbounded}"
        )
