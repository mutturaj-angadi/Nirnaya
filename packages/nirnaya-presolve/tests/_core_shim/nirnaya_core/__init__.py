"""
TEST-ONLY STAND-IN for `nirnaya-core`.

This is NOT part of the nirnaya-presolve deliverable. It exists solely so
that nirnaya-presolve's test suite can run before the real nirnaya-core
package (Part 1 of Nirnaya) is available for integration.

It implements exactly the public surface documented in
docs/ASSUMED_CORE_API.md. When the real nirnaya-core is available:

    1. Delete this directory.
    2. `pip install nirnaya-core` (or add it as a proper dependency).
    3. Re-run the test suite against the real package.
    4. Reconcile any API differences in `nirnaya_presolve/adapter.py` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Sequence


class Sense(Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


class ObjectiveSense(Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"


class SolveStatus(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"


@dataclass(frozen=True)
class Variable:
    name: str
    lower: float = float("-inf")
    upper: float = float("inf")
    is_integer: bool = False

    def __post_init__(self):
        if self.lower > self.upper:
            raise ValueError(
                f"Variable {self.name!r}: lower ({self.lower}) > upper ({self.upper})"
            )


@dataclass(frozen=True)
class Constraint:
    name: str
    coefficients: Mapping[str, float]
    sense: Sense
    rhs: float

    def __post_init__(self):
        # normalize to a plain immutable-ish dict, drop exact zeros
        cleaned = {k: v for k, v in self.coefficients.items() if v != 0.0}
        object.__setattr__(self, "coefficients", dict(cleaned))


class LPModel:
    def __init__(
        self,
        variables: Sequence[Variable],
        constraints: Sequence[Constraint],
        objective: Mapping[str, float],
        objective_sense: ObjectiveSense,
        objective_offset: float = 0.0,
        name: str = "",
    ):
        self._variables: List[Variable] = list(variables)
        self._constraints: List[Constraint] = list(constraints)
        self.objective: Dict[str, float] = {
            k: v for k, v in dict(objective).items() if v != 0.0
        }
        self.objective_sense = objective_sense
        self.objective_offset = float(objective_offset)
        self.name = name

        seen = set()
        for v in self._variables:
            if v.name in seen:
                raise ValueError(f"Duplicate variable name: {v.name!r}")
            seen.add(v.name)

        seen_c = set()
        for c in self._constraints:
            if c.name in seen_c:
                raise ValueError(f"Duplicate constraint name: {c.name!r}")
            seen_c.add(c.name)
            for var_name in c.coefficients:
                if var_name not in seen:
                    raise ValueError(
                        f"Constraint {c.name!r} references unknown variable {var_name!r}"
                    )
        for var_name in self.objective:
            if var_name not in seen:
                raise ValueError(
                    f"Objective references unknown variable {var_name!r}"
                )

        self._var_index = {v.name: v for v in self._variables}
        self._con_index = {c.name: c for c in self._constraints}

    @property
    def variables(self) -> Sequence[Variable]:
        return tuple(self._variables)

    @property
    def constraints(self) -> Sequence[Constraint]:
        return tuple(self._constraints)

    def variable_names(self) -> List[str]:
        return [v.name for v in self._variables]

    def constraint_names(self) -> List[str]:
        return [c.name for c in self._constraints]

    def get_variable(self, name: str) -> Variable:
        return self._var_index[name]

    def get_constraint(self, name: str) -> Constraint:
        return self._con_index[name]

    def __repr__(self):
        return (
            f"LPModel(name={self.name!r}, vars={len(self._variables)}, "
            f"constraints={len(self._constraints)})"
        )


class Solution:
    def __init__(
        self,
        values: Mapping[str, float],
        objective_value: float,
        status: SolveStatus,
    ):
        self.values: Dict[str, float] = dict(values)
        self.objective_value = float(objective_value)
        self.status = status

    def __repr__(self):
        return (
            f"Solution(status={self.status}, objective={self.objective_value}, "
            f"n_values={len(self.values)})"
        )
