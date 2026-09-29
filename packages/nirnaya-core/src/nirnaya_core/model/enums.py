"""Enumerations shared by the model layer.

These enums are part of the public API contract (see
``docs/INTEGRATION_CONTRACT.md``): downstream modules may switch on their
members, and new members will only ever be *added*, never renamed or
removed, within a major version.
"""

from __future__ import annotations

import enum


class VariableType(str, enum.Enum):
    """The declared type of a decision variable.

    Part 1 only *represents* these; only :data:`CONTINUOUS` is guaranteed
    to be solvable by the LP engine delivered in this part. ``INTEGER`` and
    ``BINARY`` are modeled now so that Parts 2-6 (MIP extensions) have a
    stable type to target without a breaking schema change later.
    """

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class ConstraintSense(str, enum.Enum):
    """The relational operator of a linear constraint ``a^T x {<=, ==, >=} b``."""

    LE = "<="
    EQ = "=="
    GE = ">="

    @property
    def is_equality(self) -> bool:
        return self is ConstraintSense.EQ


class ObjectiveSense(str, enum.Enum):
    """Whether the objective is minimized or maximized."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


__all__ = ["VariableType", "ConstraintSense", "ObjectiveSense"]
