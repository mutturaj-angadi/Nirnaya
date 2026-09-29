"""Decision variable representation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from nirnaya_core.model.enums import VariableType


@dataclass(frozen=True)
class Variable:
    """A single decision variable ``l <= x <= u``.

    Instances are immutable value objects. Structural validity (name
    non-empty, bounds well-ordered, type recognized) is enforced eagerly in
    ``__post_init__`` so that a :class:`Variable` can never exist in an
    invalid state; *cross-model* validation (duplicate names, references
    from constraints) happens in :mod:`nirnaya_core.validation`.

    Attributes:
        name: Unique (within a model) human-readable identifier.
        lower: Lower bound. ``-math.inf`` means unbounded below.
        upper: Upper bound. ``math.inf`` means unbounded above.
        vtype: The variable's declared :class:`~nirnaya_core.model.VariableType`.
        index: Zero-based position assigned by :class:`~nirnaya_core.model.Model`
            when the variable is added. ``None`` until then.
    """

    name: str
    lower: float = -math.inf
    upper: float = math.inf
    vtype: VariableType = VariableType.CONTINUOUS
    index: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Variable name must be a non-empty string")
        for bound_name, bound_value in (("lower", self.lower), ("upper", self.upper)):
            if not isinstance(bound_value, (int, float)) or isinstance(bound_value, bool):
                raise ValueError(f"Variable {bound_name!r} bound must be a real number")
            if math.isnan(bound_value):
                raise ValueError(f"Variable {self.name!r} has NaN {bound_name} bound")
        if self.lower > self.upper:
            raise ValueError(
                f"Variable {self.name!r} has lower bound {self.lower} > upper bound {self.upper}"
            )
        if not isinstance(self.vtype, VariableType):
            raise ValueError(f"Variable {self.name!r} has unsupported vtype {self.vtype!r}")
        if self.vtype == VariableType.BINARY and (self.lower not in (0, -math.inf) or self.upper not in (1, math.inf)):
            # Binary variables conventionally imply [0, 1]; we don't force it here
            # (callers may have already set explicit [0, 1] bounds), but we do
            # not silently override user-provided bounds either. This branch is
            # intentionally a no-op placeholder for future MIP validation hooks.
            pass

    def is_free(self) -> bool:
        """True if the variable has no finite lower or upper bound."""
        return math.isinf(self.lower) and self.lower < 0 and math.isinf(self.upper) and self.upper > 0

    def is_fixed(self) -> bool:
        """True if lower == upper (a fixed variable)."""
        return self.lower == self.upper

    def with_index(self, index: int) -> "Variable":
        """Return a copy of this variable with ``index`` set."""
        return Variable(name=self.name, lower=self.lower, upper=self.upper, vtype=self.vtype, index=index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "vtype": self.vtype.value,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Variable":
        return cls(
            name=payload["name"],
            lower=float(payload.get("lower", -math.inf)),
            upper=float(payload.get("upper", math.inf)),
            vtype=VariableType(payload.get("vtype", VariableType.CONTINUOUS.value)),
            index=payload.get("index"),
        )


__all__ = ["Variable"]
