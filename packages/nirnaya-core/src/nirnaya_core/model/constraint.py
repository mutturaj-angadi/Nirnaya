"""Linear constraint representation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from nirnaya_core.model.enums import ConstraintSense


@dataclass(frozen=True)
class Constraint:
    """A single linear constraint ``a^T x {<=, ==, >=} rhs``.

    Like :class:`~nirnaya_core.model.Variable`, coefficients are stored sparsely as
    a ``{variable_name: coefficient}`` mapping.

    Attributes:
        name: Unique (within a model) human-readable identifier.
        coefficients: Mapping from variable name to its (finite) coefficient
            in this constraint's row. Variables absent from this mapping
            have an implicit coefficient of 0.
        sense: The relational operator.
        rhs: The (finite) right-hand side.
        index: Zero-based row position assigned by
            :class:`~nirnaya_core.model.Model` when the constraint is added.
            ``None`` until then.
    """

    name: str
    coefficients: Mapping[str, float] = field(default_factory=dict)
    sense: ConstraintSense = ConstraintSense.LE
    rhs: float = 0.0
    index: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Constraint name must be a non-empty string")
        if not isinstance(self.sense, ConstraintSense):
            raise ValueError(f"Constraint {self.name!r} has unsupported sense {self.sense!r}")
        if not isinstance(self.rhs, (int, float)) or isinstance(self.rhs, bool):
            raise ValueError(f"Constraint {self.name!r} rhs must be a real number")
        if not math.isfinite(self.rhs):
            raise ValueError(f"Constraint {self.name!r} has non-finite rhs {self.rhs!r}")
        if len(self.coefficients) == 0:
            raise ValueError(f"Constraint {self.name!r} has no coefficients (empty row)")
        cleaned: dict[str, float] = {}
        for var_name, coeff in dict(self.coefficients).items():
            if not isinstance(var_name, str) or not var_name.strip():
                raise ValueError(f"Constraint {self.name!r} coefficient key must be a non-empty string")
            if not isinstance(coeff, (int, float)) or isinstance(coeff, bool):
                raise ValueError(f"Constraint {self.name!r} coefficient for {var_name!r} must be a real number")
            if not math.isfinite(coeff):
                raise ValueError(
                    f"Constraint {self.name!r} coefficient for {var_name!r} is non-finite: {coeff!r}"
                )
            cleaned[var_name] = float(coeff)
        object.__setattr__(self, "coefficients", cleaned)

    def coefficient_for(self, variable_name: str) -> float:
        return self.coefficients.get(variable_name, 0.0)

    def with_index(self, index: int) -> "Constraint":
        return Constraint(
            name=self.name,
            coefficients=dict(self.coefficients),
            sense=self.sense,
            rhs=self.rhs,
            index=index,
        )

    def residual(self, values: Mapping[str, float]) -> float:
        """Return ``a^T x - rhs`` for the given ``{name: value}`` mapping.

        For :data:`~nirnaya_core.model.ConstraintSense.LE`/:data:`GE` this is the
        signed slack/violation; for :data:`~nirnaya_core.model.ConstraintSense.EQ`
        it is the equality residual. Sign convention: zero means exactly
        satisfied; for LE, positive means violated; for GE, negative means
        violated.
        """
        lhs = sum(coeff * values.get(name, 0.0) for name, coeff in self.coefficients.items())
        return lhs - self.rhs

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coefficients": dict(sorted(self.coefficients.items())),
            "sense": self.sense.value,
            "rhs": self.rhs,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Constraint":
        return cls(
            name=payload["name"],
            coefficients=dict(payload.get("coefficients", {})),
            sense=ConstraintSense(payload["sense"]),
            rhs=float(payload.get("rhs", 0.0)),
            index=payload.get("index"),
        )


__all__ = ["Constraint"]
