"""Linear objective function representation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from nirnaya_core.model.enums import ObjectiveSense


@dataclass(frozen=True)
class LinearObjective:
    """A linear objective ``sense  c^T x + constant``.

    Coefficients are stored sparsely as a ``{variable_name: coefficient}``
    mapping so that a model with many variables but a sparse objective does
    not pay for a dense vector until :meth:`~nirnaya_core.model.Model.build` is
    called.

    Attributes:
        sense: :data:`~nirnaya_core.model.ObjectiveSense.MINIMIZE` or
            :data:`~nirnaya_core.model.ObjectiveSense.MAXIMIZE`.
        coefficients: Mapping from variable name to its (finite) objective
            coefficient. Variables absent from this mapping have an
            implicit coefficient of 0.
        constant: A finite additive constant folded into the objective
            value (does not affect the optimal ``x``).
    """

    sense: ObjectiveSense
    coefficients: Mapping[str, float] = field(default_factory=dict)
    constant: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.sense, ObjectiveSense):
            raise ValueError(f"LinearObjective.sense must be an ObjectiveSense, got {self.sense!r}")
        if not isinstance(self.constant, (int, float)) or isinstance(self.constant, bool):
            raise ValueError("LinearObjective.constant must be a real number")
        if not math.isfinite(self.constant):
            raise ValueError(f"LinearObjective.constant must be finite, got {self.constant!r}")
        cleaned: dict[str, float] = {}
        for var_name, coeff in dict(self.coefficients).items():
            if not isinstance(var_name, str) or not var_name.strip():
                raise ValueError(f"LinearObjective coefficient key must be a non-empty string, got {var_name!r}")
            if not isinstance(coeff, (int, float)) or isinstance(coeff, bool):
                raise ValueError(f"LinearObjective coefficient for {var_name!r} must be a real number")
            if not math.isfinite(coeff):
                raise ValueError(f"LinearObjective coefficient for {var_name!r} must be finite, got {coeff!r}")
            cleaned[var_name] = float(coeff)
        # Freeze via object.__setattr__ since the dataclass is frozen.
        object.__setattr__(self, "coefficients", cleaned)

    def coefficient_for(self, variable_name: str) -> float:
        """Return the coefficient for ``variable_name`` (0.0 if absent)."""
        return self.coefficients.get(variable_name, 0.0)

    def evaluate(self, values: Mapping[str, float]) -> float:
        """Evaluate ``c^T x + constant`` given a ``{name: value}`` mapping.

        Variables referenced by this objective but missing from ``values``
        default to 0.0.
        """
        total = self.constant
        for var_name, coeff in self.coefficients.items():
            total += coeff * values.get(var_name, 0.0)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "sense": self.sense.value,
            "coefficients": dict(sorted(self.coefficients.items())),
            "constant": self.constant,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearObjective":
        return cls(
            sense=ObjectiveSense(payload["sense"]),
            coefficients=dict(payload.get("coefficients", {})),
            constant=float(payload.get("constant", 0.0)),
        )


__all__ = ["LinearObjective"]
