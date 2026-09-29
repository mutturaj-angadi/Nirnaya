"""
Standard-form numerical snapshot of a :class:`~nirnaya_core.model.Model`.

``ProblemData`` is the hand-off object between the modeling layer (this
package) and any solver backend (Parts 2-6, including a future
C++/CUDA/HIP kernel). It intentionally contains *only* numerical arrays
(dense name lists + sparse matrices + numpy vectors) and no references back
to the mutable :class:`~nirnaya_core.model.Model`, so that:

* it can be serialized/deserialized independently,
* it can be handed across a process or GPU-host boundary without dragging
  along Python object graphs,
* solver backends have a single, minimal, stable numerical contract to
  implement against instead of re-deriving it from the object model.

Mathematical form::

    minimize/maximize   c^T x + objective_constant
    subject to           A_eq x = b_eq
                          A_ub x <= b_ub
                          l <= x <= u
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np

from nirnaya_core.matrix.sparse import SparseMatrix
from nirnaya_core.model.enums import ObjectiveSense, VariableType
from nirnaya_core.numeric.config import DEFAULT_DTYPE, NumericalConfig


@dataclass(frozen=True)
class ProblemData:
    """Immutable standard-form LP data.

    Attributes:
        variable_names: Ordered variable names; index ``i`` here matches
            column ``i`` of every matrix and entry ``i`` of every vector.
        constraint_names_eq: Ordered names of the equality-constraint rows,
            matching the rows of :attr:`a_eq` / entries of :attr:`b_eq`.
        constraint_names_ub: Ordered names of the inequality-constraint
            rows (all normalized to ``<=``), matching the rows of
            :attr:`a_ub` / entries of :attr:`b_ub`.
        sense: Whether :attr:`c` is minimized or maximized.
        c: Dense objective coefficient vector, length ``n_variables``.
        objective_constant: Additive constant folded into the objective.
        a_eq: Sparse equality constraint matrix, shape ``(n_eq, n_variables)``.
        b_eq: Dense equality RHS vector, length ``n_eq``.
        a_ub: Sparse inequality constraint matrix (``<=`` form), shape
            ``(n_ub, n_variables)``.
        b_ub: Dense inequality RHS vector, length ``n_ub``.
        lower: Dense variable lower bounds, length ``n_variables``.
        upper: Dense variable upper bounds, length ``n_variables``.
        var_types: Tuple of :class:`~nirnaya_core.model.VariableType`, length
            ``n_variables``.
        config: The :class:`~nirnaya_core.numeric.NumericalConfig` this snapshot
            was built under (governs the dtype of all arrays above and the
            tolerances a solver consuming this data should honor).
    """

    variable_names: Tuple[str, ...]
    constraint_names_eq: Tuple[str, ...]
    constraint_names_ub: Tuple[str, ...]
    sense: ObjectiveSense
    c: np.ndarray
    objective_constant: float
    a_eq: SparseMatrix
    b_eq: np.ndarray
    a_ub: SparseMatrix
    b_ub: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    var_types: Tuple[VariableType, ...]
    config: NumericalConfig

    def __post_init__(self) -> None:
        n = len(self.variable_names)
        errors: list[str] = []

        if self.c.shape != (n,):
            errors.append(f"c has shape {self.c.shape}, expected ({n},)")
        if self.lower.shape != (n,):
            errors.append(f"lower has shape {self.lower.shape}, expected ({n},)")
        if self.upper.shape != (n,):
            errors.append(f"upper has shape {self.upper.shape}, expected ({n},)")
        if len(self.var_types) != n:
            errors.append(f"var_types has length {len(self.var_types)}, expected {n}")

        n_eq = len(self.constraint_names_eq)
        if self.a_eq.shape != (n_eq, n):
            errors.append(f"a_eq has shape {self.a_eq.shape}, expected ({n_eq}, {n})")
        if self.b_eq.shape != (n_eq,):
            errors.append(f"b_eq has shape {self.b_eq.shape}, expected ({n_eq},)")

        n_ub = len(self.constraint_names_ub)
        if self.a_ub.shape != (n_ub, n):
            errors.append(f"a_ub has shape {self.a_ub.shape}, expected ({n_ub}, {n})")
        if self.b_ub.shape != (n_ub,):
            errors.append(f"b_ub has shape {self.b_ub.shape}, expected ({n_ub},)")

        if errors:
            raise ValueError("Invalid ProblemData: " + "; ".join(errors))

        # Enforce the canonical dtype/immutability contract: arrays handed
        # to solver backends must not be mutated out from under this
        # snapshot, and must use the configured dtype.
        for name in ("c", "b_eq", "b_ub", "lower", "upper"):
            arr = getattr(self, name)
            arr = np.asarray(arr, dtype=self.config.dtype)
            arr = arr.copy()
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    @property
    def n_eq(self) -> int:
        return len(self.constraint_names_eq)

    @property
    def n_ub(self) -> int:
        return len(self.constraint_names_ub)

    def variable_index(self, name: str) -> int:
        return self.variable_names.index(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_names": list(self.variable_names),
            "constraint_names_eq": list(self.constraint_names_eq),
            "constraint_names_ub": list(self.constraint_names_ub),
            "sense": self.sense.value,
            "c": self.c.tolist(),
            "objective_constant": self.objective_constant,
            "a_eq": self.a_eq.to_dict(),
            "b_eq": self.b_eq.tolist(),
            "a_ub": self.a_ub.to_dict(),
            "b_ub": self.b_ub.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "var_types": [vt.value for vt in self.var_types],
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProblemData":
        config = NumericalConfig.from_dict(payload.get("config", {}))
        dtype = config.dtype
        return cls(
            variable_names=tuple(payload["variable_names"]),
            constraint_names_eq=tuple(payload["constraint_names_eq"]),
            constraint_names_ub=tuple(payload["constraint_names_ub"]),
            sense=ObjectiveSense(payload["sense"]),
            c=np.asarray(payload["c"], dtype=dtype),
            objective_constant=float(payload.get("objective_constant", 0.0)),
            a_eq=SparseMatrix.from_dict(payload["a_eq"]),
            b_eq=np.asarray(payload["b_eq"], dtype=dtype),
            a_ub=SparseMatrix.from_dict(payload["a_ub"]),
            b_ub=np.asarray(payload["b_ub"], dtype=dtype),
            lower=np.asarray(payload["lower"], dtype=dtype),
            upper=np.asarray(payload["upper"], dtype=dtype),
            var_types=tuple(VariableType(v) for v in payload["var_types"]),
            config=config,
        )


__all__ = ["ProblemData"]
