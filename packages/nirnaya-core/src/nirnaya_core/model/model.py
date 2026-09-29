"""
The mutable model-building API.

:class:`Model` is the primary entry point most callers (including Parts
2-6) will use to describe an LP. It is intentionally mutable and
name-addressed (variables/constraints are referred to by string name) for
ergonomics; :meth:`Model.build` compiles it down into an immutable, purely
numerical :class:`~nirnaya_core.model.ProblemData` snapshot suitable for a solver.

Validation is layered:

* Each of :class:`~nirnaya_core.model.Variable`, :class:`~nirnaya_core.model.Constraint`,
  and :class:`~nirnaya_core.model.LinearObjective` validate their own internal
  consistency eagerly (in their own ``__post_init__``).
* :meth:`Model.validate` performs *cross-object* structural validation
  (duplicate names, unknown variable references, dimension consistency)
  and raises a single aggregated
  :class:`~nirnaya_core.validation.errors.ModelValidationError` listing every
  problem found, rather than failing on the first one.
* :meth:`Model.build` calls :meth:`Model.validate` internally, so a
  :class:`~nirnaya_core.model.ProblemData` can never be produced from an invalid
  model.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from nirnaya_core.matrix.sparse import SparseMatrix
from nirnaya_core.model.constraint import Constraint
from nirnaya_core.model.enums import ConstraintSense, ObjectiveSense, VariableType
from nirnaya_core.model.objective import LinearObjective
from nirnaya_core.model.problem_data import ProblemData
from nirnaya_core.model.variable import Variable
from nirnaya_core.numeric.config import DEFAULT_CONFIG, NumericalConfig
from nirnaya_core.validation.errors import (
    DuplicateNameError,
    EmptyModelError,
    InvalidObjectiveError,
    ModelValidationError,
    UnknownVariableReferenceError,
    UnsupportedVariableTypeError,
    ValidationError,
)


class Model:
    """A mutable linear-programming model builder.

    Example:
        >>> m = Model(name="toy")
        >>> m.add_variable("x", lower=0.0)
        >>> m.add_variable("y", lower=0.0)
        >>> m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 3.0, "y": 2.0})
        >>> m.add_constraint("cap", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 4.0)
        >>> data = m.build()
    """

    def __init__(self, name: str = "unnamed_model", config: Optional[NumericalConfig] = None) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must be a non-empty string")
        self.name = name
        self.config: NumericalConfig = config if config is not None else DEFAULT_CONFIG
        self._variables: Dict[str, Variable] = {}
        self._constraints: Dict[str, Constraint] = {}
        self._objective: Optional[LinearObjective] = None

    # -- building --------------------------------------------------------

    def add_variable(
        self,
        name: str,
        lower: float = 0.0,
        upper: float = math.inf,
        vtype: VariableType = VariableType.CONTINUOUS,
    ) -> Variable:
        """Add a variable and return it. Raises if the name is already used."""
        if name in self._variables:
            raise DuplicateNameError(
                f"Variable name {name!r} already exists in model {self.name!r}",
                details={"name": name, "namespace": "variable"},
            )
        var = Variable(name=name, lower=lower, upper=upper, vtype=vtype, index=len(self._variables))
        self._variables[name] = var
        return var

    def add_constraint(
        self,
        name: str,
        coefficients: Mapping[str, float],
        sense: ConstraintSense,
        rhs: float,
    ) -> Constraint:
        """Add a constraint and return it. Raises if the name is already used."""
        if name in self._constraints:
            raise DuplicateNameError(
                f"Constraint name {name!r} already exists in model {self.name!r}",
                details={"name": name, "namespace": "constraint"},
            )
        con = Constraint(
            name=name,
            coefficients=dict(coefficients),
            sense=sense,
            rhs=rhs,
            index=len(self._constraints),
        )
        self._constraints[name] = con
        return con

    def set_objective(
        self,
        sense: ObjectiveSense,
        coefficients: Mapping[str, float],
        constant: float = 0.0,
    ) -> LinearObjective:
        """Set (or replace) the model's objective function."""
        obj = LinearObjective(sense=sense, coefficients=dict(coefficients), constant=constant)
        self._objective = obj
        return obj

    def remove_variable(self, name: str) -> None:
        """Remove a variable by name and re-index remaining variables.

        Note: does NOT remove references to this variable from existing
        constraints/objective; :meth:`validate` will report those as
        :class:`~nirnaya_core.validation.errors.UnknownVariableReferenceError`
        if not cleaned up, rather than silently dropping terms.
        """
        if name not in self._variables:
            raise KeyError(f"Variable {name!r} not found in model {self.name!r}")
        del self._variables[name]
        self._reindex_variables()

    def remove_constraint(self, name: str) -> None:
        if name not in self._constraints:
            raise KeyError(f"Constraint {name!r} not found in model {self.name!r}")
        del self._constraints[name]
        self._reindex_constraints()

    def _reindex_variables(self) -> None:
        self._variables = {
            n: v.with_index(i) for i, (n, v) in enumerate(self._variables.items())
        }

    def _reindex_constraints(self) -> None:
        self._constraints = {
            n: c.with_index(i) for i, (n, c) in enumerate(self._constraints.items())
        }

    # -- accessors --------------------------------------------------------

    @property
    def variables(self) -> List[Variable]:
        return list(self._variables.values())

    @property
    def constraints(self) -> List[Constraint]:
        return list(self._constraints.values())

    @property
    def objective(self) -> Optional[LinearObjective]:
        return self._objective

    def variable_names(self) -> List[str]:
        return list(self._variables.keys())

    def constraint_names(self) -> List[str]:
        return list(self._constraints.keys())

    def get_variable(self, name: str) -> Variable:
        return self._variables[name]

    def get_constraint(self, name: str) -> Constraint:
        return self._constraints[name]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Model(name={self.name!r}, n_variables={len(self._variables)}, "
            f"n_constraints={len(self._constraints)}, "
            f"has_objective={self._objective is not None})"
        )

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Validate the full model, raising :class:`~nirnaya_core.validation.errors.ModelValidationError`
        aggregating every problem found, or returning ``None`` if the model
        is valid.

        This method never raises the individual error subclasses directly;
        it always wraps them in a single
        :class:`~nirnaya_core.validation.errors.ModelValidationError` so callers
        have one exception type to catch, with ``.errors`` for detail.
        """
        errors: List[ValidationError] = []

        if len(self._variables) == 0:
            errors.append(EmptyModelError(f"Model {self.name!r} has no variables"))

        known_names = set(self._variables.keys())

        if self._objective is None:
            errors.append(InvalidObjectiveError(f"Model {self.name!r} has no objective set"))
        else:
            unknown = sorted(set(self._objective.coefficients.keys()) - known_names)
            for var_name in unknown:
                errors.append(
                    UnknownVariableReferenceError(
                        f"Objective of model {self.name!r} references unknown variable {var_name!r}",
                        details={"variable": var_name, "context": "objective"},
                    )
                )

        for con in self._constraints.values():
            unknown = sorted(set(con.coefficients.keys()) - known_names)
            for var_name in unknown:
                errors.append(
                    UnknownVariableReferenceError(
                        f"Constraint {con.name!r} references unknown variable {var_name!r}",
                        details={"variable": var_name, "constraint": con.name, "context": "constraint"},
                    )
                )

        for var in self._variables.values():
            if var.vtype not in (VariableType.CONTINUOUS, VariableType.INTEGER, VariableType.BINARY):
                errors.append(
                    UnsupportedVariableTypeError(
                        f"Variable {var.name!r} has unsupported type {var.vtype!r}",
                        details={"variable": var.name, "vtype": str(var.vtype)},
                    )
                )

        if errors:
            raise ModelValidationError(errors, details={"model_name": self.name})

    def is_valid(self) -> bool:
        """Return True iff :meth:`validate` would raise nothing."""
        try:
            self.validate()
        except ModelValidationError:
            return False
        return True

    # -- compilation --------------------------------------------------------

    def build(self) -> ProblemData:
        """Validate this model and compile it into an immutable :class:`~nirnaya_core.model.ProblemData`
        snapshot in standard form.

        All constraints are normalized so that equality rows go into
        ``a_eq``/``b_eq`` and ``<=``/``>=`` rows go into ``a_ub``/``b_ub``
        (``>=`` rows are negated to ``<=`` form: ``a^T x >= b`` becomes
        ``-a^T x <= -b``), matching the standard form documented in
        :mod:`nirnaya_core.model.problem_data`.

        Raises:
            ModelValidationError: if :meth:`validate` finds any problem.
        """
        self.validate()
        assert self._objective is not None  # guaranteed by validate()

        var_names = tuple(self._variables.keys())
        n = len(var_names)
        index_of = {name: i for i, name in enumerate(var_names)}
        dtype = self.config.dtype

        c = np.zeros(n, dtype=dtype)
        for var_name, coeff in self._objective.coefficients.items():
            c[index_of[var_name]] = coeff

        lower = np.array([self._variables[name].lower for name in var_names], dtype=dtype)
        upper = np.array([self._variables[name].upper for name in var_names], dtype=dtype)
        var_types = tuple(self._variables[name].vtype for name in var_names)

        eq_names: List[str] = []
        ub_names: List[str] = []
        eq_rows: List[int] = []
        eq_cols: List[int] = []
        eq_vals: List[float] = []
        eq_rhs: List[float] = []
        ub_rows: List[int] = []
        ub_cols: List[int] = []
        ub_vals: List[float] = []
        ub_rhs: List[float] = []

        for con in self._constraints.values():
            if con.sense is ConstraintSense.EQ:
                row = len(eq_names)
                eq_names.append(con.name)
                for var_name, coeff in con.coefficients.items():
                    eq_rows.append(row)
                    eq_cols.append(index_of[var_name])
                    eq_vals.append(coeff)
                eq_rhs.append(con.rhs)
            elif con.sense is ConstraintSense.LE:
                row = len(ub_names)
                ub_names.append(con.name)
                for var_name, coeff in con.coefficients.items():
                    ub_rows.append(row)
                    ub_cols.append(index_of[var_name])
                    ub_vals.append(coeff)
                ub_rhs.append(con.rhs)
            else:  # GE -> negate to LE
                row = len(ub_names)
                ub_names.append(con.name)
                for var_name, coeff in con.coefficients.items():
                    ub_rows.append(row)
                    ub_cols.append(index_of[var_name])
                    ub_vals.append(-coeff)
                ub_rhs.append(-con.rhs)

        a_eq = SparseMatrix.from_triplets(len(eq_names), n, eq_rows, eq_cols, eq_vals, dtype=dtype)
        a_ub = SparseMatrix.from_triplets(len(ub_names), n, ub_rows, ub_cols, ub_vals, dtype=dtype)

        return ProblemData(
            variable_names=var_names,
            constraint_names_eq=tuple(eq_names),
            constraint_names_ub=tuple(ub_names),
            sense=self._objective.sense,
            c=c,
            objective_constant=self._objective.constant,
            a_eq=a_eq,
            b_eq=np.array(eq_rhs, dtype=dtype),
            a_ub=a_ub,
            b_ub=np.array(ub_rhs, dtype=dtype),
            lower=lower,
            upper=upper,
            var_types=var_types,
            config=self.config,
        )

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "variables": [v.to_dict() for v in self._variables.values()],
            "constraints": [c.to_dict() for c in self._constraints.values()],
            "objective": self._objective.to_dict() if self._objective is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Model":
        model = cls(name=payload["name"], config=NumericalConfig.from_dict(payload.get("config", {})))
        for var_payload in payload.get("variables", []):
            var = Variable.from_dict(var_payload)
            model.add_variable(var.name, lower=var.lower, upper=var.upper, vtype=var.vtype)
        for con_payload in payload.get("constraints", []):
            con = Constraint.from_dict(con_payload)
            model.add_constraint(con.name, con.coefficients, con.sense, con.rhs)
        if payload.get("objective") is not None:
            obj = LinearObjective.from_dict(payload["objective"])
            model.set_objective(obj.sense, obj.coefficients, obj.constant)
        return model


__all__ = ["Model"]
