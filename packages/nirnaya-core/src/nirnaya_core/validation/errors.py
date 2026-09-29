"""
Concrete, machine-readable validation error types.

Each class corresponds to exactly one row in the validation matrix described
in the project brief and in ``docs/INTEGRATION_CONTRACT.md``. Keeping one
class per failure mode (rather than one generic ``ValidationError`` with a
free-text message) lets downstream callers branch on ``except`` clauses or
on ``error.code`` without parsing strings.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from nirnaya_core.exceptions import ValidationError


class DimensionMismatchError(ValidationError):
    """Two arrays/matrices that must agree in shape do not.

    Example: number of columns in ``A_eq`` does not equal the number of
    variables in the model.
    """

    code = "E_DIM_MISMATCH"


class InvalidBoundsError(ValidationError):
    """A variable's bounds are invalid, e.g. ``lower > upper``, or a bound
    is NaN.
    """

    code = "E_INVALID_BOUNDS"


class InvalidConstraintDimensionError(ValidationError):
    """A constraint references a coefficient vector/row whose length or
    index set does not match the model's variable set.
    """

    code = "E_INVALID_CONSTRAINT_DIM"


class NaNInfError(ValidationError):
    """A numerical field (objective coefficient, bound, RHS, matrix entry)
    contains ``NaN`` or a non-finite value where finiteness is required.

    Note: ``+inf``/``-inf`` are legal for variable bounds (representing
    "unbounded") and are therefore *not* flagged by this error when they
    appear as a bound; they ARE flagged when they appear as a matrix
    coefficient, RHS, or objective coefficient/constant.
    """

    code = "E_NAN_OR_INF"


class EmptyModelError(ValidationError):
    """The model has zero variables, or an operation requires at least one
    variable/constraint that is not present.
    """

    code = "E_EMPTY_MODEL"


class DuplicateNameError(ValidationError):
    """Two variables, or two constraints, share the same name.

    Names are used as stable, human-readable keys in serialization and in
    downstream reporting, so they must be unique within their namespace
    (variables and constraints are separate namespaces).
    """

    code = "E_DUPLICATE_NAME"


class InvalidObjectiveError(ValidationError):
    """The objective function is malformed: e.g. it references an unknown
    variable, has a non-finite constant, or has no declared sense.
    """

    code = "E_INVALID_OBJECTIVE"


class UnsupportedVariableTypeError(ValidationError):
    """A variable declares a :class:`~nirnaya_core.model.VariableType` that is not
    a member of the supported enumeration, or is not supported by the
    requested problem class (e.g. a solver limited to continuous LP).
    """

    code = "E_UNSUPPORTED_VARTYPE"


class UnknownVariableReferenceError(ValidationError):
    """A constraint or objective references a variable name that has not
    been added to the model.
    """

    code = "E_UNKNOWN_VARIABLE"


class InvalidConstraintSenseError(ValidationError):
    """A constraint declares a sense that is not one of the supported
    :class:`~nirnaya_core.model.ConstraintSense` members.
    """

    code = "E_INVALID_SENSE"


def _fmt(errors: "list[ValidationError]") -> str:
    return "; ".join(f"{e.code}: {e.message}" for e in errors)


class ModelValidationError(ValidationError):
    """Aggregate error raised by :meth:`~nirnaya_core.model.Model.validate` when one
    or more individual validation failures are found. Carries the full list
    of underlying errors in ``details['errors']`` (as dicts) and via the
    ``.errors`` attribute (as exception objects) for programmatic handling.
    """

    code = "E_MODEL_INVALID"

    def __init__(
        self,
        errors: "list[ValidationError]",
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.errors: "list[ValidationError]" = list(errors)
        message = f"Model failed validation with {len(self.errors)} error(s): {_fmt(self.errors)}"
        merged_details: dict[str, Any] = {"errors": [e.to_dict() for e in self.errors]}
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)


__all__ = [
    "DimensionMismatchError",
    "InvalidBoundsError",
    "InvalidConstraintDimensionError",
    "NaNInfError",
    "EmptyModelError",
    "DuplicateNameError",
    "InvalidObjectiveError",
    "UnsupportedVariableTypeError",
    "UnknownVariableReferenceError",
    "InvalidConstraintSenseError",
    "ModelValidationError",
]
