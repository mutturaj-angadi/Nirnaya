"""Model and numerical validation."""

from nirnaya_core.validation.errors import (
    DimensionMismatchError,
    DuplicateNameError,
    EmptyModelError,
    InvalidBoundsError,
    InvalidConstraintDimensionError,
    InvalidConstraintSenseError,
    InvalidObjectiveError,
    ModelValidationError,
    NaNInfError,
    UnknownVariableReferenceError,
    UnsupportedVariableTypeError,
)
from nirnaya_core.validation.validator import is_problem_data_valid, validate_problem_data

__all__ = [
    "ModelValidationError",
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
    "validate_problem_data",
    "is_problem_data_valid",
]
