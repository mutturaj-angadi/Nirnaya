"""Unit tests for the exception hierarchy."""

from nirnaya_core.exceptions import ConfigurationError, NirnayaError, ValidationError
from nirnaya_core.validation.errors import (
    DimensionMismatchError,
    DuplicateNameError,
    EmptyModelError,
    InvalidBoundsError,
    ModelValidationError,
    NaNInfError,
)


class TestBaseHierarchy:
    def test_validation_error_is_nirnaya_error(self):
        assert issubclass(ValidationError, NirnayaError)

    def test_configuration_error_is_nirnaya_error(self):
        assert issubclass(ConfigurationError, NirnayaError)

    def test_specific_errors_are_validation_errors(self):
        for cls in (DimensionMismatchError, InvalidBoundsError, EmptyModelError, DuplicateNameError, NaNInfError):
            assert issubclass(cls, ValidationError)


class TestCodesAreStableAndDistinct:
    def test_each_error_class_has_unique_code(self):
        classes = [
            DimensionMismatchError,
            InvalidBoundsError,
            EmptyModelError,
            DuplicateNameError,
            NaNInfError,
        ]
        codes = [c.code for c in classes]
        assert len(codes) == len(set(codes))

    def test_code_is_accessible_without_raising(self):
        assert DimensionMismatchError.code == "E_DIM_MISMATCH"


class TestErrorSerialization:
    def test_to_dict_contains_code_and_message(self):
        err = InvalidBoundsError("bad bounds", details={"variable": "x"})
        d = err.to_dict()
        assert d["code"] == "E_INVALID_BOUNDS"
        assert d["message"] == "bad bounds"
        assert d["details"] == {"variable": "x"}

    def test_aggregate_error_includes_all_sub_errors(self):
        sub_errors = [
            EmptyModelError("no vars"),
            DuplicateNameError("dup name"),
        ]
        agg = ModelValidationError(sub_errors)
        assert len(agg.errors) == 2
        assert agg.details["errors"][0]["code"] == "E_EMPTY_MODEL"
        assert agg.details["errors"][1]["code"] == "E_DUPLICATE_NAME"

    def test_str_includes_code(self):
        err = EmptyModelError("no vars")
        assert "E_EMPTY_MODEL" in str(err)
