import pytest
from pydantic import ValidationError as PydanticValidationError

from nirnaya_api.schema import ModelSchema
from nirnaya_api.orchestrator import parse_and_validate_model
from nirnaya_api.errors import ValidationError


def test_valid_model_parses(simple_model_dict):
    parsed = ModelSchema.model_validate(simple_model_dict)
    assert set(parsed.variables) == {"x", "y"}
    assert parsed.objective.sense == "max"


def test_unknown_variable_in_constraint_rejected(simple_model_dict):
    simple_model_dict["constraints"]["c1"]["coefficients"]["z"] = 1.0
    with pytest.raises(PydanticValidationError):
        ModelSchema.model_validate(simple_model_dict)


def test_unknown_variable_in_objective_rejected(simple_model_dict):
    simple_model_dict["objective"]["coefficients"]["z"] = 1.0
    with pytest.raises(PydanticValidationError):
        ModelSchema.model_validate(simple_model_dict)


def test_empty_variables_rejected(simple_model_dict):
    simple_model_dict["variables"] = {}
    with pytest.raises(PydanticValidationError):
        ModelSchema.model_validate(simple_model_dict)


def test_bad_bounds_rejected(simple_model_dict):
    simple_model_dict["variables"]["x"]["lb"] = 100
    simple_model_dict["variables"]["x"]["ub"] = 1
    with pytest.raises(PydanticValidationError):
        ModelSchema.model_validate(simple_model_dict)


def test_empty_constraint_coefficients_rejected(simple_model_dict):
    simple_model_dict["constraints"]["c1"]["coefficients"] = {}
    with pytest.raises(PydanticValidationError):
        ModelSchema.model_validate(simple_model_dict)


def test_orchestrator_wraps_schema_errors(simple_model_dict):
    simple_model_dict["variables"] = {}
    with pytest.raises(ValidationError) as exc_info:
        parse_and_validate_model(simple_model_dict)
    assert "issues" in exc_info.value.details
