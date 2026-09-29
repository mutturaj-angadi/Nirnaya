import pytest

from nirnaya_api.errors import PayloadTooLargeError, ResourceLimitError, ValidationError
from nirnaya_api.options import SolveOptions
from nirnaya_api.orchestrator import solve_model, validate_model
from nirnaya_api.security import SecurityLimits, check_model_size, check_payload_size


def test_check_payload_size_rejects_oversized():
    limits = SecurityLimits(max_payload_bytes=10)
    with pytest.raises(PayloadTooLargeError):
        check_payload_size(b"x" * 100, limits)


def test_check_model_size_rejects_too_many_variables(simple_model_dict):
    limits = SecurityLimits(max_variables=1)
    with pytest.raises(ResourceLimitError):
        check_model_size(
            {
                "variables": simple_model_dict["variables"],
                "constraints": {},
                "objective": {"coefficients": {}},
            },
            limits,
        )


def test_solve_model_rejects_malformed_json_shape(fake_backend):
    bad = {"variables": {}, "objective": {"coefficients": {}}}
    with pytest.raises(ValidationError):
        solve_model(bad, SolveOptions())


def test_solve_model_rejects_bad_option_values(fake_backend, simple_model_dict):
    with pytest.raises(ValidationError):
        solve_model(simple_model_dict, SolveOptions(time_limit=-1))


def test_validate_model_reports_issues_instead_of_raising(fake_backend):
    bad = {"variables": {}, "objective": {"coefficients": {}}}
    report = validate_model(bad)
    assert report["valid"] is False
    assert report["issues"]


def test_missing_required_fields_rejected(fake_backend):
    with pytest.raises(ValidationError):
        solve_model({"variables": {"x": {}}}, SolveOptions())


def test_non_dict_model_rejected(fake_backend):
    with pytest.raises(Exception):
        solve_model("not a model", SolveOptions())


def test_validate_model_handles_non_dict_input_gracefully(fake_backend):
    report = validate_model(["not", "a", "model"])
    assert report["valid"] is False
    assert report["num_variables"] == 0
    assert report["issues"]


def test_extra_unknown_option_rejected():
    with pytest.raises(ValidationError):
        SolveOptions.from_dict({"solver": "auto", "bogus_option": 1})
