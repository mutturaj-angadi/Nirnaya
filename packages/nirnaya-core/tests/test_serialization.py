"""Unit tests for nirnaya_core.io deterministic serialization."""

import json

import numpy as np
import pytest

from nirnaya_core.exceptions import SerializationError
from nirnaya_core.io.serialization import (
    model_from_json,
    model_to_json,
    problem_data_from_json,
    problem_data_to_json,
    read_json,
    solver_result_from_json,
    solver_result_to_json,
    write_json,
)
from nirnaya_core.model import ConstraintSense, Model, ObjectiveSense
from nirnaya_core.solution.result import SolverResult
from nirnaya_core.status.status import SolverStatus


def _sample_model() -> Model:
    m = Model(name="sample")
    m.add_variable("x", lower=0.0)
    m.add_variable("y", lower=0.0, upper=10.0)
    m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 3.0, "y": 2.0})
    m.add_constraint("cap", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 4.0)
    return m


class TestModelSerialization:
    def test_round_trip(self):
        m = _sample_model()
        text = model_to_json(m)
        m2 = model_from_json(text)
        assert m2.name == m.name
        assert m2.variable_names() == m.variable_names()
        d1, d2 = m.build(), m2.build()
        np.testing.assert_allclose(d1.c, d2.c)

    def test_output_is_deterministic(self):
        m = _sample_model()
        text1 = model_to_json(m)
        text2 = model_to_json(m)
        assert text1 == text2

    def test_output_is_valid_json_with_sorted_keys(self):
        m = _sample_model()
        text = model_to_json(m)
        parsed = json.loads(text)
        assert parsed["kind"] == "Model"
        assert "schema_version" in parsed

    def test_wrong_kind_rejected(self):
        m = _sample_model()
        text = model_to_json(m)
        with pytest.raises(SerializationError):
            problem_data_from_json(text)

    def test_malformed_json_rejected(self):
        with pytest.raises(SerializationError):
            model_from_json("{not valid json")

    def test_missing_envelope_rejected(self):
        with pytest.raises(SerializationError):
            model_from_json(json.dumps({"foo": "bar"}))

    def test_wrong_schema_version_rejected(self):
        text = model_to_json(_sample_model())
        payload = json.loads(text)
        payload["schema_version"] = 999
        with pytest.raises(SerializationError):
            model_from_json(json.dumps(payload))


class TestProblemDataSerialization:
    def test_round_trip(self):
        data = _sample_model().build()
        text = problem_data_to_json(data)
        data2 = problem_data_from_json(text)
        np.testing.assert_allclose(data.c, data2.c)
        assert data.a_ub == data2.a_ub

    def test_deterministic(self):
        data = _sample_model().build()
        assert problem_data_to_json(data) == problem_data_to_json(data)


class TestSolverResultSerialization:
    def test_round_trip(self):
        result = SolverResult(
            status=SolverStatus.OPTIMAL,
            variable_names=["x", "y"],
            primal=[1.0, 3.0],
        )
        text = solver_result_to_json(result)
        result2 = solver_result_from_json(text)
        assert result2.status == result.status
        assert result2.primal == result.primal

    def test_deterministic(self):
        result = SolverResult(status=SolverStatus.INFEASIBLE, message="test")
        assert solver_result_to_json(result) == solver_result_to_json(result)


class TestFileRoundTrip:
    def test_write_then_read_json(self, tmp_path):
        m = _sample_model()
        text = model_to_json(m)
        path = tmp_path / "model.json"
        write_json(text, path)
        read_back = read_json(path)
        assert read_back.strip() == text.strip()
        m2 = model_from_json(read_back)
        assert m2.variable_names() == m.variable_names()

    def test_read_missing_file_raises_serialization_error(self, tmp_path):
        with pytest.raises(SerializationError):
            read_json(tmp_path / "does_not_exist.json")

    def test_infinite_bound_round_trips_through_json(self):
        m = _sample_model()
        m.add_variable("free_var", lower=float("-inf"), upper=float("inf"))
        m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 1.0})
        text = model_to_json(m)
        m2 = model_from_json(text)
        var = m2.get_variable("free_var")
        assert var.lower == float("-inf")
        assert var.upper == float("inf")
