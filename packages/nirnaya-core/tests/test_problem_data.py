"""Unit tests for ProblemData and standalone numerical validation."""

import math

import numpy as np
import pytest

from nirnaya_core.matrix.sparse import SparseMatrix
from nirnaya_core.model import ConstraintSense, Model, ObjectiveSense
from nirnaya_core.model.enums import VariableType
from nirnaya_core.model.problem_data import ProblemData
from nirnaya_core.numeric.config import NumericalConfig
from nirnaya_core.validation.errors import ModelValidationError
from nirnaya_core.validation.validator import is_problem_data_valid, validate_problem_data


def _valid_data() -> ProblemData:
    m = Model()
    m.add_variable("x", lower=0.0, upper=10.0)
    m.add_variable("y", lower=0.0, upper=10.0)
    m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 1.0, "y": 1.0})
    m.add_constraint("c1", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 5.0)
    return m.build()


class TestShapeValidation:
    def test_valid_data_passes(self):
        data = _valid_data()
        validate_problem_data(data)  # should not raise
        assert is_problem_data_valid(data)

    def test_mismatched_c_shape_rejected_by_dataclass(self):
        data = _valid_data()
        with pytest.raises(ValueError):
            ProblemData(
                variable_names=data.variable_names,
                constraint_names_eq=data.constraint_names_eq,
                constraint_names_ub=data.constraint_names_ub,
                sense=data.sense,
                c=np.array([1.0]),  # wrong length
                objective_constant=0.0,
                a_eq=data.a_eq,
                b_eq=data.b_eq,
                a_ub=data.a_ub,
                b_ub=data.b_ub,
                lower=data.lower,
                upper=data.upper,
                var_types=data.var_types,
                config=data.config,
            )

    def test_zero_variable_model_flagged(self):
        data = ProblemData(
            variable_names=(),
            constraint_names_eq=(),
            constraint_names_ub=(),
            sense=ObjectiveSense.MINIMIZE,
            c=np.array([]),
            objective_constant=0.0,
            a_eq=SparseMatrix.empty(0, 0),
            b_eq=np.array([]),
            a_ub=SparseMatrix.empty(0, 0),
            b_ub=np.array([]),
            lower=np.array([]),
            upper=np.array([]),
            var_types=(),
            config=NumericalConfig(),
        )
        with pytest.raises(ModelValidationError) as exc_info:
            validate_problem_data(data)
        codes = {e.code for e in exc_info.value.errors}
        assert "E_EMPTY_MODEL" in codes

    def test_bad_bounds_flagged(self):
        data = _valid_data()
        bad_lower = np.array(data.lower)
        bad_upper = np.array(data.upper)
        # force lower > upper via a fresh ProblemData built manually
        bad = ProblemData(
            variable_names=data.variable_names,
            constraint_names_eq=data.constraint_names_eq,
            constraint_names_ub=data.constraint_names_ub,
            sense=data.sense,
            c=data.c,
            objective_constant=data.objective_constant,
            a_eq=data.a_eq,
            b_eq=data.b_eq,
            a_ub=data.a_ub,
            b_ub=data.b_ub,
            lower=np.array([5.0, 0.0]),
            upper=np.array([1.0, 10.0]),
            var_types=data.var_types,
            config=data.config,
        )
        with pytest.raises(ModelValidationError) as exc_info:
            validate_problem_data(bad)
        codes = {e.code for e in exc_info.value.errors}
        assert "E_INVALID_BOUNDS" in codes

    def test_immutability_of_arrays(self):
        data = _valid_data()
        with pytest.raises(ValueError):
            data.c[0] = 999.0  # read-only array


class TestNumericalEdgeCases:
    def test_nan_free_data_is_valid(self):
        data = _valid_data()
        assert is_problem_data_valid(data)

    def test_dimension_mismatch_is_already_rejected_at_construction(self):
        # ProblemData's own __post_init__ enforces shape agreement eagerly,
        # so a mismatched a_eq can never reach validate_problem_data's
        # (defensive) dimension checks through the public constructor.
        data = _valid_data()
        with pytest.raises(ValueError):
            ProblemData(
                variable_names=data.variable_names,
                constraint_names_eq=("dummy_1", "dummy_2"),  # 2 names, 1 row
                constraint_names_ub=data.constraint_names_ub,
                sense=data.sense,
                c=data.c,
                objective_constant=data.objective_constant,
                a_eq=SparseMatrix.empty(1, len(data.variable_names)),
                b_eq=np.array([0.0, 0.0]),
                a_ub=data.a_ub,
                b_ub=data.b_ub,
                lower=data.lower,
                upper=data.upper,
                var_types=data.var_types,
                config=data.config,
            )


class TestSerialization:
    def test_round_trip(self):
        data = _valid_data()
        d = data.to_dict()
        data2 = ProblemData.from_dict(d)
        assert data2.variable_names == data.variable_names
        np.testing.assert_allclose(data2.c, data.c)
        assert data2.a_ub == data.a_ub
        assert data2.var_types == data.var_types
