"""
Unit tests for nirnaya_core.model.Model.

Note on "infeasible" / "unbounded" models: Part 1 ships no solver, so these
tests exercise the *representation* layer only -- they build models that
are mathematically infeasible or unbounded and assert that Model.build()
still produces a structurally correct ProblemData snapshot (since
structural validity and feasibility/boundedness are orthogonal concerns).
Proving infeasibility/unboundedness is the job of the solver in Parts 2-6.
"""

import math

import numpy as np
import pytest

from nirnaya_core.model import ConstraintSense, Model, ObjectiveSense, VariableType
from nirnaya_core.validation.errors import ModelValidationError


def _toy_max_model() -> Model:
    m = Model(name="toy")
    m.add_variable("x", lower=0.0)
    m.add_variable("y", lower=0.0)
    m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 3.0, "y": 2.0})
    m.add_constraint("cap", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 4.0)
    return m


class TestTinyLPs:
    def test_build_produces_expected_shapes(self):
        m = _toy_max_model()
        data = m.build()
        assert data.n_variables == 2
        assert data.n_eq == 0
        assert data.n_ub == 1
        np.testing.assert_allclose(data.c, [3.0, 2.0])
        np.testing.assert_allclose(data.a_ub.to_dense(), [[1.0, 1.0]])
        np.testing.assert_allclose(data.b_ub, [4.0])

    def test_variable_order_is_insertion_order(self):
        m = Model()
        m.add_variable("b")
        m.add_variable("a")
        m.set_objective(ObjectiveSense.MINIMIZE, {"a": 1.0, "b": 1.0})
        data = m.build()
        assert data.variable_names == ("b", "a")

    def test_objective_sense_preserved(self):
        m = _toy_max_model()
        data = m.build()
        assert data.sense == ObjectiveSense.MAXIMIZE


class TestEqualityOnlyModel:
    def test_build(self):
        m = Model(name="eq_only")
        m.add_variable("x")
        m.add_variable("y")
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0, "y": 1.0})
        m.add_constraint("balance", {"x": 1.0, "y": -1.0}, ConstraintSense.EQ, 0.0)
        data = m.build()
        assert data.n_eq == 1
        assert data.n_ub == 0
        np.testing.assert_allclose(data.a_eq.to_dense(), [[1.0, -1.0]])
        np.testing.assert_allclose(data.b_eq, [0.0])


class TestInequalityOnlyModel:
    def test_le_and_ge_both_normalize_into_a_ub(self):
        m = Model(name="ineq_only")
        m.add_variable("x", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        m.add_constraint("upper", {"x": 1.0}, ConstraintSense.LE, 10.0)
        m.add_constraint("lower", {"x": 1.0}, ConstraintSense.GE, 2.0)
        data = m.build()
        assert data.n_eq == 0
        assert data.n_ub == 2
        # GE row a^T x >= b is stored negated as -a^T x <= -b
        dense = data.a_ub.to_dense()
        np.testing.assert_allclose(dense[0], [1.0])
        np.testing.assert_allclose(dense[1], [-1.0])
        np.testing.assert_allclose(data.b_ub, [10.0, -2.0])

    def test_ge_semantics_preserved_after_negation(self):
        # x >= 2 with x = 1.5 should be reported as violated by the
        # negated row: -x <= -2  =>  -1.5 <= -2 is False (violated).
        m = Model(name="ge_check")
        m.add_variable("x", lower=-math.inf)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        m.add_constraint("floor", {"x": 1.0}, ConstraintSense.GE, 2.0)
        data = m.build()
        violation = data.a_ub.matvec(np.array([1.5])) - data.b_ub
        assert violation[0] > 0  # violated, as expected


class TestBoundsAndVariableKinds:
    def test_bounded_variable(self):
        m = Model()
        m.add_variable("x", lower=1.0, upper=5.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        data = m.build()
        assert data.lower[0] == 1.0
        assert data.upper[0] == 5.0

    def test_free_variable_default_bounds(self):
        m = Model()
        m.add_variable("x", lower=-math.inf, upper=math.inf)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        data = m.build()
        assert math.isinf(data.lower[0]) and data.lower[0] < 0
        assert math.isinf(data.upper[0]) and data.upper[0] > 0

    def test_integer_and_binary_vartypes_supported_structurally(self):
        m = Model()
        m.add_variable("i", lower=0, upper=10, vtype=VariableType.INTEGER)
        m.add_variable("b", lower=0, upper=1, vtype=VariableType.BINARY)
        m.set_objective(ObjectiveSense.MINIMIZE, {"i": 1.0, "b": 1.0})
        data = m.build()
        assert data.var_types == (VariableType.INTEGER, VariableType.BINARY)


class TestMinMax:
    def test_minimize(self):
        m = Model()
        m.add_variable("x", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        assert m.build().sense == ObjectiveSense.MINIMIZE

    def test_maximize(self):
        m = Model()
        m.add_variable("x", lower=0.0)
        m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 1.0})
        assert m.build().sense == ObjectiveSense.MAXIMIZE


class TestKnownInfeasibleAndUnboundedRepresentation:
    def test_infeasible_model_still_builds_structurally(self):
        # x <= 1 and x >= 5 with x in [0, inf): mathematically infeasible,
        # but structurally a perfectly valid ProblemData for a solver to
        # later prove infeasible.
        m = Model(name="infeasible")
        m.add_variable("x", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        m.add_constraint("c1", {"x": 1.0}, ConstraintSense.LE, 1.0)
        m.add_constraint("c2", {"x": 1.0}, ConstraintSense.GE, 5.0)
        data = m.build()
        assert data.n_ub == 2

    def test_unbounded_model_still_builds_structurally(self):
        # maximize x, x free, no constraints: unbounded above, but a valid
        # structural model.
        m = Model(name="unbounded")
        m.add_variable("x", lower=-math.inf, upper=math.inf)
        m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 1.0})
        data = m.build()
        assert data.n_eq == 0 and data.n_ub == 0
        assert math.isinf(data.upper[0])


class TestSparseModels:
    def test_sparse_objective_and_constraints(self):
        # 50 variables, objective and each constraint touch only 2 of them.
        m = Model()
        for i in range(50):
            m.add_variable(f"x{i}", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x0": 1.0, "x49": 1.0})
        m.add_constraint("link", {"x0": 1.0, "x1": -1.0}, ConstraintSense.EQ, 0.0)
        data = m.build()
        assert data.n_variables == 50
        assert data.a_eq.nnz == 2
        # objective vector is dense-shaped but sparse-valued
        assert np.count_nonzero(data.c) == 2


class TestValidationFailures:
    def test_no_variables_raises(self):
        m = Model()
        m.set_objective(ObjectiveSense.MINIMIZE, {})
        with pytest.raises(ModelValidationError) as exc_info:
            m.build()
        codes = {e.code for e in exc_info.value.errors}
        assert "E_EMPTY_MODEL" in codes

    def test_no_objective_raises(self):
        m = Model()
        m.add_variable("x")
        with pytest.raises(ModelValidationError) as exc_info:
            m.build()
        codes = {e.code for e in exc_info.value.errors}
        assert "E_INVALID_OBJECTIVE" in codes

    def test_duplicate_variable_name_raises(self):
        m = Model()
        m.add_variable("x")
        with pytest.raises(Exception):
            m.add_variable("x")

    def test_duplicate_constraint_name_raises(self):
        m = Model()
        m.add_variable("x")
        m.add_constraint("c1", {"x": 1.0}, ConstraintSense.LE, 1.0)
        with pytest.raises(Exception):
            m.add_constraint("c1", {"x": 1.0}, ConstraintSense.LE, 2.0)

    def test_unknown_variable_in_objective_raises(self):
        m = Model()
        m.add_variable("x")
        m.set_objective(ObjectiveSense.MINIMIZE, {"y": 1.0})
        with pytest.raises(ModelValidationError) as exc_info:
            m.build()
        codes = {e.code for e in exc_info.value.errors}
        assert "E_UNKNOWN_VARIABLE" in codes

    def test_unknown_variable_in_constraint_raises(self):
        m = Model()
        m.add_variable("x")
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        m.add_constraint("c1", {"z": 1.0}, ConstraintSense.LE, 1.0)
        with pytest.raises(ModelValidationError) as exc_info:
            m.build()
        codes = {e.code for e in exc_info.value.errors}
        assert "E_UNKNOWN_VARIABLE" in codes

    def test_multiple_errors_are_all_aggregated(self):
        m = Model()
        # no variables AND no objective simultaneously
        with pytest.raises(ModelValidationError) as exc_info:
            m.build()
        codes = {e.code for e in exc_info.value.errors}
        assert "E_EMPTY_MODEL" in codes
        assert "E_INVALID_OBJECTIVE" in codes

    def test_is_valid_false_for_invalid_model(self):
        m = Model()
        assert m.is_valid() is False

    def test_is_valid_true_for_valid_model(self):
        m = _toy_max_model()
        assert m.is_valid() is True

    def test_build_never_returns_on_invalid_model(self):
        m = Model()
        m.add_variable("x")
        m.set_objective(ObjectiveSense.MINIMIZE, {"y": 999.0})
        with pytest.raises(ModelValidationError):
            m.build()


class TestRemoval:
    def test_remove_variable_reindexes(self):
        m = Model()
        m.add_variable("a")
        m.add_variable("b")
        m.add_variable("c")
        m.remove_variable("b")
        assert [v.index for v in m.variables] == [0, 1]
        assert m.variable_names() == ["a", "c"]

    def test_remove_constraint_reindexes(self):
        m = Model()
        m.add_variable("x")
        m.add_constraint("c1", {"x": 1.0}, ConstraintSense.LE, 1.0)
        m.add_constraint("c2", {"x": 1.0}, ConstraintSense.LE, 2.0)
        m.remove_constraint("c1")
        assert [c.index for c in m.constraints] == [0]
        assert m.constraint_names() == ["c2"]

    def test_remove_missing_variable_raises_keyerror(self):
        m = Model()
        with pytest.raises(KeyError):
            m.remove_variable("nope")


class TestModelSerializationRoundTrip:
    def test_round_trip(self):
        m = _toy_max_model()
        d = m.to_dict()
        m2 = Model.from_dict(d)
        data1 = m.build()
        data2 = m2.build()
        assert data1.variable_names == data2.variable_names
        np.testing.assert_allclose(data1.c, data2.c)
        assert data1.a_ub == data2.a_ub
