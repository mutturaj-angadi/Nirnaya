"""Unit tests for Variable, Constraint, LinearObjective value objects."""

import math

import pytest

from nirnaya_core.model import ConstraintSense, ObjectiveSense, VariableType
from nirnaya_core.model.constraint import Constraint
from nirnaya_core.model.objective import LinearObjective
from nirnaya_core.model.variable import Variable


class TestVariable:
    def test_default_bounds(self):
        v = Variable(name="x")
        assert v.lower == -math.inf
        assert v.upper == math.inf
        assert v.is_free()

    def test_explicit_bounds(self):
        v = Variable(name="x", lower=0.0, upper=10.0)
        assert not v.is_free()
        assert not v.is_fixed()

    def test_fixed_variable(self):
        v = Variable(name="x", lower=5.0, upper=5.0)
        assert v.is_fixed()

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            Variable(name="")

    def test_whitespace_name_rejected(self):
        with pytest.raises(ValueError):
            Variable(name="   ")

    def test_lower_greater_than_upper_rejected(self):
        with pytest.raises(ValueError):
            Variable(name="x", lower=5.0, upper=1.0)

    def test_nan_bound_rejected(self):
        with pytest.raises(ValueError):
            Variable(name="x", lower=math.nan)
        with pytest.raises(ValueError):
            Variable(name="x", upper=math.nan)

    def test_with_index(self):
        v = Variable(name="x")
        v2 = v.with_index(3)
        assert v2.index == 3
        assert v.index is None  # original untouched (immutability)

    def test_round_trip_dict(self):
        v = Variable(name="x", lower=0.0, upper=1.0, vtype=VariableType.INTEGER, index=2)
        d = v.to_dict()
        v2 = Variable.from_dict(d)
        assert v2 == Variable(name="x", lower=0.0, upper=1.0, vtype=VariableType.INTEGER, index=2)

    def test_default_vtype_is_continuous(self):
        v = Variable(name="x")
        assert v.vtype == VariableType.CONTINUOUS


class TestLinearObjective:
    def test_basic_construction(self):
        obj = LinearObjective(sense=ObjectiveSense.MAXIMIZE, coefficients={"x": 3.0, "y": 2.0})
        assert obj.coefficient_for("x") == 3.0
        assert obj.coefficient_for("z") == 0.0  # absent -> implicit 0

    def test_evaluate(self):
        obj = LinearObjective(sense=ObjectiveSense.MINIMIZE, coefficients={"x": 2.0, "y": -1.0}, constant=5.0)
        assert obj.evaluate({"x": 3.0, "y": 1.0}) == pytest.approx(2 * 3 + (-1) * 1 + 5)

    def test_missing_variable_defaults_to_zero_in_evaluate(self):
        obj = LinearObjective(sense=ObjectiveSense.MINIMIZE, coefficients={"x": 2.0})
        assert obj.evaluate({}) == 0.0

    def test_nonfinite_constant_rejected(self):
        with pytest.raises(ValueError):
            LinearObjective(sense=ObjectiveSense.MINIMIZE, coefficients={}, constant=math.inf)

    def test_nonfinite_coefficient_rejected(self):
        with pytest.raises(ValueError):
            LinearObjective(sense=ObjectiveSense.MINIMIZE, coefficients={"x": math.nan})

    def test_invalid_sense_rejected(self):
        with pytest.raises(ValueError):
            LinearObjective(sense="minimize", coefficients={})  # type: ignore[arg-type]

    def test_round_trip_dict(self):
        obj = LinearObjective(sense=ObjectiveSense.MAXIMIZE, coefficients={"x": 1.5}, constant=2.0)
        d = obj.to_dict()
        obj2 = LinearObjective.from_dict(d)
        assert obj2.sense == obj.sense
        assert obj2.coefficients == obj.coefficients
        assert obj2.constant == obj.constant


class TestConstraint:
    def test_basic_construction(self):
        c = Constraint(name="c1", coefficients={"x": 1.0, "y": 1.0}, sense=ConstraintSense.LE, rhs=10.0)
        assert c.coefficient_for("x") == 1.0
        assert c.coefficient_for("z") == 0.0

    def test_empty_coefficients_rejected(self):
        with pytest.raises(ValueError):
            Constraint(name="c1", coefficients={}, sense=ConstraintSense.LE, rhs=1.0)

    def test_nonfinite_rhs_rejected(self):
        with pytest.raises(ValueError):
            Constraint(name="c1", coefficients={"x": 1.0}, sense=ConstraintSense.LE, rhs=math.inf)

    def test_nonfinite_coefficient_rejected(self):
        with pytest.raises(ValueError):
            Constraint(name="c1", coefficients={"x": math.nan}, sense=ConstraintSense.LE, rhs=1.0)

    def test_residual_equality(self):
        c = Constraint(name="c1", coefficients={"x": 1.0, "y": 1.0}, sense=ConstraintSense.EQ, rhs=5.0)
        assert c.residual({"x": 2.0, "y": 3.0}) == pytest.approx(0.0)
        assert c.residual({"x": 2.0, "y": 2.0}) == pytest.approx(-1.0)

    def test_residual_inequality(self):
        c = Constraint(name="c1", coefficients={"x": 1.0}, sense=ConstraintSense.LE, rhs=5.0)
        assert c.residual({"x": 7.0}) == pytest.approx(2.0)  # violated by 2
        assert c.residual({"x": 3.0}) == pytest.approx(-2.0)  # slack of 2

    def test_round_trip_dict(self):
        c = Constraint(name="c1", coefficients={"x": 1.0}, sense=ConstraintSense.GE, rhs=3.0, index=0)
        d = c.to_dict()
        c2 = Constraint.from_dict(d)
        assert c2 == c

    def test_with_index(self):
        c = Constraint(name="c1", coefficients={"x": 1.0}, sense=ConstraintSense.LE, rhs=1.0)
        c2 = c.with_index(4)
        assert c2.index == 4
        assert c.index is None
