"""Numerical edge-case tests: extreme bounds, near-zero coefficients,
large sparse dimensions, and effectively-infinite bound detection.
"""

import math

import numpy as np
import pytest

from nirnaya_core.model import ConstraintSense, Model, ObjectiveSense
from nirnaya_core.numeric.config import NumericalConfig


class TestExtremeBounds:
    def test_very_large_finite_bound(self):
        m = Model()
        m.add_variable("x", lower=0.0, upper=1e18)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        data = m.build()
        assert data.upper[0] == 1e18

    def test_effectively_infinite_bound_detected_by_config(self):
        cfg = NumericalConfig(infinity=1e20)
        assert cfg.is_effectively_infinite(1e25)
        assert not cfg.is_effectively_infinite(1e15)

    def test_ieee_infinity_bound_survives_build(self):
        m = Model()
        m.add_variable("x", lower=-math.inf, upper=math.inf)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        data = m.build()
        assert math.isinf(data.lower[0])
        assert math.isinf(data.upper[0])


class TestNearZeroCoefficients:
    def test_tiny_coefficient_preserved_not_flushed_to_zero(self):
        m = Model()
        m.add_variable("x", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1e-15})
        data = m.build()
        assert data.c[0] == pytest.approx(1e-15)

    def test_exact_zero_coefficient_produces_no_nonzero_entry(self):
        m = Model()
        m.add_variable("x", lower=0.0)
        m.add_variable("y", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        m.add_constraint("c1", {"x": 1.0, "y": 0.0}, ConstraintSense.LE, 1.0)
        data = m.build()
        # y's zero coefficient should not create a stored nonzero in a_ub
        assert data.a_ub.nnz == 1


class TestLargeSparseModel:
    def test_many_variables_few_nonzeros(self):
        n = 2000
        m = Model()
        for i in range(n):
            m.add_variable(f"x{i}", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x0": 1.0})
        m.add_constraint("c0", {"x0": 1.0, f"x{n - 1}": 1.0}, ConstraintSense.LE, 1.0)
        data = m.build()
        assert data.n_variables == n
        assert data.a_ub.nnz == 2
        assert data.a_ub.density < 0.01


class TestDtypeConsistency:
    def test_float32_config_produces_float32_arrays(self):
        cfg = NumericalConfig(dtype=np.float32)
        m = Model(config=cfg)
        m.add_variable("x", lower=0.0)
        m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0})
        data = m.build()
        assert data.c.dtype == np.float32
        assert data.a_ub.dtype == np.float32
