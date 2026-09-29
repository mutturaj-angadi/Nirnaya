"""Unit tests for nirnaya_core.numeric.NumericalConfig."""

import math

import numpy as np
import pytest

from nirnaya_core.exceptions import ConfigurationError
from nirnaya_core.numeric.config import DEFAULT_CONFIG, NumericalConfig


class TestDefaults:
    def test_default_config_valid(self):
        cfg = NumericalConfig()
        assert cfg.feasibility_tol > 0
        assert cfg.dtype == np.float64

    def test_module_level_default_instance(self):
        assert isinstance(DEFAULT_CONFIG, NumericalConfig)


class TestValidation:
    def test_negative_tolerance_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(feasibility_tol=-1e-8)

    def test_nan_tolerance_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(feasibility_tol=math.nan)

    def test_infinite_tolerance_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(feasibility_tol=math.inf)

    def test_zero_or_negative_infinity_bound_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(infinity=0.0)
        with pytest.raises(ConfigurationError):
            NumericalConfig(infinity=-100.0)

    def test_non_positive_max_iterations_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(max_iterations=0)
        with pytest.raises(ConfigurationError):
            NumericalConfig(max_iterations=-5)

    def test_non_int_max_iterations_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(max_iterations=10.5)  # type: ignore[arg-type]

    def test_negative_time_limit_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(time_limit_seconds=-1.0)

    def test_none_time_limit_allowed(self):
        cfg = NumericalConfig(time_limit_seconds=None)
        assert cfg.time_limit_seconds is None

    def test_unsupported_dtype_rejected(self):
        with pytest.raises(ConfigurationError):
            NumericalConfig(dtype=np.complex128)

    def test_float32_dtype_allowed(self):
        cfg = NumericalConfig(dtype=np.float32)
        assert cfg.dtype == np.float32


class TestImmutability:
    def test_frozen(self):
        cfg = NumericalConfig()
        with pytest.raises(Exception):
            cfg.feasibility_tol = 1.0  # type: ignore[misc]

    def test_with_overrides_produces_new_instance(self):
        cfg = NumericalConfig()
        cfg2 = cfg.with_overrides(feasibility_tol=1e-6)
        assert cfg2.feasibility_tol == 1e-6
        assert cfg.feasibility_tol != 1e-6  # original untouched

    def test_with_overrides_revalidates(self):
        cfg = NumericalConfig()
        with pytest.raises(ConfigurationError):
            cfg.with_overrides(feasibility_tol=-1.0)


class TestHelpers:
    def test_is_effectively_infinite_true_ieee_inf(self):
        cfg = NumericalConfig()
        assert cfg.is_effectively_infinite(math.inf)
        assert cfg.is_effectively_infinite(-math.inf)

    def test_is_effectively_infinite_true_large_finite(self):
        cfg = NumericalConfig(infinity=1e20)
        assert cfg.is_effectively_infinite(1e21)

    def test_is_effectively_infinite_false_for_normal_value(self):
        cfg = NumericalConfig()
        assert not cfg.is_effectively_infinite(100.0)


class TestSerialization:
    def test_round_trip(self):
        cfg = NumericalConfig(feasibility_tol=1e-7, max_iterations=500)
        d = cfg.to_dict()
        cfg2 = NumericalConfig.from_dict(d)
        assert cfg2 == cfg

    def test_to_dict_dtype_is_string(self):
        cfg = NumericalConfig()
        d = cfg.to_dict()
        assert d["dtype"] == "float64"
