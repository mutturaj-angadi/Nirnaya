"""
Numerical configuration for Nirnaya Core.

This module makes numerical tolerances and dtype policy an explicit,
first-class, serializable object (:class:`NumericalConfig`) rather than
scattering magic numbers through solver code. Parts 2-6 (presolve, the
solver kernel, GPU backends, reporting) must accept a ``NumericalConfig``
rather than hardcoding their own tolerances, so that a single config
object governs an entire solve end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from nirnaya_core.exceptions import ConfigurationError

#: The canonical floating point dtype used throughout the numerical core.
#: Kept as a single source of truth so that GPU/native backends (Parts
#: 2-6) know exactly what precision the CPU reference implementation
#: assumes. Mixed precision is a deliberate, explicit future extension
#: point, not an accident of "whatever numpy defaulted to".
DEFAULT_DTYPE = np.float64

_SUPPORTED_DTYPES = (np.float64, np.float32)


@dataclass(frozen=True)
class NumericalConfig:
    """Explicit numerical tolerances and precision policy.

    All tolerances are non-negative finite floats. Instances are immutable
    (``frozen=True``); use :meth:`with_overrides` to derive a modified copy.

    Attributes:
        feasibility_tol: Maximum allowed absolute constraint violation
            (``max(|A_eq x - b_eq|, |A_ub x - b_ub|_+ )``) for a point to be
            considered primal feasible.
        optimality_tol: Maximum allowed dual-gap / reduced-cost violation
            for a point to be considered optimal.
        bound_tol: Slack allowed when checking ``l <= x <= u``.
        pivot_tol: Minimum absolute magnitude for a matrix entry to be
            treated as numerically nonzero during factorization/pivoting.
        zero_tol: Threshold below which a value is treated as exactly zero
            for structural purposes (e.g. sparsity pattern decisions).
        infinity: The finite float used to represent "no bound". Values
            with absolute value >= ``infinity`` are treated as unbounded.
            Kept explicit (rather than ``math.inf`` everywhere) because
            some downstream numerical kernels cannot represent IEEE
            infinities in fixed-point or GPU tensor formats.
        max_iterations: Default iteration budget a solver should honor
            unless overridden at call time.
        time_limit_seconds: Default wall-clock budget, or ``None`` for no
            limit.
        dtype: Canonical floating point dtype for numerical arrays.
            Must be one of ``numpy.float64`` or ``numpy.float32``.
    """

    feasibility_tol: float = 1e-8
    optimality_tol: float = 1e-8
    bound_tol: float = 1e-8
    pivot_tol: float = 1e-10
    zero_tol: float = 1e-12
    infinity: float = 1e20
    max_iterations: int = 10_000
    time_limit_seconds: float | None = None
    dtype: Any = field(default=DEFAULT_DTYPE)

    def __post_init__(self) -> None:
        errors: list[str] = []

        for attr_name in (
            "feasibility_tol",
            "optimality_tol",
            "bound_tol",
            "pivot_tol",
            "zero_tol",
            "infinity",
        ):
            value = getattr(self, attr_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{attr_name} must be a real number, got {type(value).__name__}")
                continue
            if not math.isfinite(value):
                errors.append(f"{attr_name} must be finite, got {value!r}")
            elif value < 0:
                errors.append(f"{attr_name} must be non-negative, got {value!r}")

        if self.infinity <= 0:
            errors.append(f"infinity must be strictly positive, got {self.infinity!r}")

        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool):
            errors.append(f"max_iterations must be an int, got {type(self.max_iterations).__name__}")
        elif self.max_iterations <= 0:
            errors.append(f"max_iterations must be positive, got {self.max_iterations!r}")

        if self.time_limit_seconds is not None:
            if not isinstance(self.time_limit_seconds, (int, float)) or isinstance(
                self.time_limit_seconds, bool
            ):
                errors.append("time_limit_seconds must be a real number or None")
            elif not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0:
                errors.append(f"time_limit_seconds must be positive and finite, got {self.time_limit_seconds!r}")

        resolved_dtype = np.dtype(self.dtype)
        if resolved_dtype.type not in _SUPPORTED_DTYPES:
            errors.append(
                f"dtype must be one of {[d.__name__ for d in _SUPPORTED_DTYPES]}, "
                f"got {resolved_dtype}"
            )

        if errors:
            raise ConfigurationError(
                "Invalid NumericalConfig: " + "; ".join(errors),
                details={"errors": errors},
            )

    def with_overrides(self, **kwargs: Any) -> "NumericalConfig":
        """Return a new :class:`NumericalConfig` with the given fields replaced.

        Validates the resulting configuration exactly as the constructor
        does (via ``dataclasses.replace``, which re-runs ``__post_init__``).
        """
        return replace(self, **kwargs)

    def is_effectively_infinite(self, value: float) -> bool:
        """Return True if ``value`` should be treated as +/- infinity.

        Both true IEEE infinities and finite magnitudes at or beyond
        :attr:`infinity` are considered "no bound", since callers may
        legitimately pass either representation.
        """
        return math.isinf(value) or abs(value) >= self.infinity

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation. ``dtype`` is stored by name."""
        return {
            "feasibility_tol": self.feasibility_tol,
            "optimality_tol": self.optimality_tol,
            "bound_tol": self.bound_tol,
            "pivot_tol": self.pivot_tol,
            "zero_tol": self.zero_tol,
            "infinity": self.infinity,
            "max_iterations": self.max_iterations,
            "time_limit_seconds": self.time_limit_seconds,
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NumericalConfig":
        payload = dict(data)
        if "dtype" in payload and isinstance(payload["dtype"], str):
            payload["dtype"] = np.dtype(payload["dtype"]).type
        return cls(**payload)


#: A ready-to-use default configuration instance. Immutable, so it is safe
#: to share across the whole process.
DEFAULT_CONFIG = NumericalConfig()

__all__ = ["NumericalConfig", "DEFAULT_CONFIG", "DEFAULT_DTYPE"]
