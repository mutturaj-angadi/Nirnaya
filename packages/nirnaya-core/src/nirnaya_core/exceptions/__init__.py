"""
nirnaya_core.exceptions
========================

Machine-readable exception hierarchy for Nirnaya Core.

Every exception raised by this package derives from :class:`NirnayaError`
and carries a stable, machine-readable ``code`` string plus a ``details``
dictionary of structured context. Downstream modules (Parts 2-6) may match
on ``code`` without depending on exception message text, which is free to
change between releases.

Design rules
------------
* Exception *types* and *codes* are part of the public API and are covered
  by the stability guarantees in ``docs/INTEGRATION_CONTRACT.md``.
* Exception *messages* (``str(exc)``) are for humans and are NOT guaranteed
  to be stable across versions.
* ``details`` values must be JSON-serializable (str, int, float, bool,
  None, list, dict) so that errors can cross process/service boundaries
  (e.g. a future gRPC/HTTP solver service in Parts 2-6) without loss.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class NirnayaError(Exception):
    """Base class for every exception raised by nirnaya_core.

    Attributes:
        code: Stable machine-readable identifier, e.g. ``"E_DIM_MISMATCH"``.
            Always prefixed with ``E_`` for errors. Part of the public API.
        details: Structured, JSON-serializable context about the failure.
    """

    #: Default machine-readable code for this error class. Subclasses
    #: override this. Instances may also override via the constructor.
    code: str = "E_NIRNAYA"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = dict(details) if details else {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this error."""
        return {
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            return f"[{self.code}] {self.message} | details={self.details}"
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ValidationError(NirnayaError):
    """Base class for all model/data validation failures.

    Raised by :mod:`nirnaya_core.validation` when a :class:`~nirnaya_core.model.Model`
    or its components fail structural or numerical validation *before* being
    handed to a solver backend. See subclasses in
    :mod:`nirnaya_core.validation.errors` for specific failure modes.
    """

    code = "E_VALIDATION"


class SerializationError(NirnayaError):
    """Raised when serializing or deserializing a model/result fails."""

    code = "E_SERIALIZATION"


class ConfigurationError(NirnayaError):
    """Raised when a :class:`~nirnaya_core.numeric.NumericalConfig` or other
    configuration object is constructed with inconsistent or invalid values.
    """

    code = "E_CONFIGURATION"


class ImmutabilityError(NirnayaError):
    """Raised when mutation is attempted on a frozen/built object.

    For example, attempting to add a variable to a :class:`~nirnaya_core.model.Model`
    after :meth:`~nirnaya_core.model.Model.build` has produced a
    :class:`~nirnaya_core.model.ProblemData` snapshot that callers are relying on,
    without explicitly invalidating that snapshot.
    """

    code = "E_IMMUTABLE"


__all__ = [
    "NirnayaError",
    "ValidationError",
    "SerializationError",
    "ConfigurationError",
    "ImmutabilityError",
]
