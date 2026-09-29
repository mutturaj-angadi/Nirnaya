"""
Structured error hierarchy.

Every error nirnaya-api raises carries:

    * ``code``     - a stable machine-readable string (e.g. "VALIDATION_ERROR")
    * ``message``  - a human-readable message
    * ``details``  - optional structured extra context (dict)

The REST layer (:mod:`nirnaya_api.server`) catches these and renders them as
a single, consistent JSON error envelope (see docs/API.md). The CLI
(:mod:`nirnaya_api.cli`) catches them and renders either a human-readable
message or a JSON envelope, depending on ``--json``.

No internal stack traces, file paths, or dependency internals are ever
included in the rendered message - only in server-side logs.
"""

from __future__ import annotations

from typing import Any, Optional


class NirnayaAPIError(Exception):
    """Base class for all errors raised by nirnaya-api.

    Subclasses should set ``code`` and an appropriate ``http_status``.
    """

    code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ValidationError(NirnayaAPIError):
    """The submitted model / request payload is malformed or inconsistent."""

    code = "VALIDATION_ERROR"
    http_status = 422


class PayloadTooLargeError(NirnayaAPIError):
    """Request payload exceeded configured size limits."""

    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class ResourceLimitError(NirnayaAPIError):
    """A configured resource limit (variables, constraints, memory) was hit."""

    code = "RESOURCE_LIMIT_EXCEEDED"
    http_status = 422


class SolveTimeoutError(NirnayaAPIError):
    """The solve did not complete within the configured time limit."""

    code = "SOLVE_TIMEOUT"
    http_status = 200  # a timed-out solve is still a well-formed *response*


class DependencyUnavailableError(NirnayaAPIError):
    """A required backend package (core/presolve/solver/gpu) is not installed
    or not importable, and no permitted fallback applies."""

    code = "DEPENDENCY_UNAVAILABLE"
    http_status = 503


class DeviceUnavailableError(NirnayaAPIError):
    """A specifically-requested device/backend does not exist or is down."""

    code = "DEVICE_UNAVAILABLE"
    http_status = 409


class JobNotFoundError(NirnayaAPIError):
    """An async job id was not found (or has expired)."""

    code = "JOB_NOT_FOUND"
    http_status = 404


class NotFoundError(NirnayaAPIError):
    code = "NOT_FOUND"
    http_status = 404


def unexpected_error(exc: Exception) -> NirnayaAPIError:
    """Wrap an unexpected, non-NirnayaAPIError exception for safe rendering.

    Ensures internal exception text (which may contain paths, tracebacks,
    or dependency internals) never reaches the client verbatim; only the
    exception's type name is surfaced.
    """
    return NirnayaAPIError(
        f"An internal error occurred ({type(exc).__name__}).",
        details={},
    )
