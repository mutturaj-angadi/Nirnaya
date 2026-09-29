"""Exception hierarchy for nirnaya-gpu.

Kept small and explicit so nirnaya-solver can catch precisely what it
needs to (e.g. "fall back to CPU") without swallowing unrelated bugs.
"""


class NirnayaGPUError(Exception):
    """Base class for all nirnaya-gpu errors."""


class DeviceUnavailableError(NirnayaGPUError):
    """Raised when a requested device/backend genuinely cannot be used.

    This is raised instead of silently degrading, so callers can decide
    whether to fall back to CPU explicitly. nirnaya-gpu never pretends
    a GPU backend is active when it is not.
    """


class BackendNotImplementedError(NirnayaGPUError):
    """Raised when an operation is not implemented for a given backend."""


class ShapeMismatchError(NirnayaGPUError):
    """Raised when operand shapes are incompatible for an operation."""


class ToleranceExceededError(NirnayaGPUError):
    """Raised by correctness-checking utilities when CPU/GPU results
    disagree by more than a configured tolerance."""
