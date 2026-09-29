"""
nirnaya-gpu: device-independent hardware acceleration layer for Nirnaya.

Public API surface (what nirnaya-core / nirnaya-solver should import):

    from nirnaya_gpu import Session, SparseCSR
    from nirnaya_gpu.registry import discover_all, get_backend, get_backend_by_name
    from nirnaya_gpu.device import Device, DeviceType

Everything else (backends.cpu, backends.gpu, the raw Backend interface)
is available for advanced/performance-critical use but is not required
for ordinary solver code. See docs/INTEGRATION_CONTRACT.md.
"""

from .device import Device, DeviceBuffer, DeviceType, ExecutionContext, Kernel
from .exceptions import (
    BackendNotImplementedError,
    DeviceUnavailableError,
    NirnayaGPUError,
    ShapeMismatchError,
    ToleranceExceededError,
)
from .ops import Session
from .registry import discover_all, get_backend, get_backend_by_name
from .sparse import SparseCSR

__version__ = "0.1.0"

__all__ = [
    "Session",
    "SparseCSR",
    "Device",
    "DeviceBuffer",
    "DeviceType",
    "ExecutionContext",
    "Kernel",
    "discover_all",
    "get_backend",
    "get_backend_by_name",
    "NirnayaGPUError",
    "DeviceUnavailableError",
    "BackendNotImplementedError",
    "ShapeMismatchError",
    "ToleranceExceededError",
    "__version__",
]
