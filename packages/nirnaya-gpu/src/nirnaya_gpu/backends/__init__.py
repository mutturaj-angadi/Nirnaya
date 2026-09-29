from .base import Backend
from .cpu import CPUBackend

__all__ = ["Backend", "CPUBackend", "GPUBackend"]


def __getattr__(name):
    # Lazily import GPUBackend so importing nirnaya_gpu.backends never
    # requires cupy to be installed (honest optional dependency).
    if name == "GPUBackend":
        from .gpu import GPUBackend

        return GPUBackend
    raise AttributeError(name)
