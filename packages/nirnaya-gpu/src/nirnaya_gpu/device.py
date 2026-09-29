"""
Device abstraction layer for nirnaya-gpu.

This module defines the vendor-neutral contract that every backend
(CPU, CUDA, and any future HIP/SYCL backend) must implement. Nothing
in nirnaya-core or nirnaya-solver should ever import a vendor-specific
module (e.g. `cupy`, `torch`, `numba.cuda`) directly -- it should only
ever see `Device`, `DeviceBuffer`, `ExecutionContext`, and `Kernel`.

Design principles
------------------
1. Honesty over convenience: a Device's `is_gpu` / `backend_name`
   fields must reflect what actually executed the work, never what
   was requested. If CUDA is unavailable, code must fall back to the
   CPU backend and *say so* -- never silently claim GPU execution.
2. No hidden global state: an ExecutionContext is created explicitly
   from a Device and carries all backend-specific handles (streams,
   cuBLAS/cuSPARSE handles, thread pools, etc.).
3. Buffers are opaque handles. Callers never assume a DeviceBuffer's
   underlying storage; they only move data via `to_device`/`to_host`.
"""

from __future__ import annotations

import abc
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


class DeviceType(enum.Enum):
    """Concrete kinds of compute device this layer knows how to target."""

    CPU = "cpu"
    CUDA = "cuda"
    # Placeholders for future backends. Adding one of these does NOT
    # require touching nirnaya-solver -- only a new Backend subclass
    # (see backends/base.py) registered in registry.py.
    HIP = "hip"
    SYCL = "sycl"


@dataclass(frozen=True)
class Device:
    """
    A concrete, discovered compute device.

    Instances are only ever produced by `Backend.discover_devices()`
    after a real capability probe (e.g. actually importing cupy and
    querying `cupy.cuda.runtime.getDeviceCount()`). Nothing here is
    guessed or hard-coded.
    """

    device_id: int
    device_type: DeviceType
    backend_name: str          # e.g. "cpu", "cuda"
    vendor_name: str           # e.g. "Generic CPU", "NVIDIA", "AMD"
    model_name: str            # e.g. CPU model string or GPU name
    total_memory_bytes: Optional[int]  # None if unknown/unmeasurable
    compute_capability: Optional[str] = None  # e.g. "8.6" for CUDA
    is_gpu: bool = False

    def describe(self) -> str:
        mem = (
            f"{self.total_memory_bytes / (1024**3):.2f} GiB"
            if self.total_memory_bytes
            else "unknown"
        )
        return (
            f"[{self.backend_name}:{self.device_id}] {self.vendor_name} "
            f"{self.model_name} (gpu={self.is_gpu}, memory={mem})"
        )


class DeviceBuffer(abc.ABC):
    """
    Opaque handle to memory resident on a Device.

    Subclasses (CPUBuffer, CUDABuffer, ...) wrap the actual storage
    (a numpy ndarray, a cupy ndarray, ...). Calling code must not
    reach into `.raw` except inside a backend implementation.
    """

    def __init__(self, device: Device, shape: Tuple[int, ...], dtype: np.dtype):
        self.device = device
        self.shape = shape
        self.dtype = np.dtype(dtype)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    @abc.abstractmethod
    def to_host(self) -> np.ndarray:
        """Copy this buffer back to a host (numpy) array."""
        raise NotImplementedError

    @abc.abstractmethod
    def copy_from_host(self, array: np.ndarray) -> None:
        """Overwrite this buffer's contents from a host array."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<{self.__class__.__name__} shape={self.shape} "
            f"dtype={self.dtype} device={self.device.describe()}>"
        )


@dataclass
class TransferStats:
    """Real, measured host<->device transfer accounting."""

    bytes_transferred: int = 0
    seconds: float = 0.0

    @property
    def gbps(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return (self.bytes_transferred / 1e9) / self.seconds


class ExecutionContext(abc.ABC):
    """
    A live handle for issuing work on a specific Device.

    Owns whatever backend-specific execution state is needed (a CUDA
    stream, a thread pool, etc.) and is responsible for synchronizing
    before any timing or correctness-sensitive read of results.
    """

    def __init__(self, device: Device):
        self.device = device
        self.transfer_stats = TransferStats()

    @abc.abstractmethod
    def synchronize(self) -> None:
        """Block until all outstanding work on this context has completed."""
        raise NotImplementedError

    def __enter__(self) -> "ExecutionContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.synchronize()

    def _record_transfer(self, nbytes: int, seconds: float) -> None:
        self.transfer_stats.bytes_transferred += nbytes
        self.transfer_stats.seconds += seconds


class Kernel(abc.ABC):
    """
    A named, backend-bound unit of computation.

    Kernels are looked up from a Backend (see backends/base.py) and
    invoked with DeviceBuffers + an ExecutionContext. This class
    exists mainly so tooling/benchmarks can introspect kernel names
    and report which backend actually ran them, rather than assuming.
    """

    name: str = "unnamed_kernel"

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    @abc.abstractmethod
    def __call__(self, ctx: ExecutionContext, *args, **kwargs) -> Any:
        raise NotImplementedError

    def timed_call(self, ctx: ExecutionContext, *args, **kwargs):
        """
        Execute the kernel and return (result, elapsed_seconds), where
        elapsed_seconds is measured *after* an explicit synchronize()
        so GPU async dispatch can't make a kernel look artificially fast.
        """
        start = time.perf_counter()
        result = self(ctx, *args, **kwargs)
        ctx.synchronize()
        elapsed = time.perf_counter() - start
        return result, elapsed
