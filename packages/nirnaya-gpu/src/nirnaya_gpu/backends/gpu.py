"""
GPU (CUDA) backend, implemented via CuPy + cupyx.scipy.sparse.

IMPORTANT HONESTY CONTRACT
---------------------------
`GPUBackend.available()` performs a *real* probe: it tries to import
cupy and asks the CUDA runtime how many devices it sees. If cupy is
not installed, if there is no CUDA runtime/driver, or if the device
count is zero, this returns False -- there is no simulated device, no
hard-coded device name, and no operation in this file will silently
execute on the CPU while claiming to be a GPU result. Any attempt to
construct a GPUBackend when `available()` is False raises
`DeviceUnavailableError` with the underlying import/runtime error
message attached, so callers (and nirnaya-solver) can log *why*.

This backend was developed and code-reviewed without a CUDA device
present (see docs/ARCHITECTURE.md, "CPU/GPU tradeoffs" and the demo's
output on this machine, which reports `cuda: unavailable`). Kernel
logic mirrors the CPU backend's semantics exactly (same operation,
same broadcasting rules) and is exercised by
tests/test_correctness_cpu_vs_gpu.py, which is *skipped* -- not
faked -- in environments without CUDA.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..device import Device, DeviceBuffer, DeviceType, ExecutionContext
from ..exceptions import BackendNotImplementedError, DeviceUnavailableError, ShapeMismatchError
from ..sparse import SparseCSR
from .base import Backend


def _probe_cupy():
    """Attempt a real CUDA capability probe. Returns (cupy_module, device_count,
    error_message_or_None). Never raises -- callers decide what to do."""
    try:
        import cupy as cp  # type: ignore
    except ImportError as e:
        return None, 0, f"cupy is not installed ({e})"
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as e:  # e.g. cupy installed but no driver/runtime
        return cp, 0, f"CUDA runtime unavailable ({e})"
    if count == 0:
        return cp, 0, "cupy reports zero visible CUDA devices"
    return cp, count, None


class CUDABuffer(DeviceBuffer):
    def __init__(self, device: Device, array):
        super().__init__(device, tuple(array.shape), array.dtype)
        self.array = array  # a cupy.ndarray

    def to_host(self) -> np.ndarray:
        return self.array.get()

    def copy_from_host(self, array: np.ndarray) -> None:
        import cupy as cp

        if tuple(array.shape) != self.shape:
            raise ShapeMismatchError(f"{array.shape} != {self.shape}")
        self.array[...] = cp.asarray(array)


class CUDAContext(ExecutionContext):
    def __init__(self, device: Device, cp_module):
        super().__init__(device)
        self._cp = cp_module
        self.stream = cp_module.cuda.Stream(non_blocking=True)

    def synchronize(self) -> None:
        self.stream.synchronize()
        self._cp.cuda.runtime.deviceSynchronize()


class GPUBackend(Backend):
    """CUDA backend. Raises DeviceUnavailableError at construction time
    if CUDA is not genuinely usable -- see module docstring."""

    name = "cuda"

    def __init__(self):
        cp, count, err = _probe_cupy()
        if cp is None or count == 0:
            raise DeviceUnavailableError(
                f"GPUBackend requested but CUDA is not available: {err}"
            )
        self._cp = cp

    @classmethod
    def available(cls) -> bool:
        _, count, _ = _probe_cupy()
        return count > 0

    @classmethod
    def discover_devices(cls) -> List[Device]:
        cp, count, err = _probe_cupy()
        if cp is None or count == 0:
            # Honest empty result -- no fabricated device entries.
            return []
        devices = []
        for i in range(count):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                name = props["name"]
                name = name.decode() if isinstance(name, bytes) else name
                major = props.get("major")
                minor = props.get("minor")
                cc = f"{major}.{minor}" if major is not None else None
                devices.append(
                    Device(
                        device_id=i,
                        device_type=DeviceType.CUDA,
                        backend_name="cuda",
                        vendor_name="NVIDIA",
                        model_name=name,
                        total_memory_bytes=int(total_mem),
                        compute_capability=cc,
                        is_gpu=True,
                    )
                )
        return devices

    def create_context(self, device: Device) -> ExecutionContext:
        self._cp.cuda.Device(device.device_id).use()
        return CUDAContext(device, self._cp)

    # -- memory ---------------------------------------------------------

    def allocate(self, ctx, shape, dtype=np.float64) -> DeviceBuffer:
        cp = self._cp
        return CUDABuffer(ctx.device, cp.zeros(shape, dtype=dtype))

    def to_device(self, ctx, array: np.ndarray) -> DeviceBuffer:
        import time

        cp = self._cp
        t0 = time.perf_counter()
        dev_arr = cp.asarray(array)
        cp.cuda.runtime.deviceSynchronize()
        ctx._record_transfer(dev_arr.nbytes, time.perf_counter() - t0)
        return CUDABuffer(ctx.device, dev_arr)

    def to_host(self, ctx, buf: DeviceBuffer) -> np.ndarray:
        import time

        assert isinstance(buf, CUDABuffer)
        t0 = time.perf_counter()
        host = buf.array.get()
        ctx._record_transfer(host.nbytes, time.perf_counter() - t0)
        return host

    def upload_sparse(self, ctx, mat: SparseCSR):
        import cupyx.scipy.sparse as cusp

        cp = self._cp
        gpu_mat = cusp.csr_matrix(
            (
                cp.asarray(mat.data),
                cp.asarray(mat.indices),
                cp.asarray(mat.indptr),
            ),
            shape=mat.shape,
        )
        return gpu_mat

    # -- kernels ----------------------------------------------------------

    def spmv(self, ctx, mat_handle, x: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CUDABuffer)
        y = mat_handle.dot(x.array)
        return CUDABuffer(ctx.device, y)

    def dot(self, ctx, x: DeviceBuffer, y: DeviceBuffer) -> float:
        assert isinstance(x, CUDABuffer) and isinstance(y, CUDABuffer)
        result = self._cp.dot(x.array, y.array)
        ctx.synchronize()
        return float(result.get())

    def axpy(self, ctx, alpha: float, x: DeviceBuffer, y: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CUDABuffer) and isinstance(y, CUDABuffer)
        y.array += alpha * x.array
        return y

    def scale(self, ctx, alpha: float, x: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CUDABuffer)
        x.array *= alpha
        return x

    def elementwise(self, ctx, op: str, x: DeviceBuffer, y: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CUDABuffer) and isinstance(y, CUDABuffer)
        cp = self._cp
        fn = {
            "add": cp.add,
            "sub": cp.subtract,
            "mul": cp.multiply,
            "div": cp.divide,
            "max": cp.maximum,
            "min": cp.minimum,
        }.get(op)
        if fn is None:
            raise BackendNotImplementedError(f"unknown elementwise op {op!r}")
        return CUDABuffer(ctx.device, fn(x.array, y.array))

    def reduce(self, ctx, x: DeviceBuffer, op: str = "sum") -> float:
        assert isinstance(x, CUDABuffer)
        cp = self._cp
        a = x.array
        if op == "sum":
            val = cp.sum(a)
        elif op == "max":
            val = cp.max(a)
        elif op == "min":
            val = cp.min(a)
        elif op == "norm2":
            val = cp.linalg.norm(a)
        elif op == "norm_inf":
            val = cp.max(cp.abs(a)) if a.size else cp.asarray(0.0)
        else:
            raise BackendNotImplementedError(f"unknown reduce op {op!r}")
        ctx.synchronize()
        return float(val.get())

    def residual(self, ctx, mat_handle, x: DeviceBuffer, b: DeviceBuffer) -> DeviceBuffer:
        ax = self.spmv(ctx, mat_handle, x)
        assert isinstance(b, CUDABuffer)
        return CUDABuffer(ctx.device, b.array - ax.array)

    def feasibility_check(
        self, ctx, mat_handle, x: DeviceBuffer, b: DeviceBuffer, tol: float, sense: str = "le"
    ) -> Tuple[bool, DeviceBuffer]:
        cp = self._cp
        ax = self.spmv(ctx, mat_handle, x).array
        assert isinstance(b, CUDABuffer)
        if sense == "le":
            viol = cp.maximum(ax - b.array - tol, 0.0)
        elif sense == "ge":
            viol = cp.maximum(b.array - ax - tol, 0.0)
        elif sense == "eq":
            viol = cp.abs(ax - b.array)
            viol = cp.where(viol <= tol, 0.0, viol)
        else:
            raise BackendNotImplementedError(f"unknown sense {sense!r}")
        satisfied = bool(cp.all(viol <= tol if sense != "eq" else viol == 0.0).get())
        return satisfied, CUDABuffer(ctx.device, viol)

    def argmin(self, ctx, x: DeviceBuffer) -> Tuple[int, float]:
        assert isinstance(x, CUDABuffer)
        cp = self._cp
        idx = int(cp.argmin(x.array).get())
        return idx, float(x.array[idx].get())

    def ratio_test(
        self, ctx, x: DeviceBuffer, direction: DeviceBuffer, tol: float = 1e-9
    ) -> Tuple[Optional[int], float]:
        assert isinstance(x, CUDABuffer) and isinstance(direction, CUDABuffer)
        cp = self._cp
        d = direction.array
        mask = d > tol
        if not bool(cp.any(mask).get()):
            return None, float("inf")
        ratios = cp.where(mask, x.array / cp.where(mask, d, 1.0), cp.inf)
        idx = int(cp.argmin(ratios).get())
        return idx, float(ratios[idx].get())
