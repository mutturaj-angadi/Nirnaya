"""
CPU backend.

Always available -- this is the reliable fallback every other backend
is checked against. Uses scipy.sparse for SpMV when scipy is installed
(fast, well-tested C implementation); otherwise falls back to a pure
numpy CSR matvec (np.add.at-based), so nirnaya-gpu works with only
numpy installed. Both paths are exact for the same inputs (up to
floating-point summation order, see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import platform
from typing import List, Optional, Tuple

import numpy as np

from ..device import Device, DeviceBuffer, DeviceType, ExecutionContext
from ..exceptions import BackendNotImplementedError, ShapeMismatchError
from ..sparse import SparseCSR
from .base import Backend

try:
    import scipy.sparse as _sp

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


class CPUBuffer(DeviceBuffer):
    def __init__(self, device: Device, array: np.ndarray):
        super().__init__(device, tuple(array.shape), array.dtype)
        self.array = array

    def to_host(self) -> np.ndarray:
        return self.array.copy()

    def copy_from_host(self, array: np.ndarray) -> None:
        if array.shape != self.shape:
            raise ShapeMismatchError(f"{array.shape} != {self.shape}")
        self.array[...] = array


class CPUContext(ExecutionContext):
    """CPU execution is synchronous, so synchronize() is a no-op --
    but it still exists so solver code is backend-agnostic."""

    def synchronize(self) -> None:
        return None


def _cpu_model_name() -> str:
    try:
        return platform.processor() or platform.machine() or "unknown CPU"
    except Exception:  # pragma: no cover
        return "unknown CPU"


class CPUBackend(Backend):
    name = "cpu"

    @classmethod
    def available(cls) -> bool:
        # The CPU backend only needs numpy, which is a hard dependency
        # of this package, so it is always available.
        return True

    @classmethod
    def discover_devices(cls) -> List[Device]:
        try:
            import os

            n_threads = os.cpu_count() or 1
        except Exception:  # pragma: no cover
            n_threads = 1
        return [
            Device(
                device_id=0,
                device_type=DeviceType.CPU,
                backend_name="cpu",
                vendor_name="Generic CPU",
                model_name=f"{_cpu_model_name()} ({n_threads} logical cores)",
                total_memory_bytes=_query_total_ram(),
                compute_capability=None,
                is_gpu=False,
            )
        ]

    def create_context(self, device: Device) -> ExecutionContext:
        return CPUContext(device)

    # -- memory ---------------------------------------------------------

    def allocate(self, ctx, shape, dtype=np.float64) -> DeviceBuffer:
        return CPUBuffer(ctx.device, np.zeros(shape, dtype=dtype))

    def to_device(self, ctx, array: np.ndarray) -> DeviceBuffer:
        return CPUBuffer(ctx.device, np.array(array, copy=True))

    def to_host(self, ctx, buf: DeviceBuffer) -> np.ndarray:
        assert isinstance(buf, CPUBuffer)
        return buf.to_host()

    def upload_sparse(self, ctx, mat: SparseCSR):
        if _HAS_SCIPY:
            return _sp.csr_matrix((mat.data, mat.indices, mat.indptr), shape=mat.shape)
        # Pure-numpy fallback handle: just keep the CSR arrays.
        return mat

    # -- kernels ----------------------------------------------------------

    def spmv(self, ctx, mat_handle, x: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CPUBuffer)
        if _HAS_SCIPY and isinstance(mat_handle, _sp.csr_matrix):
            y = mat_handle.dot(x.array)
        else:
            y = _csr_matvec_numpy(mat_handle, x.array)
        return CPUBuffer(ctx.device, y)

    def dot(self, ctx, x: DeviceBuffer, y: DeviceBuffer) -> float:
        assert isinstance(x, CPUBuffer) and isinstance(y, CPUBuffer)
        return float(np.dot(x.array, y.array))

    def axpy(self, ctx, alpha: float, x: DeviceBuffer, y: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CPUBuffer) and isinstance(y, CPUBuffer)
        y.array += alpha * x.array
        return y

    def scale(self, ctx, alpha: float, x: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CPUBuffer)
        x.array *= alpha
        return x

    def elementwise(self, ctx, op: str, x: DeviceBuffer, y: DeviceBuffer) -> DeviceBuffer:
        assert isinstance(x, CPUBuffer) and isinstance(y, CPUBuffer)
        fn = {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": np.divide,
            "max": np.maximum,
            "min": np.minimum,
        }.get(op)
        if fn is None:
            raise BackendNotImplementedError(f"unknown elementwise op {op!r}")
        return CPUBuffer(ctx.device, fn(x.array, y.array))

    def reduce(self, ctx, x: DeviceBuffer, op: str = "sum") -> float:
        assert isinstance(x, CPUBuffer)
        a = x.array
        if op == "sum":
            return float(np.sum(a))
        if op == "max":
            return float(np.max(a))
        if op == "min":
            return float(np.min(a))
        if op == "norm2":
            return float(np.linalg.norm(a))
        if op == "norm_inf":
            return float(np.max(np.abs(a))) if a.size else 0.0
        raise BackendNotImplementedError(f"unknown reduce op {op!r}")

    def residual(self, ctx, mat_handle, x: DeviceBuffer, b: DeviceBuffer) -> DeviceBuffer:
        ax = self.spmv(ctx, mat_handle, x)
        assert isinstance(b, CPUBuffer)
        return CPUBuffer(ctx.device, b.array - ax.array)

    def feasibility_check(
        self, ctx, mat_handle, x: DeviceBuffer, b: DeviceBuffer, tol: float, sense: str = "le"
    ) -> Tuple[bool, DeviceBuffer]:
        ax = self.spmv(ctx, mat_handle, x).array
        assert isinstance(b, CPUBuffer)
        if sense == "le":
            viol = np.maximum(ax - b.array - tol, 0.0)
        elif sense == "ge":
            viol = np.maximum(b.array - ax - tol, 0.0)
        elif sense == "eq":
            viol = np.abs(ax - b.array)
            viol = np.where(viol <= tol, 0.0, viol)
        else:
            raise BackendNotImplementedError(f"unknown sense {sense!r}")
        satisfied = bool(np.all(viol <= tol if sense != "eq" else viol == 0.0))
        return satisfied, CPUBuffer(ctx.device, viol)

    def argmin(self, ctx, x: DeviceBuffer) -> Tuple[int, float]:
        assert isinstance(x, CPUBuffer)
        idx = int(np.argmin(x.array))
        return idx, float(x.array[idx])

    def ratio_test(
        self, ctx, x: DeviceBuffer, direction: DeviceBuffer, tol: float = 1e-9
    ) -> Tuple[Optional[int], float]:
        assert isinstance(x, CPUBuffer) and isinstance(direction, CPUBuffer)
        d = direction.array
        mask = d > tol
        if not np.any(mask):
            return None, float("inf")
        ratios = np.where(mask, x.array / np.where(mask, d, 1.0), np.inf)
        idx = int(np.argmin(ratios))
        return idx, float(ratios[idx])


def _csr_matvec_numpy(mat: SparseCSR, x: np.ndarray) -> np.ndarray:
    """Pure-numpy CSR matvec, used only when scipy is not installed.

    Vectorized via np.bincount over per-nonzero row ids rather than a
    Python loop over rows/nnz, and rather than np.add.reduceat (which
    has awkward edge-case semantics for empty/trailing rows). bincount
    with `minlength=n_rows` handles empty rows, an empty matrix, and
    duplicate row ids uniformly and correctly. For performance-critical
    use, install the `scipy` extra instead.
    """
    n_rows = mat.shape[0]
    out_dtype = np.result_type(mat.data.dtype, x.dtype)
    if mat.data.size == 0:
        return np.zeros(n_rows, dtype=out_dtype)
    row_counts = np.diff(mat.indptr)
    row_ids = np.repeat(np.arange(n_rows), row_counts)
    prod = mat.data * x[mat.indices]
    y = np.bincount(row_ids, weights=prod.astype(np.float64), minlength=n_rows)
    return y.astype(out_dtype, copy=False)


def _query_total_ram() -> Optional[int]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass
    return None
