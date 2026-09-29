"""
Abstract Backend contract.

Every backend (CPU, CUDA, and any future HIP/SYCL backend) implements
this interface. `nirnaya-solver` should only ever type-hint against
`Backend`, `Device`, `DeviceBuffer`, and `ExecutionContext` -- never
against `CPUBackend`/`GPUBackend` directly -- so that swapping or
adding hardware never touches solver code. See
docs/INTEGRATION_CONTRACT.md for the exact call pattern nirnaya-solver
is expected to use.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import numpy as np

from ..device import Device, DeviceBuffer, ExecutionContext
from ..sparse import SparseCSR


class Backend(abc.ABC):
    """Vendor-neutral compute backend.

    Concrete backends must be honest about `available()`: it should
    perform a *real* capability probe (attempt the import, query the
    runtime/device count) rather than returning a hard-coded value.
    """

    name: str = "abstract"

    # -- discovery / lifecycle -------------------------------------------

    @classmethod
    @abc.abstractmethod
    def available(cls) -> bool:
        """Return True only if this backend can genuinely execute work
        right now on this machine (drivers present, runtime importable,
        at least one device visible, etc.)."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def discover_devices(cls) -> List[Device]:
        """Return the list of real devices this backend can drive.
        Must return an empty list (not raise) if none are available,
        UNLESS the backend's runtime itself is missing, in which case
        callers should have already checked `available()` first."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_context(self, device: Device) -> ExecutionContext:
        raise NotImplementedError

    # -- memory management --------------------------------------------------

    @abc.abstractmethod
    def allocate(
        self, ctx: ExecutionContext, shape: Tuple[int, ...], dtype=np.float64
    ) -> DeviceBuffer:
        raise NotImplementedError

    @abc.abstractmethod
    def to_device(self, ctx: ExecutionContext, array: np.ndarray) -> DeviceBuffer:
        raise NotImplementedError

    @abc.abstractmethod
    def to_host(self, ctx: ExecutionContext, buf: DeviceBuffer) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def upload_sparse(self, ctx: ExecutionContext, mat: SparseCSR):
        """Move a SparseCSR to this backend's native sparse representation
        on-device. Return value is an opaque backend-specific handle,
        passed back into spmv()/residual()/etc."""
        raise NotImplementedError

    # -- core numerical kernels ---------------------------------------------
    # All of these must synchronize internally when they need to return
    # a host-visible scalar; when returning a DeviceBuffer they may be
    # asynchronous (the caller is responsible for ctx.synchronize()
    # before reading results, as documented in ExecutionContext).

    @abc.abstractmethod
    def spmv(self, ctx: ExecutionContext, mat_handle, x: DeviceBuffer) -> DeviceBuffer:
        """y = A @ x for sparse A (CSR)."""
        raise NotImplementedError

    @abc.abstractmethod
    def dot(self, ctx: ExecutionContext, x: DeviceBuffer, y: DeviceBuffer) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def axpy(
        self, ctx: ExecutionContext, alpha: float, x: DeviceBuffer, y: DeviceBuffer
    ) -> DeviceBuffer:
        """y <- alpha * x + y (in place, also returned for convenience)."""
        raise NotImplementedError

    @abc.abstractmethod
    def scale(self, ctx: ExecutionContext, alpha: float, x: DeviceBuffer) -> DeviceBuffer:
        """x <- alpha * x (in place, also returned for convenience)."""
        raise NotImplementedError

    @abc.abstractmethod
    def elementwise(
        self, ctx: ExecutionContext, op: str, x: DeviceBuffer, y: DeviceBuffer
    ) -> DeviceBuffer:
        """op in {'add', 'sub', 'mul', 'div', 'max', 'min'}."""
        raise NotImplementedError

    @abc.abstractmethod
    def reduce(self, ctx: ExecutionContext, x: DeviceBuffer, op: str = "sum") -> float:
        """op in {'sum', 'max', 'min', 'norm2', 'norm_inf'}."""
        raise NotImplementedError

    @abc.abstractmethod
    def residual(
        self, ctx: ExecutionContext, mat_handle, x: DeviceBuffer, b: DeviceBuffer
    ) -> DeviceBuffer:
        """r = b - A @ x."""
        raise NotImplementedError

    @abc.abstractmethod
    def feasibility_check(
        self,
        ctx: ExecutionContext,
        mat_handle,
        x: DeviceBuffer,
        b: DeviceBuffer,
        tol: float,
        sense: str = "le",
    ) -> Tuple[bool, DeviceBuffer]:
        """Parallel feasibility check for a linear system of constraints:
        evaluates A @ x against b elementwise (sense in {'le','ge','eq'})
        within `tol`, and returns (all_satisfied, per_row_violation).

        per_row_violation[i] = max(0, violation amount) for 'le'/'ge',
        or abs(Ax_i - b_i) for 'eq' -- always >= 0, so a caller can also
        find the worst-violated row via reduce(..., op='max') without a
        host round trip.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def argmin(self, ctx: ExecutionContext, x: DeviceBuffer) -> Tuple[int, float]:
        """Return (index, value) of the minimum element of x.

        Used by simplex-style solvers for entering-variable / pivot
        column selection (most negative reduced cost)."""
        raise NotImplementedError

    @abc.abstractmethod
    def ratio_test(
        self,
        ctx: ExecutionContext,
        x: DeviceBuffer,
        direction: DeviceBuffer,
        tol: float = 1e-9,
    ) -> Tuple[Optional[int], float]:
        """Minimum-ratio test used for leaving-variable / basis-change
        selection: over all i with direction[i] > tol, find
        argmin_i x[i] / direction[i]. Returns (index_or_None, ratio),
        where index is None if no i has direction[i] > tol (unbounded)."""
        raise NotImplementedError
