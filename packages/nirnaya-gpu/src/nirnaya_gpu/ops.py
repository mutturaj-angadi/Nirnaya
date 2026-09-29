"""
High-level, numpy-in / numpy-out convenience API.

`backends/base.py` defines the low-level Device/Buffer/Context
contract (needed so solver-internal hot loops can avoid host<->device
transfers between iterations). This module wraps that contract in a
`Session` object that nirnaya-solver can use without ever touching a
DeviceBuffer directly, for call sites where convenience matters more
than avoiding a transfer (e.g. one-off residual checks, setup code).

See docs/INTEGRATION_CONTRACT.md for guidance on when to use `Session`
vs. the raw Backend/Buffer API.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .backends.base import Backend
from .device import Device, ExecutionContext
from .registry import get_backend
from .sparse import SparseCSR


class Session:
    """A backend + device + context bundle, with numpy-friendly methods.

    Example
    -------
    >>> session = Session()  # picks GPU if available, else CPU
    >>> y = session.spmv(A, x)          # A: SparseCSR, x: np.ndarray
    >>> r = session.residual(A, x, b)
    >>> ok, worst = session.feasibility(A, x, b, tol=1e-6)
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        device: Optional[Device] = None,
        prefer_gpu: bool = True,
    ):
        self.backend: Backend = backend or get_backend(prefer_gpu=prefer_gpu)
        devices = self.backend.discover_devices()
        if device is not None:
            self.device = device
        elif devices:
            self.device = devices[0]
        else:  # pragma: no cover - CPU backend always reports >=1 device
            raise RuntimeError(
                f"backend {self.backend.name!r} reported no usable devices"
            )
        self.ctx: ExecutionContext = self.backend.create_context(self.device)

    # -- introspection --------------------------------------------------

    @property
    def is_gpu(self) -> bool:
        return self.device.is_gpu

    def describe(self) -> str:
        return f"Session(backend={self.backend.name}, device={self.device.describe()})"

    # -- sparse matrix caching -------------------------------------------

    def upload_matrix(self, mat: SparseCSR):
        """Upload a SparseCSR once and reuse the returned handle across
        many spmv/residual/feasibility calls to avoid re-transferring it
        every call (see docs/ARCHITECTURE.md, "transfer costs")."""
        return self.backend.upload_sparse(self.ctx, mat)

    # -- numpy-in/out kernels ---------------------------------------------

    def spmv(self, mat, x: np.ndarray) -> np.ndarray:
        mat_handle = mat if not isinstance(mat, SparseCSR) else self.upload_matrix(mat)
        xb = self.backend.to_device(self.ctx, x)
        yb = self.backend.spmv(self.ctx, mat_handle, xb)
        self.ctx.synchronize()
        return self.backend.to_host(self.ctx, yb)

    def dot(self, x: np.ndarray, y: np.ndarray) -> float:
        xb = self.backend.to_device(self.ctx, x)
        yb = self.backend.to_device(self.ctx, y)
        return self.backend.dot(self.ctx, xb, yb)

    def axpy(self, alpha: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xb = self.backend.to_device(self.ctx, x)
        yb = self.backend.to_device(self.ctx, y)
        out = self.backend.axpy(self.ctx, alpha, xb, yb)
        self.ctx.synchronize()
        return self.backend.to_host(self.ctx, out)

    def scale(self, alpha: float, x: np.ndarray) -> np.ndarray:
        xb = self.backend.to_device(self.ctx, x)
        out = self.backend.scale(self.ctx, alpha, xb)
        self.ctx.synchronize()
        return self.backend.to_host(self.ctx, out)

    def elementwise(self, op: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xb = self.backend.to_device(self.ctx, x)
        yb = self.backend.to_device(self.ctx, y)
        out = self.backend.elementwise(self.ctx, op, xb, yb)
        self.ctx.synchronize()
        return self.backend.to_host(self.ctx, out)

    def reduce(self, x: np.ndarray, op: str = "sum") -> float:
        xb = self.backend.to_device(self.ctx, x)
        return self.backend.reduce(self.ctx, xb, op)

    def residual(self, mat, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        mat_handle = mat if not isinstance(mat, SparseCSR) else self.upload_matrix(mat)
        xb = self.backend.to_device(self.ctx, x)
        bb = self.backend.to_device(self.ctx, b)
        rb = self.backend.residual(self.ctx, mat_handle, xb, bb)
        self.ctx.synchronize()
        return self.backend.to_host(self.ctx, rb)

    def feasibility(
        self, mat, x: np.ndarray, b: np.ndarray, tol: float, sense: str = "le"
    ) -> Tuple[bool, np.ndarray]:
        mat_handle = mat if not isinstance(mat, SparseCSR) else self.upload_matrix(mat)
        xb = self.backend.to_device(self.ctx, x)
        bb = self.backend.to_device(self.ctx, b)
        ok, viol = self.backend.feasibility_check(self.ctx, mat_handle, xb, bb, tol, sense)
        self.ctx.synchronize()
        return ok, self.backend.to_host(self.ctx, viol)

    def argmin(self, x: np.ndarray) -> Tuple[int, float]:
        xb = self.backend.to_device(self.ctx, x)
        return self.backend.argmin(self.ctx, xb)

    def ratio_test(
        self, x: np.ndarray, direction: np.ndarray, tol: float = 1e-9
    ) -> Tuple[Optional[int], float]:
        xb = self.backend.to_device(self.ctx, x)
        db = self.backend.to_device(self.ctx, direction)
        return self.backend.ratio_test(self.ctx, xb, db, tol)
