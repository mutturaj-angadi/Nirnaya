"""
Device-agnostic sparse matrix representation.

nirnaya-core's mathematical model (constraint matrices, basis matrices,
etc.) is expressed here as plain host-side CSR arrays -- three flat
numpy arrays plus a shape. This is intentionally the *lowest common
denominator* representation: every backend (CPU/scipy, CUDA/cupy,
future HIP/SYCL) can construct its native sparse type from these three
arrays in O(1) (a view/copy, not a re-parse), and every backend can
hand results back in the same form.

Memory layout
-------------
CSR (Compressed Sparse Row) was chosen over COO/CSC because:
  * Row-major access matches sparse matrix-vector multiplication
    (the dominant operation here), giving coalesced reads per row.
  * It is the native format for scipy.sparse.csr_matrix and
    cupyx.scipy.sparse.csr_matrix, so conversion to/from either
    backend is a direct data copy, not a restructuring pass.
  * indptr (length n_rows+1) lets a GPU kernel assign one thread
    (or warp) per row with O(1) bounds lookup, which is the standard
    approach for GPU SpMV (see docs/ARCHITECTURE.md for the kernel
    design and its tradeoffs vs. CSC/COO/ELL formats).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .exceptions import ShapeMismatchError


@dataclass
class SparseCSR:
    """A host-resident CSR sparse matrix: the wire format between
    nirnaya-core / nirnaya-solver and this package.

    Attributes
    ----------
    indptr : np.ndarray[int32 or int64], shape (n_rows + 1,)
    indices : np.ndarray[int32 or int64], shape (nnz,)
    data : np.ndarray[float64 or float32], shape (nnz,)
    shape : (n_rows, n_cols)
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: Tuple[int, int]

    def __post_init__(self) -> None:
        n_rows, n_cols = self.shape
        if self.indptr.shape[0] != n_rows + 1:
            raise ShapeMismatchError(
                f"indptr length {self.indptr.shape[0]} does not match "
                f"n_rows+1={n_rows + 1}"
            )
        if self.indices.shape[0] != self.data.shape[0]:
            raise ShapeMismatchError(
                f"indices ({self.indices.shape[0]}) and data "
                f"({self.data.shape[0]}) length mismatch"
            )
        if self.indices.size and self.indices.max(initial=0) >= n_cols:
            raise ShapeMismatchError(
                "column index out of bounds for declared shape"
            )

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    @property
    def density(self) -> float:
        n_rows, n_cols = self.shape
        total = n_rows * n_cols
        return self.nnz / total if total else 0.0

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_dense(cls, dense: np.ndarray, dtype=np.float64) -> "SparseCSR":
        dense = np.asarray(dense)
        n_rows, n_cols = dense.shape
        indptr = [0]
        indices = []
        data = []
        for r in range(n_rows):
            row = dense[r]
            nz = np.nonzero(row)[0]
            indices.extend(nz.tolist())
            data.extend(row[nz].tolist())
            indptr.append(len(indices))
        return cls(
            indptr=np.asarray(indptr, dtype=np.int64),
            indices=np.asarray(indices, dtype=np.int64),
            data=np.asarray(data, dtype=dtype),
            shape=(n_rows, n_cols),
        )

    @classmethod
    def from_scipy(cls, mat) -> "SparseCSR":
        mat = mat.tocsr()
        return cls(
            indptr=np.asarray(mat.indptr),
            indices=np.asarray(mat.indices),
            data=np.asarray(mat.data),
            shape=tuple(mat.shape),
        )

    @classmethod
    def random(
        cls,
        n_rows: int,
        n_cols: int,
        density: float,
        seed: int = 0,
        dtype=np.float64,
    ) -> "SparseCSR":
        """Generate a reproducible random sparse matrix for tests/benchmarks.

        Uses only numpy (no scipy dependency) so this works in the
        minimal install. Values are drawn uniformly from [-1, 1] with
        a fixed seed so results are reproducible across runs/machines.
        """
        rng = np.random.default_rng(seed)
        nnz_per_row = max(1, int(round(n_cols * density)))
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        indices_list = []
        data_list = []
        for r in range(n_rows):
            cols = rng.choice(n_cols, size=min(nnz_per_row, n_cols), replace=False)
            cols.sort()
            vals = rng.uniform(-1.0, 1.0, size=cols.shape[0])
            indices_list.append(cols)
            data_list.append(vals)
            indptr[r + 1] = indptr[r] + cols.shape[0]
        indices = np.concatenate(indices_list).astype(np.int64)
        data = np.concatenate(data_list).astype(dtype)
        return cls(indptr=indptr, indices=indices, data=data, shape=(n_rows, n_cols))

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.data.dtype)
        n_rows = self.shape[0]
        for r in range(n_rows):
            start, end = self.indptr[r], self.indptr[r + 1]
            out[r, self.indices[start:end]] = self.data[start:end]
        return out

    def to_scipy(self):
        try:
            import scipy.sparse as sp
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "scipy is required for to_scipy(); install nirnaya-gpu[scipy]"
            ) from e
        return sp.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)
