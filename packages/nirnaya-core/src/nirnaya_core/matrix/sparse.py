"""
Sparse matrix representation used throughout Nirnaya Core.

Why a wrapper instead of raw ``scipy.sparse``:

* It fixes a single canonical dtype (see :mod:`nirnaya_core.numeric.config`) and
  refuses silent dtype-narrowing conversions (e.g. complex -> float,
  float64 -> float32) that could otherwise happen implicitly inside scipy.
* It gives Parts 2-6 (which may replace scipy with a custom CSR
  implementation, a GPU CSR/CSC format, or a native C++/CUDA backend) a
  single, stable, documented surface (:class:`SparseMatrix`) to target,
  instead of depending on ``scipy.sparse`` internals directly.
* It provides deterministic serialization (COO triplets sorted by
  ``(row, col)``) independent of whatever internal ordering scipy happens
  to use.

Dense arrays are never used as the default storage. Conversion to dense
(:meth:`SparseMatrix.to_dense`) is always explicit and is intended only for
small models, debugging, or tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from nirnaya_core.exceptions import ValidationError
from nirnaya_core.numeric.config import DEFAULT_DTYPE

Number = float


class SparseMatrixError(ValidationError):
    """Raised when constructing or manipulating a :class:`SparseMatrix` fails
    structural or numerical validation (bad shape, non-finite data,
    out-of-range indices, dtype mismatch).
    """

    code = "E_SPARSE_MATRIX"


@dataclass(frozen=True)
class SparseMatrix:
    """An immutable sparse matrix in canonical CSR storage.

    Construct via :meth:`from_triplets`, :meth:`from_scipy`,
    :meth:`from_dense`, or :meth:`empty`; the plain constructor performs
    strict validation and is also usable directly if you already have a
    validated ``scipy.sparse.csr_matrix``.

    Attributes:
        n_rows: Number of rows.
        n_cols: Number of columns.
        data: The underlying ``scipy.sparse.csr_matrix`` (canonical form:
            sorted indices, no explicit stored zeros, dtype ``float64``
            unless a config with a different dtype was used to build it).
    """

    n_rows: int
    n_cols: int
    data: sp.csr_matrix

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.n_rows, int) or self.n_rows < 0:
            errors.append(f"n_rows must be a non-negative int, got {self.n_rows!r}")
        if not isinstance(self.n_cols, int) or self.n_cols < 0:
            errors.append(f"n_cols must be a non-negative int, got {self.n_cols!r}")
        if not sp.issparse(self.data):
            errors.append(f"data must be a scipy.sparse matrix, got {type(self.data).__name__}")
        else:
            if self.data.shape != (self.n_rows, self.n_cols):
                errors.append(
                    f"data.shape {self.data.shape} does not match declared "
                    f"({self.n_rows}, {self.n_cols})"
                )
            if self.data.nnz > 0 and not np.all(np.isfinite(self.data.data)):
                errors.append("data contains NaN or Inf entries; matrix coefficients must be finite")
        if errors:
            raise SparseMatrixError(
                "Invalid SparseMatrix: " + "; ".join(errors),
                details={"errors": errors},
            )

    # -- constructors --------------------------------------------------

    @classmethod
    def empty(cls, n_rows: int, n_cols: int, dtype: Any = DEFAULT_DTYPE) -> "SparseMatrix":
        """Return an all-zero ``n_rows`` x ``n_cols`` sparse matrix."""
        return cls(n_rows, n_cols, sp.csr_matrix((n_rows, n_cols), dtype=dtype))

    @classmethod
    def from_triplets(
        cls,
        n_rows: int,
        n_cols: int,
        rows: Sequence[int],
        cols: Sequence[int],
        values: Sequence[Number],
        dtype: Any = DEFAULT_DTYPE,
    ) -> "SparseMatrix":
        """Build from COO triplets ``(rows[k], cols[k], values[k])``.

        Duplicate ``(row, col)`` pairs are summed, matching standard COO
        semantics. Raises :class:`SparseMatrixError` if index arrays are
        out of range, mismatched in length, or values are non-finite.
        """
        rows_arr = np.asarray(rows)
        cols_arr = np.asarray(cols)
        vals_arr = np.asarray(values, dtype=dtype)

        errors: list[str] = []
        if not (len(rows_arr) == len(cols_arr) == len(vals_arr)):
            errors.append(
                f"rows/cols/values length mismatch: {len(rows_arr)}, {len(cols_arr)}, {len(vals_arr)}"
            )
        if len(rows_arr) > 0:
            if rows_arr.min(initial=0) < 0 or (len(rows_arr) and rows_arr.max(initial=-1) >= n_rows):
                errors.append(f"row index out of bounds for n_rows={n_rows}")
            if cols_arr.min(initial=0) < 0 or (len(cols_arr) and cols_arr.max(initial=-1) >= n_cols):
                errors.append(f"col index out of bounds for n_cols={n_cols}")
            if len(vals_arr) and not np.all(np.isfinite(vals_arr)):
                errors.append("values contains NaN or Inf; matrix coefficients must be finite")
        if errors:
            raise SparseMatrixError(
                "Invalid triplet data for SparseMatrix: " + "; ".join(errors),
                details={"errors": errors},
            )

        coo = sp.coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_rows, n_cols), dtype=dtype)
        csr = coo.tocsr()
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        return cls(n_rows, n_cols, csr)

    @classmethod
    def from_scipy(cls, matrix: "sp.spmatrix", dtype: Any = DEFAULT_DTYPE) -> "SparseMatrix":
        """Wrap an existing scipy sparse matrix, converting to canonical CSR."""
        if not sp.issparse(matrix):
            raise SparseMatrixError(
                f"from_scipy requires a scipy.sparse matrix, got {type(matrix).__name__}"
            )
        csr = matrix.tocsr().astype(dtype, copy=True)
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        n_rows, n_cols = csr.shape
        return cls(n_rows, n_cols, csr)

    @classmethod
    def from_dense(cls, array: "np.ndarray | Sequence[Sequence[Number]]", dtype: Any = DEFAULT_DTYPE) -> "SparseMatrix":
        """Build from a dense 2D array-like. Intended for small/test matrices."""
        dense = np.asarray(array, dtype=dtype)
        if dense.ndim != 2:
            raise SparseMatrixError(f"from_dense requires a 2D array, got shape {dense.shape}")
        if dense.size and not np.all(np.isfinite(dense)):
            raise SparseMatrixError("dense array contains NaN or Inf; matrix coefficients must be finite")
        return cls.from_scipy(sp.csr_matrix(dense, dtype=dtype), dtype=dtype)

    # -- properties -------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)

    @property
    def nnz(self) -> int:
        return int(self.data.nnz)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def density(self) -> float:
        total = self.n_rows * self.n_cols
        return (self.nnz / total) if total else 0.0

    # -- conversions (always explicit) ------------------------------------

    def to_dense(self) -> np.ndarray:
        """Materialize as a dense ``numpy.ndarray``. Explicit and potentially
        expensive; never called implicitly elsewhere in this package.
        """
        return np.asarray(self.data.todense())

    def to_csr(self) -> sp.csr_matrix:
        return self.data.copy()

    def to_csc(self) -> sp.csc_matrix:
        return self.data.tocsc()

    def to_coo_triplets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic ``(rows, cols, values)`` triplets sorted by
        ``(row, col)``, suitable for serialization.
        """
        coo = self.data.tocoo()
        order = np.lexsort((coo.col, coo.row))
        return coo.row[order].copy(), coo.col[order].copy(), coo.data[order].copy()

    # -- structural ops -----------------------------------------------

    def transpose(self) -> "SparseMatrix":
        return SparseMatrix.from_scipy(self.data.transpose(), dtype=self.dtype)

    def row(self, index: int) -> "SparseMatrix":
        """Return row ``index`` as a ``1 x n_cols`` SparseMatrix."""
        if not (0 <= index < self.n_rows):
            raise SparseMatrixError(f"row index {index} out of range [0, {self.n_rows})")
        return SparseMatrix.from_scipy(self.data[index : index + 1, :], dtype=self.dtype)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute ``A @ x``. ``x`` must have length ``n_cols``."""
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.shape != (self.n_cols,):
            raise SparseMatrixError(
                f"matvec dimension mismatch: matrix has {self.n_cols} columns, "
                f"x has shape {x_arr.shape}"
            )
        return np.asarray(self.data @ x_arr).reshape(-1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseMatrix):
            return NotImplemented
        if self.shape != other.shape:
            return False
        diff = (self.data - other.data)
        return diff.nnz == 0

    def to_dict(self) -> dict[str, Any]:
        """Deterministic JSON-serializable representation (COO triplets)."""
        rows, cols, values = self.to_coo_triplets()
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "dtype": str(self.dtype),
            "rows": rows.tolist(),
            "cols": cols.tolist(),
            "values": values.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SparseMatrix":
        dtype = np.dtype(payload.get("dtype", DEFAULT_DTYPE)).type
        return cls.from_triplets(
            n_rows=int(payload["n_rows"]),
            n_cols=int(payload["n_cols"]),
            rows=payload["rows"],
            cols=payload["cols"],
            values=payload["values"],
            dtype=dtype,
        )


def vstack(matrices: Iterable[SparseMatrix], dtype: Any = DEFAULT_DTYPE) -> SparseMatrix:
    """Stack sparse matrices row-wise. All must share the same ``n_cols``."""
    mats = list(matrices)
    if not mats:
        raise SparseMatrixError("vstack requires at least one matrix")
    n_cols = mats[0].n_cols
    for m in mats:
        if m.n_cols != n_cols:
            raise SparseMatrixError(
                f"vstack requires matching n_cols; got {n_cols} and {m.n_cols}"
            )
    stacked = sp.vstack([m.to_csr() for m in mats], format="csr", dtype=dtype)
    return SparseMatrix.from_scipy(stacked, dtype=dtype)


__all__ = ["SparseMatrix", "SparseMatrixError", "vstack"]
