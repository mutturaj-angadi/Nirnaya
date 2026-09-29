"""Unit tests for nirnaya_core.matrix.SparseMatrix."""

import math

import numpy as np
import pytest
import scipy.sparse as sp

from nirnaya_core.matrix.sparse import SparseMatrix, SparseMatrixError, vstack


class TestConstruction:
    def test_empty(self):
        m = SparseMatrix.empty(3, 4)
        assert m.shape == (3, 4)
        assert m.nnz == 0
        assert m.density == 0.0

    def test_from_triplets(self):
        m = SparseMatrix.from_triplets(2, 2, [0, 1], [0, 1], [1.0, 2.0])
        assert m.shape == (2, 2)
        assert m.nnz == 2
        dense = m.to_dense()
        np.testing.assert_allclose(dense, [[1.0, 0.0], [0.0, 2.0]])

    def test_from_triplets_duplicate_entries_summed(self):
        m = SparseMatrix.from_triplets(1, 1, [0, 0], [0, 0], [1.0, 2.0])
        assert m.to_dense()[0, 0] == pytest.approx(3.0)

    def test_from_triplets_out_of_bounds_row(self):
        with pytest.raises(SparseMatrixError):
            SparseMatrix.from_triplets(2, 2, [5], [0], [1.0])

    def test_from_triplets_out_of_bounds_col(self):
        with pytest.raises(SparseMatrixError):
            SparseMatrix.from_triplets(2, 2, [0], [5], [1.0])

    def test_from_triplets_mismatched_lengths(self):
        with pytest.raises(SparseMatrixError):
            SparseMatrix.from_triplets(2, 2, [0, 1], [0], [1.0, 2.0])

    def test_from_triplets_nan_rejected(self):
        with pytest.raises(SparseMatrixError):
            SparseMatrix.from_triplets(1, 1, [0], [0], [math.nan])

    def test_from_triplets_inf_rejected(self):
        with pytest.raises(SparseMatrixError):
            SparseMatrix.from_triplets(1, 1, [0], [0], [math.inf])

    def test_from_dense(self):
        m = SparseMatrix.from_dense([[1.0, 0.0], [0.0, 2.0]])
        assert m.nnz == 2

    def test_from_dense_rejects_non_2d(self):
        with pytest.raises(SparseMatrixError):
            SparseMatrix.from_dense([1.0, 2.0, 3.0])

    def test_from_scipy(self):
        csr = sp.csr_matrix(np.eye(3))
        m = SparseMatrix.from_scipy(csr)
        assert m.shape == (3, 3)
        assert m.nnz == 3

    def test_from_scipy_rejects_dense_input(self):
        with pytest.raises(SparseMatrixError):
            SparseMatrix.from_scipy(np.eye(3))  # type: ignore[arg-type]

    def test_shape_mismatch_rejected_in_direct_construction(self):
        csr = sp.csr_matrix((2, 2))
        with pytest.raises(SparseMatrixError):
            SparseMatrix(n_rows=3, n_cols=2, data=csr)

    def test_nan_in_direct_construction_rejected(self):
        csr = sp.csr_matrix(np.array([[1.0, math.nan]]))
        with pytest.raises(SparseMatrixError):
            SparseMatrix(n_rows=1, n_cols=2, data=csr)


class TestOperations:
    def test_matvec(self):
        m = SparseMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
        result = m.matvec(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_matvec_dimension_mismatch(self):
        m = SparseMatrix.from_dense([[1.0, 2.0]])
        with pytest.raises(SparseMatrixError):
            m.matvec(np.array([1.0, 2.0, 3.0]))

    def test_transpose(self):
        m = SparseMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
        mt = m.transpose()
        np.testing.assert_allclose(mt.to_dense(), [[1.0, 3.0], [2.0, 4.0]])

    def test_row(self):
        m = SparseMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
        r = m.row(1)
        assert r.shape == (1, 2)
        np.testing.assert_allclose(r.to_dense(), [[3.0, 4.0]])

    def test_row_out_of_range(self):
        m = SparseMatrix.from_dense([[1.0]])
        with pytest.raises(SparseMatrixError):
            m.row(5)

    def test_equality(self):
        a = SparseMatrix.from_dense([[1.0, 0.0], [0.0, 2.0]])
        b = SparseMatrix.from_triplets(2, 2, [0, 1], [0, 1], [1.0, 2.0])
        assert a == b

    def test_inequality_different_values(self):
        a = SparseMatrix.from_dense([[1.0]])
        b = SparseMatrix.from_dense([[2.0]])
        assert a != b

    def test_vstack(self):
        a = SparseMatrix.from_dense([[1.0, 2.0]])
        b = SparseMatrix.from_dense([[3.0, 4.0]])
        stacked = vstack([a, b])
        np.testing.assert_allclose(stacked.to_dense(), [[1.0, 2.0], [3.0, 4.0]])

    def test_vstack_mismatched_cols(self):
        a = SparseMatrix.from_dense([[1.0, 2.0]])
        b = SparseMatrix.from_dense([[3.0, 4.0, 5.0]])
        with pytest.raises(SparseMatrixError):
            vstack([a, b])

    def test_vstack_empty_raises(self):
        with pytest.raises(SparseMatrixError):
            vstack([])


class TestSerialization:
    def test_round_trip(self):
        m = SparseMatrix.from_triplets(3, 3, [0, 1, 2], [2, 1, 0], [1.5, -2.5, 3.5])
        d = m.to_dict()
        m2 = SparseMatrix.from_dict(d)
        assert m == m2

    def test_to_dict_is_deterministic_sorted_by_row_col(self):
        m = SparseMatrix.from_triplets(2, 2, [1, 0, 1], [1, 0, 0], [1.0, 2.0, 3.0])
        d = m.to_dict()
        pairs = list(zip(d["rows"], d["cols"]))
        assert pairs == sorted(pairs)

    def test_empty_matrix_round_trip(self):
        m = SparseMatrix.empty(2, 3)
        d = m.to_dict()
        m2 = SparseMatrix.from_dict(d)
        assert m == m2
        assert m2.nnz == 0
