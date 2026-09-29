"""Floating-point / structural edge cases for the CPU backend (always
runs). Mirrors the cases docs/ARCHITECTURE.md flags as precision-sensitive."""

import numpy as np
import pytest

from nirnaya_gpu import Session, SparseCSR
from nirnaya_gpu.exceptions import ShapeMismatchError


@pytest.fixture
def session():
    return Session(prefer_gpu=False)


def test_all_zero_vector_reduce(session):
    x = np.zeros(10)
    assert session.reduce(x, "sum") == 0.0
    assert session.reduce(x, "norm2") == 0.0
    assert session.reduce(x, "norm_inf") == 0.0


def test_single_element_vector(session):
    x = np.array([3.14])
    assert session.reduce(x, "sum") == pytest.approx(3.14)
    idx, val = session.argmin(x)
    assert idx == 0


def test_nan_propagation(session):
    x = np.array([1.0, np.nan, 3.0])
    y = np.array([1.0, 1.0, 1.0])
    out = session.elementwise("add", x, y)
    assert np.isnan(out[1])
    assert not np.isnan(out[0]) and not np.isnan(out[2])


def test_inf_in_ratio_test(session):
    x = np.array([1.0, 2.0])
    d = np.array([1e-15, 1.0])  # first direction below default tol
    idx, ratio = session.ratio_test(x, d, tol=1e-9)
    assert idx == 1
    assert np.isclose(ratio, 2.0)


def test_very_small_and_large_magnitudes(session):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(1e-12, 1e-10, 10), rng.uniform(1e10, 1e12, 10)])
    A = SparseCSR.from_dense(np.eye(20))
    y = session.spmv(A, x)
    assert np.allclose(y, x, rtol=1e-9)


def test_tolerance_boundary_feasibility(session):
    # Ax - b sits *exactly* at tol: should be reported as satisfied
    # (implementation uses `> tol` as violation, not `>= tol`).
    A = SparseCSR.from_dense(np.eye(3))
    x = np.array([1.0, 2.0, 3.0])
    tol = 1e-3
    b = x - tol  # Ax - b == tol exactly for every row
    ok, viol = session.feasibility(A, x, b, tol=tol, sense="le")
    assert ok is True
    assert np.allclose(viol, 0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ShapeMismatchError):
        SparseCSR(
            indptr=np.array([0, 1, 2]),
            indices=np.array([0, 1, 2]),  # length mismatch vs data
            data=np.array([1.0, 2.0]),
            shape=(2, 3),
        )


def test_out_of_bounds_column_index_raises():
    with pytest.raises(ShapeMismatchError):
        SparseCSR(
            indptr=np.array([0, 1]),
            indices=np.array([5]),  # out of bounds for 3 columns
            data=np.array([1.0]),
            shape=(1, 3),
        )


def test_empty_matrix(session):
    A = SparseCSR(
        indptr=np.array([0, 0, 0], dtype=np.int64),
        indices=np.array([], dtype=np.int64),
        data=np.array([], dtype=np.float64),
        shape=(2, 2),
    )
    x = np.array([1.0, 2.0])
    y = session.spmv(A, x)
    assert np.array_equal(y, np.zeros(2))


def test_dense_roundtrip_random(session):
    rng = np.random.default_rng(42)
    dense = rng.uniform(-1, 1, size=(15, 12))
    dense[rng.uniform(size=dense.shape) < 0.7] = 0.0  # sparsify
    A = SparseCSR.from_dense(dense)
    assert np.allclose(A.to_dense(), dense)
    x = rng.uniform(-1, 1, size=12)
    assert np.allclose(session.spmv(A, x), dense @ x, atol=1e-10)
