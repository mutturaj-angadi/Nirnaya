"""Correctness tests for the CPU backend against dense-numpy reference
implementations. These run unconditionally (CPU is always available)."""

import numpy as np
import pytest

from nirnaya_gpu import Session, SparseCSR
from nirnaya_gpu.registry import get_backend_by_name


@pytest.fixture
def session():
    return Session(backend=get_backend_by_name("cpu"))


def _random_problem(n_rows=64, n_cols=64, density=0.12, seed=0):
    rng = np.random.default_rng(seed)
    A = SparseCSR.random(n_rows, n_cols, density=density, seed=seed)
    x = rng.uniform(-2, 2, size=n_cols)
    b = rng.uniform(-2, 2, size=n_rows)
    return A, x, b


def test_spmv_matches_dense(session):
    A, x, _ = _random_problem()
    y = session.spmv(A, x)
    y_ref = A.to_dense() @ x
    assert np.allclose(y, y_ref, atol=1e-10)


def test_spmv_zero_matrix(session):
    A = SparseCSR(
        indptr=np.zeros(6, dtype=np.int64),
        indices=np.array([], dtype=np.int64),
        data=np.array([], dtype=np.float64),
        shape=(5, 5),
    )
    x = np.arange(5, dtype=np.float64)
    y = session.spmv(A, x)
    assert np.allclose(y, np.zeros(5))


def test_spmv_empty_rows_interspersed(session):
    # rows: [nonzero, empty, nonzero, empty, empty]
    dense = np.zeros((5, 4))
    dense[0] = [1.0, 0, 0, 2.0]
    dense[2] = [0, 3.0, 0, 0]
    A = SparseCSR.from_dense(dense)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = session.spmv(A, x)
    assert np.allclose(y, dense @ x)


def test_dot(session):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 30)
    y = rng.uniform(-1, 1, 30)
    assert np.isclose(session.dot(x, y), np.dot(x, y))


def test_axpy(session):
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 20)
    y = rng.uniform(-1, 1, 20)
    alpha = 3.5
    out = session.axpy(alpha, x, y)
    assert np.allclose(out, alpha * x + y)


def test_scale(session):
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 20)
    out = session.scale(-2.0, x)
    assert np.allclose(out, -2.0 * x)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "max", "min"])
def test_elementwise(session, op):
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, 25)
    y = rng.uniform(-1, 1, 25)
    out = session.elementwise(op, x, y)
    ref = {
        "add": x + y,
        "sub": x - y,
        "mul": x * y,
        "max": np.maximum(x, y),
        "min": np.minimum(x, y),
    }[op]
    assert np.allclose(out, ref)


@pytest.mark.parametrize("op", ["sum", "max", "min", "norm2", "norm_inf"])
def test_reduce(session, op):
    rng = np.random.default_rng(5)
    x = rng.uniform(-3, 3, 40)
    out = session.reduce(x, op)
    ref = {
        "sum": np.sum(x),
        "max": np.max(x),
        "min": np.min(x),
        "norm2": np.linalg.norm(x),
        "norm_inf": np.max(np.abs(x)),
    }[op]
    assert np.isclose(out, ref)


def test_residual(session):
    A, x, b = _random_problem()
    r = session.residual(A, x, b)
    assert np.allclose(r, b - A.to_dense() @ x)


@pytest.mark.parametrize("sense", ["le", "ge", "eq"])
def test_feasibility_check_matches_manual(session, sense):
    A, x, b = _random_problem(n_rows=30, n_cols=30, seed=9)
    tol = 1e-6
    ok, viol = session.feasibility(A, x, b, tol=tol, sense=sense)
    ax = A.to_dense() @ x
    if sense == "le":
        manual_viol = np.maximum(ax - b - tol, 0.0)
    elif sense == "ge":
        manual_viol = np.maximum(b - ax - tol, 0.0)
    else:
        manual_viol = np.abs(ax - b)
        manual_viol = np.where(manual_viol <= tol, 0.0, manual_viol)
    assert np.allclose(viol, manual_viol, atol=1e-9)
    assert ok == bool(np.all(manual_viol <= tol if sense != "eq" else manual_viol == 0.0))


def test_feasibility_check_trivially_satisfied(session):
    A, x, _ = _random_problem(seed=11)
    b_big = (A.to_dense() @ x) + 1000.0  # Ax <= b trivially true
    ok, viol = session.feasibility(A, x, b_big, tol=1e-6, sense="le")
    assert ok is True
    assert np.allclose(viol, 0.0)


def test_argmin(session):
    rng = np.random.default_rng(6)
    x = rng.uniform(-5, 5, 50)
    idx, val = session.argmin(x)
    assert idx == int(np.argmin(x))
    assert np.isclose(val, x[idx])


def test_ratio_test_basic(session):
    x = np.array([4.0, 6.0, 10.0, 2.0])
    d = np.array([2.0, 3.0, -1.0, 1.0])  # only indices 0,1,3 have d>tol
    idx, ratio = session.ratio_test(x, d)
    # ratios: 4/2=2, 6/3=2, (skip), 2/1=2 -> tie; argmin picks first = index 0
    assert idx in (0, 1, 3)
    assert np.isclose(ratio, 2.0)


def test_ratio_test_unbounded(session):
    x = np.array([1.0, 2.0, 3.0])
    d = np.array([-1.0, -2.0, 0.0])  # no positive direction component
    idx, ratio = session.ratio_test(x, d)
    assert idx is None
    assert ratio == float("inf")


def test_session_reports_cpu_honestly(session):
    assert session.backend.name == "cpu"
    assert session.is_gpu is False
    assert session.device.is_gpu is False
