import numpy as np
import scipy.sparse as sp

from nirnaya_core import NumericalConfig
from nirnaya_solver.solver import _BasisFactorization


def test_product_form_basis_updates_match_sparse_lu_ftran_and_btran():
    A = sp.csc_matrix(np.array([
        [4.0, 1.0, 0.0, 1.0, 2.0, 1.0],
        [1.0, 3.0, 1.0, 2.0, 1.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 5.0, 2.0],
    ]))
    basis = np.array([0, 1, 2], dtype=np.intp)
    cfg = NumericalConfig()
    metrics = {"factorization_seconds": 0.0, "factorization_count": 0,
        "factor_solve_seconds": 0.0, "basis_update_seconds": 0.0,
        "basis_update_count": 0, "basis_residual_check_seconds": 0.0}
    factor = _BasisFactorization(A, basis, cfg, metrics)

    for entering in (3, 4):
        direction = factor.solve(A[:, entering].toarray().ravel())
        leave = int(np.argmax(np.abs(direction)))
        basis[leave] = entering
        factor.pivot(leave, direction, basis)

    B = A[:, basis].toarray()
    rhs = np.array([2.0, -1.0, 3.0])
    x = factor.solve(rhs)
    y = factor.solve(rhs, trans="T")
    assert np.allclose(B @ x, rhs, atol=1e-12, rtol=1e-12)
    assert np.allclose(B.T @ y, rhs, atol=1e-12, rtol=1e-12)
    assert metrics["factorization_count"] == 1
    assert metrics["basis_update_count"] == 2


def test_basis_updates_refactor_after_fixed_update_window():
    A = sp.hstack([sp.eye(3, format="csc"), sp.csc_matrix(np.ones((3, 35)))], format="csc")
    basis = np.array([0, 1, 2], dtype=np.intp)
    metrics = {"factorization_seconds": 0.0, "factorization_count": 0,
        "factor_solve_seconds": 0.0, "basis_update_seconds": 0.0,
        "basis_update_count": 0, "basis_residual_check_seconds": 0.0}
    factor = _BasisFactorization(A, basis, NumericalConfig(), metrics)
    for col in range(3, 35):
        direction = factor.solve(A[:, col].toarray().ravel())
        row = int(np.argmax(np.abs(direction)))
        basis[row] = col
        factor.pivot(row, direction, basis)
        assert len(factor.etas) < factor.REFACTOR_INTERVAL
    assert metrics["factorization_count"] > 1
