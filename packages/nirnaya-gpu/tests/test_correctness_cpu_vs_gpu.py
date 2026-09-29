"""
CPU vs GPU numerical agreement tests.

These tests are SKIPPED (not faked, not passed vacuously) when no CUDA
device is available -- pytest will report them as `skipped`, and the
skip reason states exactly why, so a test run's summary never claims
GPU coverage it doesn't have. Run on a CUDA-capable machine with
`pip install nirnaya-gpu[cuda]` to actually exercise this file.
"""

import numpy as np
import pytest

from nirnaya_gpu import Session, SparseCSR
from nirnaya_gpu.registry import get_backend_by_name

try:
    from nirnaya_gpu.backends.gpu import GPUBackend

    _CUDA_AVAILABLE = GPUBackend.available()
except ImportError:
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="CUDA backend unavailable on this machine (no cupy / no CUDA device)",
)

# Configurable numerical tolerance for CPU/GPU agreement. GPU reductions
# (sum, dot, norm) use a different summation order than CPU (parallel
# tree reduction vs. sequential/pairwise), so exact bitwise equality is
# not expected -- see docs/ARCHITECTURE.md, "numerical precision".
RTOL = 1e-6
ATOL = 1e-8


@pytest.fixture
def sessions():
    cpu = Session(backend=get_backend_by_name("cpu"))
    gpu = Session(backend=get_backend_by_name("cuda"))
    return cpu, gpu


def _problem(n_rows=200, n_cols=200, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    A = SparseCSR.random(n_rows, n_cols, density=density, seed=seed)
    x = rng.uniform(-3, 3, size=n_cols)
    b = rng.uniform(-3, 3, size=n_rows)
    return A, x, b


def test_spmv_agrees(sessions):
    cpu, gpu = sessions
    A, x, _ = _problem()
    y_cpu = cpu.spmv(A, x)
    y_gpu = gpu.spmv(A, x)
    assert np.allclose(y_cpu, y_gpu, rtol=RTOL, atol=ATOL)


def test_residual_agrees(sessions):
    cpu, gpu = sessions
    A, x, b = _problem(seed=1)
    r_cpu = cpu.residual(A, x, b)
    r_gpu = gpu.residual(A, x, b)
    assert np.allclose(r_cpu, r_gpu, rtol=RTOL, atol=ATOL)


def test_dot_agrees(sessions):
    cpu, gpu = sessions
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 5000)
    y = rng.uniform(-1, 1, 5000)
    assert np.isclose(cpu.dot(x, y), gpu.dot(x, y), rtol=RTOL, atol=ATOL)


def test_reduce_norm2_agrees(sessions):
    cpu, gpu = sessions
    rng = np.random.default_rng(3)
    x = rng.uniform(-10, 10, 10000)
    assert np.isclose(
        cpu.reduce(x, "norm2"), gpu.reduce(x, "norm2"), rtol=RTOL, atol=ATOL
    )


def test_feasibility_check_agrees(sessions):
    cpu, gpu = sessions
    A, x, b = _problem(seed=4)
    ok_cpu, viol_cpu = cpu.feasibility(A, x, b, tol=1e-6, sense="le")
    ok_gpu, viol_gpu = gpu.feasibility(A, x, b, tol=1e-6, sense="le")
    assert ok_cpu == ok_gpu
    assert np.allclose(viol_cpu, viol_gpu, rtol=RTOL, atol=ATOL)


def test_argmin_agrees(sessions):
    cpu, gpu = sessions
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, 1000)
    idx_cpu, val_cpu = cpu.argmin(x)
    idx_gpu, val_gpu = gpu.argmin(x)
    assert idx_cpu == idx_gpu
    assert np.isclose(val_cpu, val_gpu, rtol=RTOL, atol=ATOL)


def test_ratio_test_agrees(sessions):
    cpu, gpu = sessions
    rng = np.random.default_rng(6)
    x = np.abs(rng.uniform(0, 5, 500))
    d = rng.uniform(-2, 2, 500)
    idx_cpu, r_cpu = cpu.ratio_test(x, d)
    idx_gpu, r_gpu = gpu.ratio_test(x, d)
    assert idx_cpu == idx_gpu
    assert np.isclose(r_cpu, r_gpu, rtol=RTOL, atol=ATOL)
