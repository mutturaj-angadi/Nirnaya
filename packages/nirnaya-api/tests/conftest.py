import pytest

from nirnaya_api import _integration as integ
from nirnaya_api.testing import FakeSolverBackend


@pytest.fixture()
def fake_backend():
    """Install a fresh FakeSolverBackend (no GPU devices) for a test and
    restore the real backend afterwards."""
    backend = FakeSolverBackend(gpu_devices=[])
    integ.set_backend(backend)
    yield backend
    integ.reset_backend()


@pytest.fixture()
def fake_backend_with_gpu():
    backend = FakeSolverBackend(
        gpu_devices=[{"id": "gpu:0", "name": "Fake GPU 0", "available": True, "detail": {}}]
    )
    integ.set_backend(backend)
    yield backend
    integ.reset_backend()


@pytest.fixture()
def simple_model_dict():
    # Bounds are deliberately small enough that pushing every variable to
    # its most-favorable bound (what FakeSolverBackend's trivial per-
    # variable solve does) still satisfies the shared constraint, so this
    # fixture is solvable by the fake backend used throughout these tests.
    return {
        "name": "toy",
        "variables": {
            "x": {"lb": 0.0, "ub": 5.0, "kind": "continuous"},
            "y": {"lb": 0.0, "ub": 5.0, "kind": "continuous"},
        },
        "constraints": {
            "c1": {"coefficients": {"x": 1.0, "y": 1.0}, "sense": "<=", "rhs": 12.0}
        },
        "objective": {"coefficients": {"x": 1.0, "y": 2.0}, "sense": "max", "offset": 0.0},
    }
