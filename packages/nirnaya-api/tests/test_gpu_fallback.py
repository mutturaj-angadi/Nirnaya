"""
Backend/device selection tests.

These exercise :mod:`nirnaya_api.backends` directly against the injected
fake backend, covering: no-GPU-available fallback, GPU-available
preference, explicit cpu/gpu requests, pinned device requests, and a
GPU-listing failure being surfaced (not silently swallowed).

A real GPU-hardware test (skipped when no GPU is present) is included at
the bottom, per the "GPU tests when hardware exists" requirement.
"""

from __future__ import annotations

import pytest

from nirnaya_api import _integration as integ
from nirnaya_api.backends import list_devices, select_backend
from nirnaya_api.errors import DeviceUnavailableError
from nirnaya_api.testing import FakeSolverBackend


def test_no_gpu_auto_falls_back_to_cpu(fake_backend):
    backend, device, warnings = select_backend("auto")
    assert backend == "cpu"
    assert device == "cpu:0"
    assert warnings  # a fallback warning must be surfaced, not hidden


def test_gpu_available_auto_prefers_gpu(fake_backend_with_gpu):
    backend, device, warnings = select_backend("auto")
    assert backend == "gpu"
    assert device == "gpu:0"
    assert warnings == []


def test_explicit_cpu_never_touches_gpu_listing():
    calls = {"count": 0}

    class CountingBackend(FakeSolverBackend):
        def list_gpu_devices(self):
            calls["count"] += 1
            return super().list_gpu_devices()

    integ.set_backend(CountingBackend(gpu_devices=[{"id": "gpu:0", "name": "g", "available": True, "detail": {}}]))
    try:
        backend, device, warnings = select_backend("cpu")
        assert backend == "cpu"
        assert calls["count"] == 0
    finally:
        integ.reset_backend()


def test_explicit_gpu_without_hardware_raises(fake_backend):
    with pytest.raises(DeviceUnavailableError):
        select_backend("gpu")


def test_explicit_gpu_with_hardware_succeeds(fake_backend_with_gpu):
    backend, device, warnings = select_backend("gpu")
    assert backend == "gpu"
    assert device == "gpu:0"


def test_pinned_unavailable_device_raises(fake_backend):
    with pytest.raises(DeviceUnavailableError):
        select_backend("auto", device="gpu:99")


def test_pinned_available_device_honored(fake_backend_with_gpu):
    backend, device, warnings = select_backend("auto", device="gpu:0")
    assert backend == "gpu"
    assert device == "gpu:0"


def test_gpu_listing_failure_is_reported_not_hidden(fake_backend):
    integ.set_backend(FakeSolverBackend(fail_gpu_listing="driver not loaded"))
    try:
        devices = list_devices()
        unavailable = [d for d in devices if d.id == "gpu:unavailable"]
        assert unavailable and "driver not loaded" in unavailable[0].detail["reason"]
        # And auto-selection still gracefully falls back to CPU rather
        # than raising.
        backend, device, warnings = select_backend("auto")
        assert backend == "cpu"
    finally:
        integ.reset_backend()


def test_cpu_device_always_listed():
    devices = list_devices()
    assert any(d.id == "cpu:0" and d.available for d in devices)


@pytest.mark.gpu
def test_real_gpu_hardware_if_present():
    """Only meaningful when nirnaya-gpu is installed and real hardware is
    present; skipped otherwise so CI without GPUs still passes cleanly."""
    try:
        import nirnaya_gpu  # type: ignore
    except ImportError:
        pytest.skip("nirnaya-gpu not installed")

    integ.reset_backend()
    devices = [d for d in list_devices() if d.kind == "gpu" and d.available]
    if not devices:
        pytest.skip("no real GPU device available")

    backend, device, warnings = select_backend("gpu")
    assert backend == "gpu"
