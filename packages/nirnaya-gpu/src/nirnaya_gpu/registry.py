"""
Device discovery and backend selection.

This is the single place that decides "what hardware do we actually
have, and what should we use by default." nirnaya-solver is expected
to go through `get_backend()` / `list_devices()` rather than importing
CPUBackend/GPUBackend directly (see docs/INTEGRATION_CONTRACT.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .backends.base import Backend
from .backends.cpu import CPUBackend
from .device import Device, DeviceType
from .exceptions import DeviceUnavailableError


@dataclass
class DiscoveryReport:
    """Real, timestamped record of what was found -- used by the demo
    and by tests to assert nothing is fabricated."""

    devices: List[Device]
    cuda_probe_error: Optional[str]

    def summary(self) -> str:
        lines = [f"Discovered {len(self.devices)} device(s):"]
        for d in self.devices:
            lines.append(f"  - {d.describe()}")
        if self.cuda_probe_error:
            lines.append(f"  (CUDA backend unavailable: {self.cuda_probe_error})")
        return "\n".join(lines)


def discover_all() -> DiscoveryReport:
    """Probe every known backend for real devices. Never raises; a
    missing/broken GPU backend just contributes zero devices plus an
    explanatory message."""
    devices: List[Device] = list(CPUBackend.discover_devices())
    cuda_error: Optional[str] = None
    try:
        from .backends.gpu import GPUBackend, _probe_cupy

        _, count, err = _probe_cupy()
        if count > 0:
            devices.extend(GPUBackend.discover_devices())
        else:
            cuda_error = err
    except ImportError as e:  # pragma: no cover - gpu.py itself missing
        cuda_error = str(e)
    return DiscoveryReport(devices=devices, cuda_probe_error=cuda_error)


def get_backend(prefer_gpu: bool = True) -> Backend:
    """Return a live Backend instance.

    If `prefer_gpu` is True (default), attempts to construct a
    GPUBackend; if CUDA is genuinely unavailable, falls back to
    CPUBackend and this fact is NOT hidden -- callers can check
    `backend.name` and `backend.discover_devices()` to confirm what
    they actually got.
    """
    if prefer_gpu:
        try:
            from .backends.gpu import GPUBackend

            return GPUBackend()
        except DeviceUnavailableError:
            pass
    return CPUBackend()


def get_backend_by_name(name: str) -> Backend:
    """Explicit selection, for callers (or config files) that want to
    force a specific backend rather than auto-detect. Raises
    DeviceUnavailableError if that exact backend cannot be used --
    it will NOT silently substitute a different backend."""
    if name == "cpu":
        return CPUBackend()
    if name == "cuda":
        from .backends.gpu import GPUBackend

        return GPUBackend()
    raise DeviceUnavailableError(f"unknown backend name {name!r}")
