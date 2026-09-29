"""
Device discovery and backend selection.

This module is the *only* place that talks to ``nirnaya-gpu``. It is
deliberately isolated from solving logic: its job is strictly "which device
should we ask nirnaya-solver to run on", including the graceful fallback to
CPU whenever GPU is unavailable, disabled, or requested-but-missing without
being explicitly required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .errors import DeviceUnavailableError
from . import _integration as integ


@dataclass
class DeviceInfo:
    id: str
    kind: str  # "cpu" | "gpu"
    name: str
    available: bool
    detail: dict


def list_devices() -> List[DeviceInfo]:
    """Return every device nirnaya-api knows how to target.

    Always includes at least one CPU device (nirnaya-api assumes
    nirnaya-solver can always run on CPU). GPU devices are appended only
    if ``nirnaya-gpu`` is importable and reports them.
    """
    devices = [
        DeviceInfo(
            id="cpu:0",
            kind="cpu",
            name="CPU (host)",
            available=True,
            detail={"source": "builtin"},
        )
    ]

    gpu_devices, gpu_error = integ.try_list_gpu_devices()
    for d in gpu_devices:
        devices.append(
            DeviceInfo(
                id=d["id"],
                kind="gpu",
                name=d.get("name", d["id"]),
                available=d.get("available", True),
                detail=d.get("detail", {}),
            )
        )
    if gpu_error is not None:
        devices.append(
            DeviceInfo(
                id="gpu:unavailable",
                kind="gpu",
                name="GPU backend unavailable",
                available=False,
                detail={"reason": gpu_error},
            )
        )
    return devices


def select_backend(
    backend: str = "auto",
    device: Optional[str] = None,
) -> tuple[str, str, list[str]]:
    """Resolve a requested (backend, device) pair to an actual one.

    Returns ``(resolved_backend, resolved_device, warnings)``.

    Rules:
      * ``backend="cpu"``            -> always CPU, never touches nirnaya-gpu.
      * ``backend="gpu"``            -> requires a usable GPU device; raises
                                          :class:`DeviceUnavailableError` if
                                          none exists (explicit request must
                                          not silently downgrade).
      * ``backend="auto"`` (default) -> prefer GPU if available and healthy;
                                          otherwise fall back to CPU with a
                                          warning. This is the "graceful
                                          fallback" path.
      * ``device=<id>``              -> pins a specific device; validated
                                          against the resolved backend.
    """
    warnings: list[str] = []
    if backend == "cpu" and device is None:
        return "cpu", "cpu:0", warnings
    devices = {d.id: d for d in list_devices()}

    if device is not None:
        d = devices.get(device)
        if d is None or not d.available:
            raise DeviceUnavailableError(f"Requested device {device!r} is not available")
        return d.kind, d.id, warnings

    if backend == "cpu":
        return "cpu", "cpu:0", warnings

    if backend == "gpu":
        gpu = next((d for d in devices.values() if d.kind == "gpu" and d.available), None)
        if gpu is None:
            raise DeviceUnavailableError(
                "backend='gpu' was requested explicitly but no usable GPU device "
                "was found. Use backend='auto' to allow automatic CPU fallback."
            )
        return "gpu", gpu.id, warnings

    # backend == "auto"
    gpu = next((d for d in devices.values() if d.kind == "gpu" and d.available), None)
    if gpu is not None:
        return "gpu", gpu.id, warnings

    warnings.append("No usable GPU device found; falling back to CPU backend.")
    return "cpu", "cpu:0", warnings
