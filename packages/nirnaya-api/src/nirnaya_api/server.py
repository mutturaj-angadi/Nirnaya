"""
The REST API.

Endpoints
---------
  GET  /health              liveness / dependency status
  GET  /version              api + dependency versions
  GET  /devices               device/backend discovery
  POST /validate              schema + size validation only, never solves
  POST /solve                  synchronous solve
  POST /jobs                    submit an asynchronous solve
  GET  /jobs/{job_id}            poll an asynchronous solve

All request bodies are validated against strict pydantic schemas
(:mod:`nirnaya_api.schema`) before anything else happens. All errors -
validation failures, oversized payloads, resource-limit breaches, timeouts,
missing dependencies, and any unexpected internal exception - are rendered
through the same structured JSON error envelope
(:meth:`nirnaya_api.errors.NirnayaAPIError.to_dict`). No endpoint executes
shell commands, evaluates user-supplied code, or reads/writes files outside
of the isolated temp-file helper in :mod:`nirnaya_api.security`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import _integration as integ
from .backends import list_devices
from .errors import NirnayaAPIError, unexpected_error
from .jobs import default_job_manager
from .options import SolveOptions
from .orchestrator import solve_model, validate_model
from .schema import (
    DeviceSchema,
    HealthResponseSchema,
    JobStatusResponseSchema,
    JobSubmitResponseSchema,
    SolveRequestSchema,
    SolveResponseSchema,
    ValidateResponseSchema,
    VersionResponseSchema,
)
from .security import DEFAULT_LIMITS, SecurityLimits, check_payload_size
from .version import API_SCHEMA_VERSION, JSON_MODEL_FORMAT_VERSION, __version__

app = FastAPI(
    title="Nirnaya API",
    version=__version__,
    description="Application/service layer around the Nirnaya solver stack.",
)

# Browser access is opt-in and origin-specific. The canonical local launcher
# supplies its exact UI origin; deployments can set the same comma-separated
# environment variable to their trusted dashboard origins.
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("NIRNAYA_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

_limits: SecurityLimits = DEFAULT_LIMITS
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


@app.middleware("http")
async def enforce_payload_limit(request: Request, call_next):
    """Reject oversized bodies before they reach any route handler / JSON
    parser, for every POST/PUT request."""
    # Preserve the supplied dashboard's documented /api prefix while keeping
    # the original unprefixed routes available to Python/CLI users.
    if request.scope.get("path", "").startswith("/api/"):
        request.scope["path"] = request.scope["path"][4:]
        request.scope["raw_path"] = request.scope["path"].encode("ascii", "ignore")
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        try:
            check_payload_size(body, _limits)
        except NirnayaAPIError as exc:
            return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)
    return await call_next(request)


@app.exception_handler(NirnayaAPIError)
async def nirnaya_error_handler(request: Request, exc: NirnayaAPIError):
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    wrapped = unexpected_error(exc)
    return JSONResponse(status_code=wrapped.http_status, content=wrapped.to_dict())


# --------------------------------------------------------------------------
# Health / version / devices
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponseSchema)
def health() -> HealthResponseSchema:
    checks = {}
    for pkg in ("nirnaya_core", "nirnaya_presolve", "nirnaya_solver", "nirnaya_gpu"):
        try:
            __import__(pkg)
            checks[pkg] = True
        except ImportError:
            checks[pkg] = False
    # nirnaya-api is "ok" as long as it can respond at all; missing
    # upstream dependencies make it "degraded" (still serves /validate,
    # /health, /version, /devices, but /solve will fail with a clear
    # DEPENDENCY_UNAVAILABLE error).
    status = "ok" if all(checks.values()) else "degraded"
    return HealthResponseSchema(status=status, checks=checks)


@app.get("/version", response_model=VersionResponseSchema)
def version() -> VersionResponseSchema:
    deps = {}
    for pkg in ("nirnaya_core", "nirnaya_presolve", "nirnaya_solver", "nirnaya_gpu"):
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = None
    return VersionResponseSchema(
        api_version=__version__,
        json_model_format_version=JSON_MODEL_FORMAT_VERSION,
        api_schema_version=API_SCHEMA_VERSION,
        dependencies=deps,
    )


@app.get("/devices", response_model=list[DeviceSchema])
def devices() -> list[DeviceSchema]:
    return [DeviceSchema(**d.__dict__) for d in list_devices()]


@app.get("/system/info")
def system_info() -> dict:
    """Hardware and package information; solver GPU execution is reported separately."""
    import os
    import platform
    try:
        import nirnaya_gpu
        report = nirnaya_gpu.discover_all()
        gpu = next((d for d in report.devices if d.is_gpu), None)
        gpu_info = {"available": gpu is not None,
            "device_name": gpu.model_name if gpu else None,
            "reason": None if gpu else (report.cuda_probe_error or "No CUDA device detected")}
    except Exception as exc:
        gpu_info = {"available": False, "device_name": None, "reason": type(exc).__name__}
    versions = {}
    for pkg in ("nirnaya_core", "nirnaya_solver", "nirnaya_gpu"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
    return {"cpu": {"model": platform.processor() or platform.machine(),
            "physical_cores": None, "logical_cores": os.cpu_count(), "available": True},
            "gpu": gpu_info, "solver_backend": "cpu", "backend_versions": versions}


# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------
@app.post("/validate", response_model=ValidateResponseSchema)
@app.post("/model/validate", response_model=ValidateResponseSchema)
def validate(request: dict) -> ValidateResponseSchema:
    """Accepts a raw model dict (the same shape as ``model`` in
    :class:`SolveRequestSchema`) and reports whether it is well-formed,
    without ever invoking a solver."""
    report = validate_model(request.get("model", request), _limits)
    return ValidateResponseSchema(**report)


@app.get("/benchmarks")
def benchmarks() -> dict:
    """Expose only the repository's local, explicitly recorded benchmark run."""
    path = _WORKSPACE_ROOT / "benchmarks/results/reproducible/latest.json"
    if not path.is_file():
        return {"available": False, "reason": "No reproducible benchmark matrix has been recorded in this workspace.", "runs": []}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "reason": "The recorded benchmark report could not be read safely.", "runs": []}
    runs = []
    for item in report.get("runs", []):
        dims = item.get("dimensions", {})
        runs.append({"problem": item.get("dataset_id", item.get("model", "unknown")),
            "num_variables": dims.get("original_columns"), "num_constraints": dims.get("original_rows"),
            "nonzeros": dims.get("original_nonzeros"), "backend_requested": item.get("backend", "cpu"),
            "status": item.get("nirnaya_status"), "objective": item.get("objective"),
            "iterations": item.get("iterations"),
            "solve_time_ms": (item["solver_seconds"] * 1000.0
                if item.get("solver_seconds") is not None else None),
            "presolve_time_ms": (item["presolve_seconds"] * 1000.0
                if item.get("presolve_seconds") is not None else None),
            "max_residual": item.get("nirnaya_primal_max_violation"),
            "reference_status": item.get("reference_status"),
            "objective_abs_difference": item.get("objective_abs_difference")})
    return {"available": True, "run_id": report.get("run_id"),
        "generated_at_utc": report.get("generated_at_utc"),
        "summary": report.get("summary"), "methodology": report.get("methodology"), "runs": runs}


@app.get("/validation")
def validation_report() -> dict:
    """Serve a compact view of the latest executed independent validation."""
    report_path = _WORKSPACE_ROOT / "apps/nirnaya-ui/validation/validation_report.json"
    meta_path = _WORKSPACE_ROOT / "apps/nirnaya-ui/validation/validation_run_metadata.json"
    try:
        records = json.loads(report_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "reason": "No validation run metadata and report are available in this workspace.", "runs": []}
    compact = [{"case": row.get("case"), "backend": row.get("backend"),
        "reference_status": row.get("reference_status"), "nirnaya_status": row.get("api_status"),
        "reference_objective": row.get("objective_reference"), "nirnaya_objective": row.get("objective_api"),
        "objective_abs_difference": row.get("objective_diff"),
        "max_violation": row.get("max_residual_api"), "solve_time_ms": (row.get("api") or {}).get("solve_time_ms"),
        "iterations": (row.get("api") or {}).get("iterations"), "passed": row.get("passed"),
        "reason": row.get("reason")} for row in records]
    return {"available": True, **metadata, "runs": compact}


# --------------------------------------------------------------------------
# Synchronous solve
# --------------------------------------------------------------------------
@app.post("/solve", response_model=SolveResponseSchema)
def solve(body: SolveRequestSchema) -> SolveResponseSchema:
    options = SolveOptions(**body.options.model_dump())
    if body.backend is not None:
        options.backend = body.backend
    result = solve_model(body.model, options, _limits)
    return SolveResponseSchema(**result.to_dict())


# --------------------------------------------------------------------------
# Asynchronous jobs (optional per spec)
# --------------------------------------------------------------------------
@app.post("/jobs", response_model=JobSubmitResponseSchema, status_code=202)
def submit_job(body: SolveRequestSchema) -> JobSubmitResponseSchema:
    options = SolveOptions(**body.options.model_dump())
    if body.backend is not None:
        options.backend = body.backend
    model_plain = body.model

    def _run():
        return solve_model(model_plain, options, _limits)

    job = default_job_manager.submit(_run)
    return JobSubmitResponseSchema(job_id=job.id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponseSchema)
def get_job(job_id: str) -> JobStatusResponseSchema:
    job = default_job_manager.get(job_id)
    return JobStatusResponseSchema(
        job_id=job.id,
        status=job.status,
        result=SolveResponseSchema(**job.result.to_dict()) if job.result else None,
        error=job.error,
    )
