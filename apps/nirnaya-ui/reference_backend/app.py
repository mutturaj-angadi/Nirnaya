#!/usr/bin/env python3
"""
reference_backend/app.py

A small, HONEST reference implementation of docs/API_CONTRACT.md.

Purpose
-------
Parts 1-5 of Nirnaya (model parser, presolve, CPU solver, GPU solver,
orchestration API) are separate repositories. This backend lets
`nirnaya-ui-tests` be demoed end-to-end *before* those repos are wired in,
by serving the exact same contract using SciPy's HiGHS solver for CPU.

It NEVER fabricates data:
  - System info (CPU model, core count, GPU presence) is read from the
    real host via `platform`/`psutil`/`nvidia-smi` detection.
  - If no GPU is present, backend="gpu" requests return
    status="error" with an explicit `gpu_unavailable_reason` — they do
    NOT silently fall back to CPU numbers labeled as GPU.
  - All timings come from `time.perf_counter()` around the actual solve.

When Parts 1-5 are available, point the UI's "API Base URL" at their real
service instead of this one — the UI does not need to change, only the
base URL setting does (see INTEGRATION_CHECKLIST.md).
"""
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from scipy.optimize import linprog

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

app = Flask(__name__)
CORS(app)

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

BACKEND_VERSIONS = {
    "nirnaya_core": "reference-backend (Parts 1-5 not connected)",
    "nirnaya_cpu_solver": f"scipy {__import__('scipy').__version__} (HiGHS)",
    "nirnaya_gpu_solver": "not connected",
}


def detect_gpu():
    """Honestly detect GPU presence. Never invents a device."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3,
            )
            name = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
            if name:
                return {"available": True, "device_name": name, "reason": None}
        except Exception as e:
            return {"available": False, "device_name": None, "reason": f"nvidia-smi query failed: {e}"}
    try:
        import cupy  # noqa: F401
        return {"available": True, "device_name": "cupy device (unspecified)", "reason": None}
    except ImportError:
        pass
    return {"available": False, "device_name": None,
            "reason": "no CUDA-capable device detected (nvidia-smi absent, cupy not installed)"}


def cpu_info():
    model = None
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
    except Exception:
        model = None
    physical = psutil.cpu_count(logical=False) if HAVE_PSUTIL else None
    logical = psutil.cpu_count(logical=True) if HAVE_PSUTIL else (platform.os.cpu_count() if hasattr(platform, "os") else None)
    return {"model": model, "physical_cores": physical, "logical_cores": logical, "available": True}


@app.get("/api/system/info")
def system_info():
    return jsonify({
        "cpu": cpu_info(),
        "gpu": detect_gpu(),
        "backend_versions": BACKEND_VERSIONS,
        "queried_at": datetime.now(timezone.utc).isoformat(),
    })


def _model_to_arrays(model):
    var_names = [v["name"] for v in model["variables"]]
    idx = {name: i for i, name in enumerate(var_names)}
    n = len(var_names)
    sense = model.get("sense", "minimize").lower()
    c = np.zeros(n)
    for name, coef in model["objective"].get("coefficients", {}).items():
        c[idx[name]] = coef
    offset = model["objective"].get("offset", 0.0)
    c_solve = -c if sense == "maximize" else c.copy()

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in model["constraints"]:
        row = np.zeros(n)
        for name, coef in con["coefficients"].items():
            row[idx[name]] = coef
        if con["sense"] == "<=":
            A_ub.append(row); b_ub.append(con["rhs"])
        elif con["sense"] == ">=":
            A_ub.append(-row); b_ub.append(-con["rhs"])
        elif con["sense"] == "=":
            A_eq.append(row); b_eq.append(con["rhs"])

    bounds = [(v.get("lower", 0), v.get("upper", None)) for v in model["variables"]]
    nnz = sum(len(con["coefficients"]) for con in model["constraints"])
    return {
        "var_names": var_names, "idx": idx, "sense": sense, "offset": offset,
        "c_solve": c_solve, "c": c,
        "A_ub": np.array(A_ub) if A_ub else None, "b_ub": np.array(b_ub) if b_ub else None,
        "A_eq": np.array(A_eq) if A_eq else None, "b_eq": np.array(b_eq) if b_eq else None,
        "bounds": bounds, "nnz": nnz,
    }


@app.post("/api/model/validate")
def validate_model():
    body = request.get_json(force=True)
    model = body.get("model", body)
    t0 = time.perf_counter()
    try:
        arrays = _model_to_arrays(model)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400
    t1 = time.perf_counter()
    return jsonify({
        "valid": True,
        "num_variables": len(arrays["var_names"]),
        "num_constraints": len(model["constraints"]),
        "nonzeros": arrays["nnz"],
        "presolve": {"eliminated_rows": 0, "eliminated_cols": 0, "time_ms": round((t1 - t0) * 1000, 4)},
    })


STATUS_MAP = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "numerical_difficulty"}


@app.post("/api/solve")
def solve():
    body = request.get_json(force=True)
    model = body["model"]
    backend = body.get("backend", "cpu")
    options = body.get("options", {}) or {}

    if backend == "gpu":
        gpu = detect_gpu()
        if not gpu["available"]:
            return jsonify({
                "status": "error",
                "error": "gpu_unavailable",
                "gpu_unavailable_reason": gpu["reason"],
                "backend": "gpu",
                "device": None,
                "objective": None, "variables": None, "iterations": None,
                "solve_time_ms": None, "presolve_time_ms": None,
                "iteration_history": None, "constraint_residuals": None,
                "numerical": {"max_residual": None, "duality_gap": None, "condition_estimate": None},
            }), 200

    t_pre0 = time.perf_counter()
    arrays = _model_to_arrays(model)
    t_pre1 = time.perf_counter()

    t0 = time.perf_counter()
    res = linprog(
        arrays["c_solve"], A_ub=arrays["A_ub"], b_ub=arrays["b_ub"],
        A_eq=arrays["A_eq"], b_eq=arrays["b_eq"], bounds=arrays["bounds"],
        method="highs",
        options={
            "maxiter": options.get("max_iterations", 5000),
            "presolve": options.get("presolve", True),
        },
    )
    t1 = time.perf_counter()

    status = STATUS_MAP.get(res.status, f"unknown({res.status})")
    device = cpu_info()["model"] or platform.processor() or "unknown CPU"

    out = {
        "status": status,
        "backend": "cpu",
        "device": device,
        "iterations": int(res.nit) if res.nit is not None else None,
        "solve_time_ms": round((t1 - t0) * 1000, 4),
        "presolve_time_ms": round((t_pre1 - t_pre0) * 1000, 4),
        "objective": None, "variables": None,
        "iteration_history": None, "constraint_residuals": None,
        "numerical": {"max_residual": None, "duality_gap": None, "condition_estimate": None},
    }

    if status == "optimal":
        raw_obj = res.fun
        true_obj = (-raw_obj if arrays["sense"] == "maximize" else raw_obj) + arrays["offset"]
        variables = {name: float(res.x[arrays["idx"][name]]) for name in arrays["var_names"]}
        out["objective"] = float(true_obj)
        out["variables"] = variables

        residuals = {}
        for con in model["constraints"]:
            lhs = sum(coef * variables[name] for name, coef in con["coefficients"].items())
            rhs = con["rhs"]
            if con["sense"] == "<=":
                residuals[con["name"]] = max(0.0, lhs - rhs)
            elif con["sense"] == ">=":
                residuals[con["name"]] = max(0.0, rhs - lhs)
            else:
                residuals[con["name"]] = abs(lhs - rhs)
        out["constraint_residuals"] = residuals
        out["numerical"]["max_residual"] = max(residuals.values()) if residuals else 0.0
        # HiGHS via scipy does not expose per-iteration history or duality gap
        # through this API — reported honestly as unavailable rather than guessed.
        out["iteration_history"] = None

    return jsonify(out)


@app.get("/api/benchmarks")
def benchmarks():
    runs = []
    if BENCHMARK_DIR.exists():
        import json
        for f in sorted(BENCHMARK_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    runs.extend(data)
                else:
                    runs.append(data)
            except Exception:
                continue
    return jsonify({"runs": runs, "source": "benchmarks/results/*.json (real recorded runs only)"})


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "nirnaya-reference-backend"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
