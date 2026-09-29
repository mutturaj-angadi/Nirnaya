#!/usr/bin/env python3
"""
run_benchmarks.py

Legacy optional client for the live production API (`--base-url`, default http://127.0.0.1:8000). It records only what the API actually returns for each requested backend.

If a backend is unavailable (e.g. no GPU), the run is recorded with
`*_time_ms: null` and the reported `*_unavailable_reason` â€” never a filled-in
guess. This script never invents numbers; it is a thin, honest client.

Usage:
    python3 run_benchmarks.py --base-url http://127.0.0.1:8000 \
        --models-dir ../demo-data --out results/run_$(date +%s).json
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


def call_api(base_url, path, payload=None, method="GET", timeout=120):
    url = f"{base_url}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urlrequest.Request(url, data=data, method=method,
                              headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_one(base_url, model_path, backend):
    model = json.loads(Path(model_path).read_text())
    payload = {"model": model, "backend": backend, "options": {"max_iterations": 20000}}
    t0 = time.perf_counter()
    try:
        result = call_api(base_url, "/api/solve", payload, method="POST")
    except HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            err = json.loads(e.read().decode()).get("error", {})
        except (ValueError, AttributeError):
            err = {}
        nnz = sum(len(c["coefficients"]) for c in model["constraints"])
        unavailable = err.get("message", "Backend unavailable")
        return {"problem": model.get("name", Path(model_path).stem),
            "num_variables": len(model["variables"]), "num_constraints": len(model["constraints"]),
            "nonzeros": nnz, "backend_requested": backend, "status": "unavailable",
            "objective": None, "iterations": None, "solve_time_ms": None,
            "presolve_time_ms": None, "wall_clock_ms": round(elapsed_ms, 4),
            "max_residual": None, "gpu_unavailable_reason": unavailable,
            "error_code": err.get("code"), "device": None,
            "timestamp": datetime.now(timezone.utc).isoformat()}
    except URLError as e:
        return {"error": f"request failed: {e}"}
    wall_ms = (time.perf_counter() - t0) * 1000

    nnz = sum(len(c["coefficients"]) for c in model["constraints"])
    record = {
        "problem": model.get("name", Path(model_path).stem),
        "num_variables": len(model["variables"]),
        "num_constraints": len(model["constraints"]),
        "nonzeros": nnz,
        "backend_requested": backend,
        "status": result.get("status"),
        "objective": result.get("objective"),
        "iterations": result.get("iterations"),
        "solve_time_ms": result.get("solve_time_ms"),
        "presolve_time_ms": result.get("presolve_time_ms"),
        "wall_clock_ms": round(wall_ms, 4),
        "max_residual": (result.get("numerical") or {}).get("max_residual"),
        "gpu_unavailable_reason": result.get("gpu_unavailable_reason"),
        "device": result.get("device"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parent.parent / "demo-data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--backends", nargs="+", default=["cpu", "gpu"])
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    model_files = sorted(p for p in models_dir.glob("*.json"))
    if not model_files:
        print(f"No model files found in {models_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        sysinfo = call_api(args.base_url, "/api/system/info")
    except URLError as e:
        print(f"Could not reach API at {args.base_url}: {e}", file=sys.stderr)
        print("Start the production API and dashboard with: python scripts/launch_demo.py", file=sys.stderr)
        sys.exit(1)

    runs = []
    for mf in model_files:
        for backend in args.backends:
            print(f"Running {mf.name} on backend={backend} ...")
            rec = run_one(args.base_url, mf, backend)
            runs.append(rec)
            status = rec.get("status", rec.get("error"))
            t = rec.get("solve_time_ms")
            print(f"  -> status={status} solve_time_ms={t}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base_url": args.base_url,
        "system_info": sysinfo,
        "runs": runs,
    }

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results" / f"benchmark_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()



