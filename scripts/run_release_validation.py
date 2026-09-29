"""Run the canonical live-API release validation matrix.

Requires a running Nirnaya API. The production answers come only from that
HTTP API; SciPy/HiGHS is invoked separately as a validation reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/validation"))
from reference_solver import solve_reference

from benchmarks.run_scenarios import transform_model


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def post_solve(base_url: str, model: dict, timeout: float) -> tuple[dict, float]:
    payload = json.dumps({"model": model, "backend": "cpu", "options": {
        "time_limit": 30, "iteration_limit": 10_000, "presolve": True,
        "verbose": True,
    }}).encode("utf-8")
    req = request.Request(base_url.rstrip("/") + "/api/solve", data=payload,
        method="POST", headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, time.perf_counter() - started


def evaluate(base_url: str, name: str, model: dict, timeout: float,
        source_path: Path | None = None) -> dict:
    result, http_wall = post_solve(base_url, model, timeout)
    reference = solve_reference(model)
    obs = result.get("observability") or {}
    numerical = result.get("numerical") or {}
    objective = result.get("objective")
    ref_objective = reference.get("objective")
    objective_delta = (abs(objective - ref_objective)
        if objective is not None and ref_objective is not None else None)
    objective_ok = result.get("status") != "optimal" or (
        objective_delta is not None and objective_delta <= 1e-4 * max(1.0, abs(ref_objective)))
    residual = numerical.get("max_residual")
    feasibility_ok = result.get("status") != "optimal" or (
        residual is not None and residual <= 1e-7)
    passed = result.get("status") == reference.get("status") and objective_ok and feasibility_ok
    return {
        "case": name,
        "model_sha256": hashlib.sha256(source_path.read_bytes() if source_path else
            json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "model_sha256_method": "source_file_bytes" if source_path else "canonical_json_bytes_for_transformed_scenario",
        "status": result.get("status", "missing"),
        "objective": objective,
        "reference_status": reference.get("status"),
        "reference_objective": ref_objective,
        "objective_abs_difference": objective_delta,
        "max_violation": residual,
        "solve_time_seconds": obs.get("solve_time_s", result.get("solve_time_ms", 0) / 1000),
        "http_wall_seconds": http_wall,
        "iterations": obs.get("iterations", result.get("iterations")),
        "backend": obs.get("backend", result.get("backend")),
        "passed": bool(passed),
        "reference_method": "SciPy linprog(method='highs'); validation only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0,
        help="HTTP client timeout; does not change the solver's 30-second limit")
    parser.add_argument("--out", type=Path,
        default=ROOT / "benchmarks/results/release-gate/latest.json")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    with request.urlopen(base + "/health", timeout=5) as response:
        health = json.loads(response.read().decode("utf-8"))
    if response.status != 200 or health.get("status") != "ok" or not all(health.get("checks", {}).values()):
        raise SystemExit(f"Nirnaya API health check failed: {health}")

    cases_dir = ROOT / "apps/nirnaya-ui/validation/cases"
    names = ("bounded_production", "feasible_lp", "infeasible_lp",
        "unbounded_lp", "sparse_large_lp")
    records = []
    for name in names:
        model_path = cases_dir / f"{name}.json"
        model = load(model_path)
        records.append(evaluate(base, name, model, args.timeout, model_path))

    flagship_path = ROOT / "examples/flagship/refinery_decision_demo.json"
    flagship = load(flagship_path)
    records.append(evaluate(base, "industrial_flagship", flagship, args.timeout, flagship_path))
    scenario_file = load(ROOT / "examples/flagship/scenarios.json")
    for scenario in scenario_file["scenarios"]:
        scenario_model, _ = transform_model(flagship, scenario)
        records.append(evaluate(base, f"refinery_scenario:{scenario['id']}",
            scenario_model, args.timeout))

    summary = {"total": len(records), "passed": sum(r["passed"] for r in records),
        "failed": sum(not r["passed"] for r in records),
        "time_limit": sum(r["status"] == "time_limit" for r in records)}
    output = {"run_id": "nirnaya_live_api_release_validation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_base_url": base, "api_health": health,
        "solver_options": {"backend": "cpu", "time_limit_seconds": 30,
            "iteration_limit": 10_000, "presolve": True,
            "reference_role": "independent validation only"},
        "summary": summary, "results": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.out.parent / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = output["generated_at_utc"].replace(":", "").replace("-", "")
    run_stamp = re.sub(r"[^0-9TZ]", "", run_stamp)
    run_path = run_dir / f"run_{run_stamp}.json"
    serialized = json.dumps(output, indent=2) + "\n"
    run_path.write_text(serialized, encoding="utf-8")
    args.out.write_text(serialized, encoding="utf-8")
    for item in records:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['case']}: "
            f"status={item['status']} objective={item['objective']} "
            f"max_violation={item['max_violation']} solve_s={item['solve_time_seconds']} "
            f"iterations={item['iterations']} backend={item['backend']}")
    print(json.dumps(summary))
    print(f"Archived run: {run_path}")
    print(f"Wrote {args.out}")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
