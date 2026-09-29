#!/usr/bin/env python3
"""Run Nirnaya and an independent reference over selected NOC-10K cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in (
    "packages/nirnaya-core/src", "packages/nirnaya-presolve/src",
    "packages/nirnaya-gpu/src", "packages/nirnaya-solver/src",
    "packages/nirnaya-api/src", "apps/nirnaya-ui/validation")]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nirnaya_api.options import SolveOptions  # noqa: E402
from nirnaya_api.orchestrator import solve_model  # noqa: E402
from reference_solver import solve_reference  # noqa: E402
from noc10k_common import (DEFAULT_SEED, FAMILY_COUNTS, iter_specs, materialize_instance)  # noqa: E402

TOL = 1e-7
MANIFEST = ROOT / "datasets/noc10k/manifest.jsonl.gz"


def load_manifest(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def machine_info() -> dict:
    import scipy
    return {"platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "scipy": scipy.__version__, "memory_bytes": None,
            "memory_measurement_note": "not measured; no portable per-case RSS sampler configured"}


def violations(model: dict, values: dict | None) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    constraint = 0.0
    for row in model["constraints"]:
        lhs = sum(float(c) * values.get(name, 0.0)
                  for name, c in row["coefficients"].items())
        rhs, sense = float(row["rhs"]), row["sense"]
        err = abs(lhs - rhs) if sense in ("=", "==") else max(0.0, lhs-rhs) if sense == "<=" else max(0.0, rhs-lhs)
        constraint = max(constraint, err)
    bound = 0.0
    for var in model["variables"]:
        x = values.get(var["name"], 0.0)
        lo, hi = var.get("lower", 0.0), var.get("upper")
        if lo is not None: bound = max(bound, float(lo)-x)
        if hi is not None: bound = max(bound, x-float(hi))
    return constraint, max(0.0, bound)


def classification(nirnaya: str, reference: str, objective: float | None,
                   reference_objective: float | None, cviol: float | None,
                   bviol: float | None, ref_feasible: bool | None) -> str:
    if nirnaya == "time_limit": return "NIRNAYA_TIMEOUT"
    if nirnaya == "numerical_error": return "NUMERICAL_WARNING"
    if nirnaya in ("iteration_limit", "error"): return "NIRNAYA_ERROR"
    if reference == "unavailable": return "REFERENCE_UNAVAILABLE"
    if nirnaya != reference: return "STATUS_MISMATCH"
    if nirnaya == "optimal":
        if cviol is None or bviol is None or cviol > TOL or bviol > TOL or ref_feasible is False:
            return "FEASIBILITY_MISMATCH"
        if objective is None or reference_objective is None:
            return "REFERENCE_UNAVAILABLE"
        if abs(objective-reference_objective) > TOL * max(1.0, abs(reference_objective)):
            return "OBJECTIVE_MISMATCH"
    return "PASS"


def run_one(job: tuple[dict, int]) -> dict:
    meta, corpus_seed = job
    family, index, instance_seed = meta["family"], meta["family_index"], meta["seed"]
    # Count is reconstructed from the standard allocation recorded in the manifest split map.
    family_count = meta["generation_parameters"].get("family_count", FAMILY_COUNTS.get(family, 0))
    # Family allocations are fixed for the default corpus. The split is already
    # embedded and serves as the authoritative assignment for this generated input.
    model, regenerated = materialize_instance(family, instance_seed, index,
        FAMILY_COUNTS.get(family, family_count), corpus_seed)
    if regenerated["model_sha256"] != meta["model_sha256"]:
        return {"instance_id": meta["instance_id"], "classification": "NIRNAYA_ERROR",
                "failure_reason": "regenerated model hash differs from manifest",
                "model_sha256": meta["model_sha256"], "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    reference_started = None
    ref_result = None
    ref_error = None
    nirnaya_result = None
    try:
        nirnaya_result = solve_model(model, SolveOptions(backend="cpu", presolve=True,
            verbose=True, time_limit=30.0, iteration_limit=10_000))
    except Exception as exc:  # Keep one malformed/unexpected instance in the corpus record.
        nirnaya_error = f"{type(exc).__name__}: {exc}"
    else:
        nirnaya_error = None
    nirnaya_wall = time.perf_counter() - started
    reference_started = time.perf_counter()
    try:
        ref_result = solve_reference(model, time_limit=30.0)
    except Exception as exc:
        ref_error = f"{type(exc).__name__}: {exc}"
    reference_wall = time.perf_counter() - reference_started

    if nirnaya_result is None:
        nstatus, objective, values = "error", None, None
        iterations, solver_time, presolve_time, extra, pinfo = 0, None, None, {}, {}
    else:
        nstatus = nirnaya_result.status.value
        objective, values = nirnaya_result.objective_value, nirnaya_result.variable_values
        iterations, solver_time = nirnaya_result.iterations, nirnaya_result.solve_time_s
        presolve_time, extra, pinfo = nirnaya_result.presolve_time_s, nirnaya_result.diagnostics.extra, nirnaya_result.presolve_info
    ref_status = (ref_result or {}).get("status", "unavailable")
    if ref_status in ("iteration_limit", "numerical_difficulty") or ref_error:
        ref_unavailable_reason = ref_error or (ref_result or {}).get("raw_message", ref_status)
        ref_status_for_compare = "unavailable"
    else:
        ref_unavailable_reason = None
        ref_status_for_compare = ref_status
    constraint_violation, bound_violation = violations(model, values) if nstatus == "optimal" else (None, None)
    ref_values = (ref_result or {}).get("variables")
    ref_cviol, ref_bviol = violations(model, ref_values) if ref_values else (None, None)
    ref_feasible = (ref_cviol <= TOL and ref_bviol <= TOL) if ref_cviol is not None else None
    category = classification(nstatus, ref_status_for_compare, objective,
        (ref_result or {}).get("objective"), constraint_violation, bound_violation, ref_feasible)
    if category == "PASS":
        failure_reason = None
    elif nirnaya_error:
        failure_reason = nirnaya_error
    elif nstatus == "time_limit":
        failure_reason = "Nirnaya reached its configured 30-second solver deadline"
    elif nstatus == "numerical_error":
        failure_reason = "Nirnaya reported a numerical error"
    elif ref_unavailable_reason:
        failure_reason = f"independent reference unavailable: {ref_unavailable_reason}"
    elif category == "FEASIBILITY_MISMATCH":
        failure_reason = "original-space primal feasibility checks disagree or fail"
    elif category == "OBJECTIVE_MISMATCH":
        failure_reason = "optimal objectives differ beyond the documented tolerance"
    elif category == "STATUS_MISMATCH":
        failure_reason = f"Nirnaya={nstatus}, reference={ref_status_for_compare}"
    else:
        failure_reason = category
    stats = pinfo.get("statistics", {})
    diag = extra if isinstance(extra, dict) else {}
    try:
        solver_hash = hashlib.sha256((ROOT / "packages/nirnaya-solver/src/nirnaya_solver/solver.py").read_bytes()).hexdigest()
    except OSError:
        solver_hash = None
    try:
        presolve_hash = hashlib.sha256((ROOT / "packages/nirnaya-presolve/src/nirnaya_presolve/presolve/passes.py").read_bytes()).hexdigest()
        api_hash = hashlib.sha256((ROOT / "packages/nirnaya-api/src/nirnaya_api/_integration.py").read_bytes()).hexdigest()
    except OSError:
        presolve_hash = api_hash = None
    presolve_terminal = bool(pinfo.get("infeasible") or pinfo.get("unbounded"))
    return {
        "instance_id": meta["instance_id"], "family": family, "split": meta["split"],
        "family_index": index, "corpus_seed": corpus_seed,
        "model_sha256": meta["model_sha256"], "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "solver_version": "workspace-source", "solver_source_sha256": solver_hash,
        "presolve_source_sha256": presolve_hash, "api_integration_source_sha256": api_hash,
        "backend": nirnaya_result.backend if nirnaya_result else "cpu",
        "status": nstatus, "objective": objective, "iterations": iterations,
        "phase1_iterations": diag.get("phase1_iterations", 0 if presolve_terminal else None),
        "phase2_iterations": diag.get("phase2_iterations", 0 if presolve_terminal else None),
        "solve_time_seconds": solver_time, "solver_wall_seconds": nirnaya_wall,
        "presolve_time_seconds": presolve_time, "postsolve_time_seconds": None,
        "postsolve_time_note": "not separately exposed by production API",
        "factorization_time_seconds": diag.get("factorization_seconds"),
        "basis_lu_factorizations": diag.get("factorization_count", 0 if presolve_terminal else None),
        "factor_solve_time_seconds": diag.get("factor_solve_seconds"),
        "basis_update_count": diag.get("basis_update_count", 0 if presolve_terminal else None),
        "basis_update_time_seconds": diag.get("basis_update_seconds"),
        "basis_residual_check_time_seconds": diag.get("basis_residual_check_seconds"),
        "pricing_time_seconds": diag.get("pricing_seconds"),
        "ratio_test_time_seconds": diag.get("ratio_test_seconds"),
        "max_constraint_violation": constraint_violation, "max_bound_violation": bound_violation,
        "presolve_original_variables": stats.get("original_num_variables"),
        "presolve_reduced_variables": stats.get("reduced_num_variables"),
        "presolve_original_constraints": stats.get("original_num_constraints"),
        "presolve_reduced_constraints": stats.get("reduced_num_constraints"),
        "presolve_variables_removed": ((stats.get("original_num_variables") - stats.get("reduced_num_variables"))
            if stats.get("original_num_variables") is not None and stats.get("reduced_num_variables") is not None else None),
        "presolve_constraints_removed": ((stats.get("original_num_constraints") - stats.get("reduced_num_constraints"))
            if stats.get("original_num_constraints") is not None and stats.get("reduced_num_constraints") is not None else None),
        "presolve_fixed_variables": stats.get("variables_fixed"),
        "presolve_redundant_constraints": stats.get("rows_removed_redundant", 0),
        "presolve_duplicate_constraints": stats.get("rows_removed_duplicate", 0),
        "presolve_bound_tightening_count": stats.get("bounds_tightened"),
        "presolve_infeasibility_detected": bool(pinfo.get("infeasible", False)),
        "presolve_unbounded_detected": bool(pinfo.get("unbounded", False)),
        "memory_bytes": None, "memory_note": "not measured",
        "nirnaya_failure_reason": nirnaya_error or (nirnaya_result.warnings[0] if nirnaya_result and nstatus != "optimal" and nirnaya_result.warnings else None),
        "reference_status": ref_status_for_compare,
        "reference_status_raw": ref_status,
        "reference_objective": (ref_result or {}).get("objective"),
        "reference_feasibility": ref_feasible, "reference_max_violation": max(ref_cviol, ref_bviol) if ref_cviol is not None else None,
        "reference_runtime_seconds": reference_wall,
        "reference_solver": "SciPy linprog(method='highs')",
        "reference_version": __import__("scipy").__version__,
        "reference_failure_reason": ref_unavailable_reason,
        "classification": category,
        "failure_reason": failure_reason,
        "machine": machine_info(), "python_version": platform.python_version(),
        "generator_version": meta["generator_version"], "generation_seed": meta["seed"],
        "worker_count": _WORKERS,
    }


_WORKERS = 1


def main() -> None:
    global _WORKERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--family", choices=sorted(FAMILY_COUNTS))
    parser.add_argument("--split", choices=("development", "validation", "holdout"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.start < 0 or args.count is not None and args.count < 0:
        parser.error("workers/count must be positive/nonnegative and start nonnegative")
    if not args.manifest.exists():
        parser.error(f"manifest not found; run `python scripts/generate_noc10k.py`: {args.manifest}")
    rows = load_manifest(args.manifest)
    if args.seed != DEFAULT_SEED:
        expected = list(iter_specs(args.seed, args.family, args.count))
        rows = [materialize_instance(f, s, i, n, args.seed)[1]
                for f, i, s, n in expected]
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    end = len(rows) if args.end is None else min(args.end, len(rows))
    if end < args.start:
        parser.error("--end must be greater than or equal to --start")
    if args.count is not None:
        end = min(end, args.start + args.count)
    rows = rows[args.start:end]
    args.output = args.output or ROOT / "benchmarks/results/noc10k/runs" / (
        "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _WORKERS = args.workers
    jobs = [(r, args.seed) for r in rows]
    # executor.map preserves input order irrespective of completion order.
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        if args.workers == 1:
            results = map(run_one, jobs)
            for result in results:
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(run_one, jobs):
                    stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
    print(json.dumps({"requested": len(rows), "workers": args.workers,
        "output": str(args.output), "ordering": "manifest order"}, sort_keys=True))


if __name__ == "__main__":
    main()
