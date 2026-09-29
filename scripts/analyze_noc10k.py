#!/usr/bin/env python3
"""Aggregate NOC-10K JSONL results without discarding failed attempts."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noc10k_common import DEFAULT_SEED, FAMILY_COUNTS, materialize_instance  # noqa: E402

OUT = ROOT / "benchmarks/results/noc10k"
TIME_FIELDS = ("solve_time_seconds", "presolve_time_seconds", "postsolve_time_seconds",
               "factorization_time_seconds", "factor_solve_time_seconds",
               "basis_update_time_seconds", "basis_residual_check_time_seconds",
               "pricing_time_seconds", "ratio_test_time_seconds")


def read_records(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if path.suffix == ".jsonl" or path.suffix == ".gz":
            return [json.loads(line) for line in stream if line.strip()]
        payload = json.load(stream)
        return payload if isinstance(payload, list) else payload.get("records", [])


def quantiles(values: list[float | int | None]) -> dict:
    ordered = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not ordered:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None}
    def q(p: float) -> float:
        # Nearest-rank quantile (documented, deterministic for repeated runs).
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return {"count": len(ordered), "p50": q(.50), "p90": q(.90),
            "p95": q(.95), "p99": q(.99)}


def summarize(rows: list[dict]) -> dict:
    classifications: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        classifications[row.get("classification", "UNKNOWN")] = classifications.get(row.get("classification", "UNKNOWN"), 0) + 1
        statuses[row.get("status", "unknown")] = statuses.get(row.get("status", "unknown"), 0) + 1
    complete = sum(row.get("classification") == "PASS" for row in rows)
    completed_statuses = {"optimal", "infeasible", "unbounded"}
    completed = sum(row.get("status") in completed_statuses for row in rows)
    agree_den = sum(row.get("reference_status") != "unavailable" for row in rows)
    status_agree = sum(row.get("reference_status") != "unavailable" and row.get("status") == row.get("reference_status") for row in rows)
    opt_pairs = [row for row in rows if row.get("status") == "optimal" and row.get("reference_status") == "optimal"]
    obj_agree = sum(row.get("objective") is not None and row.get("reference_objective") is not None and
        abs(row["objective"] - row["reference_objective"]) <= 1e-7 * max(1.0, abs(row["reference_objective"]))
        for row in opt_pairs)
    feasibility_compared = [row for row in opt_pairs if row.get("reference_feasibility") is not None]
    feas_agree = sum(row.get("reference_feasibility") is True and
        (row.get("max_constraint_violation") or 0) <= 1e-7 and
        (row.get("max_bound_violation") or 0) <= 1e-7 for row in feasibility_compared)
    var_reductions = [r["presolve_variables_removed"] for r in rows if r.get("presolve_variables_removed") is not None]
    row_reductions = [r["presolve_constraints_removed"] for r in rows if r.get("presolve_constraints_removed") is not None]
    presolve_measured = sum(r.get("presolve_variables_removed") is not None and
                            r.get("presolve_constraints_removed") is not None for r in rows)
    reduced_cases = sum((r.get("presolve_variables_removed") or 0) > 0 or
                        (r.get("presolve_constraints_removed") or 0) > 0 for r in rows)
    result = {"total": len(rows), "optimal": statuses.get("optimal", 0),
        "infeasible": statuses.get("infeasible", 0), "unbounded": statuses.get("unbounded", 0),
        "timeout": statuses.get("time_limit", 0),
        "numerical_error": statuses.get("numerical_error", 0),
        "execution_error": statuses.get("error", 0),
        "iteration_limit": statuses.get("iteration_limit", 0),
        "reference_unavailable": sum(row.get("reference_status") == "unavailable" for row in rows),
        "successful_cases": complete,
        "successful_rate": complete / len(rows) if rows else None,
        "completed_cases": completed,
        "completion_rate": completed / len(rows) if rows else None,
        "classification_counts": dict(sorted(classifications.items())),
        "status_agreement": {"cases_compared": agree_den, "agreements": status_agree,
            "rate": status_agree / agree_den if agree_den else None},
        "objective_agreement": {"optimal_pairs": len(opt_pairs), "agreements": obj_agree,
            "rate": obj_agree / len(opt_pairs) if opt_pairs else None},
        "feasibility_agreement": {"optimal_pairs_compared": len(feasibility_compared),
            "agreements": feas_agree, "rate": feas_agree / len(feasibility_compared) if feasibility_compared else None},
        "timeout_rate": statuses.get("time_limit", 0) / len(rows) if rows else None,
        "numerical_failure_rate": statuses.get("numerical_error", 0) / len(rows) if rows else None,
        "presolve": {
            "cases_measured": presolve_measured,
            "median_variables_removed": statistics.median(var_reductions) if var_reductions else None,
            "median_constraints_removed": statistics.median(row_reductions) if row_reductions else None,
            "cases_with_any_reduction": reduced_cases,
            "fraction_with_any_reduction": reduced_cases / presolve_measured if presolve_measured else None,
            "largest_variable_reduction": max(var_reductions) if var_reductions else None,
            "largest_constraint_reduction": max(row_reductions) if row_reductions else None,
            "presolve_infeasibility_detections": sum(bool(r.get("presolve_infeasibility_detected")) for r in rows)},
        "solve_time_seconds": quantiles([r.get("solve_time_seconds") for r in rows]),
        "iterations": quantiles([r.get("iterations") for r in rows]),
        "basis_lu_factorizations": quantiles([r.get("basis_lu_factorizations") for r in rows]),
        "basis_lu_factorizations_per_simplex_iteration": quantiles([
            r["basis_lu_factorizations"] / r["iterations"] for r in rows
            if r.get("basis_lu_factorizations") is not None and r.get("iterations", 0) > 0]),
        "profiles": {key: quantiles([r.get(key) for r in rows]) for key in TIME_FIELDS}}
    return result


def preserve_regression_cases() -> int:
    """Copy every observed non-PASS LP and retain each attempt record."""
    runs = ROOT / "benchmarks/results/noc10k/runs"
    reg_dir = ROOT / "datasets/regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    models_written = set()
    for run_path in sorted(runs.glob("*.jsonl*")):
        for row in read_records(run_path):
            if row.get("classification") in (None, "PASS"):
                continue
            instance_id = row.get("instance_id")
            family = row.get("family")
            if not instance_id or family not in FAMILY_COUNTS:
                continue
            case_dir = reg_dir / instance_id
            case_dir.mkdir(parents=True, exist_ok=True)
            model_path = case_dir / "model.json"
            if instance_id not in models_written and not model_path.exists():
                index = int(row.get("family_index", instance_id.rsplit("-", 1)[-1]))
                model, meta = materialize_instance(family, row["generation_seed"], index,
                    FAMILY_COUNTS[family], row.get("corpus_seed", DEFAULT_SEED))
                if meta["model_sha256"] != row.get("model_sha256"):
                    raise ValueError(f"cannot reproduce failed case {instance_id}: model hash differs")
                model_path.write_text(json.dumps(model, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
                models_written.add(instance_id)
            attempt_name = f"{run_path.stem}_{row.get('timestamp_utc','unknown').replace(':','-')}.json"
            attempt_path = case_dir / attempt_name
            if not attempt_path.exists():
                attempt_path.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
            attempts.append({"instance_id": instance_id, "run": run_path.name,
                             "attempt_record": str(attempt_path.relative_to(ROOT).as_posix()),
                             "classification": row.get("classification"), "status": row.get("status")})
    (reg_dir / "README.md").write_text(
        "# Preserved NOC-10K regression cases\n\n"
        "Every non-PASS case observed in a NOC-10K run is retained here with its deterministic model and per-attempt result JSON. "
        "Do not delete a case after a later run passes; original and corrective outcomes are separate files. "
        "This folder is refreshed by `python scripts/analyze_noc10k.py`.\n", encoding="utf-8")
    index_file = reg_dir / "attempts.jsonl"
    with index_file.open("w", encoding="utf-8", newline="\n") as stream:
        for row in attempts:
            stream.write(json.dumps(row, sort_keys=True)+"\n")
    return len(attempts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--family", choices=sorted(FAMILY_COUNTS))
    parser.add_argument("--split", choices=("development", "validation", "holdout"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1, help="recorded for report provenance")
    parser.add_argument("--input", type=Path, help="one run JSONL; defaults to the newest run")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    runs = sorted((args.output_dir / "runs").glob("run_*.jsonl*"), key=lambda p: p.name)
    source = args.input or (runs[-1] if runs else None)
    if source is None or not source.exists():
        parser.error("no run JSONL found; run scripts/run_noc10k.py first")
    rows = read_records(source)
    if args.family:
        rows = [r for r in rows if r.get("family") == args.family]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    end = len(rows) if args.end is None else min(args.end, len(rows))
    if args.count is not None: end = min(end, args.start + args.count)
    rows = rows[args.start:end]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    family_rows = []
    by_family = {}
    for family in sorted({r.get("family", "unknown") for r in rows}):
        group = [r for r in rows if r.get("family") == family]
        summary = summarize(group)
        family_rows.append({"family": family, **summary})
        by_family[family] = summary
    split_rows = {name: [r for r in rows if r.get("split") == name]
                  for name in ("development", "validation", "holdout")}
    by_family_split = {
        split: {family: summarize([r for r in records if r.get("family") == family])
                for family in sorted({r.get("family", "unknown") for r in records})}
        for split, records in split_rows.items()
    }
    summary = {"source_run": str(source), "seed": args.seed,
        "workers_argument": args.workers,
        "selection": {"family": args.family, "start": args.start, "end": end,
                      "requested_count": args.count},
        "observed": summarize(rows),
        "by_family": by_family,
        "by_family_split": by_family_split,
        "by_split": {name: summarize(group) for name, group in split_rows.items()},
        "expected_family_counts": FAMILY_COUNTS,
        "note": "Summarizes exactly this selected run; attempts are not merged or overwritten."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    overall = summary["observed"]
    summary_fields = ["total", "optimal", "infeasible", "unbounded", "timeout",
        "numerical_error", "execution_error", "iteration_limit", "reference_unavailable",
        "successful_cases", "successful_rate", "completed_cases", "completion_rate",
        "timeout_rate", "numerical_failure_rate"]
    profile_sections = tuple(dict.fromkeys(("solve_time_seconds", "iterations", *TIME_FIELDS)))
    for section in profile_sections:
        for q in ("p50", "p90", "p95", "p99"):
            summary_fields.append(f"{section}_{q}")
    summary_fields += ["objective_agreement_rate", "status_agreement_rate", "feasibility_agreement_rate",
        "presolve_median_variables_removed", "presolve_median_constraints_removed",
        "presolve_fraction_with_any_reduction", "presolve_largest_variable_reduction",
        "presolve_largest_constraint_reduction", "presolve_infeasibility_detections"]
    flat = {k: overall.get(k) for k in summary_fields}
    for section in profile_sections:
        for q in ("p50", "p90", "p95", "p99"):
            flat[f"{section}_{q}"] = overall.get(section, {}).get(q)
    for key in ("objective_agreement", "status_agreement", "feasibility_agreement"):
        flat[f"{key}_rate"] = overall.get(key, {}).get("rate")
    for key in summary_fields:
        if key.startswith("presolve_"):
            flat[key] = overall.get("presolve", {}).get(key.removeprefix("presolve_"))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields); writer.writeheader(); writer.writerow(flat)
    with (out / "family_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["family", "total", "optimal", "infeasible", "unbounded", "timeout",
                  "numerical_error", "execution_error", "iteration_limit", "reference_unavailable",
                  "successful_cases", "successful_rate", "completed_cases", "completion_rate", "timeout_rate",
                  "objective_agreement_rate", "status_agreement_rate", "feasibility_agreement_rate",
                  "solve_time_seconds_p50", "solve_time_seconds_p90", "solve_time_seconds_p95", "solve_time_seconds_p99",
                  "iterations_p50", "iterations_p90", "iterations_p95", "iterations_p99"]
        for metric in ("presolve_time_seconds", "factorization_time_seconds", "factor_solve_time_seconds",
                       "basis_update_time_seconds", "basis_residual_check_time_seconds", "pricing_time_seconds",
                       "ratio_test_time_seconds", "postsolve_time_seconds"):
            fields.extend(f"{metric}_p{q}" for q in (50, 90, 95, 99))
        fields.extend(["presolve_median_variables_removed", "presolve_median_constraints_removed",
                       "presolve_fraction_with_any_reduction", "presolve_largest_variable_reduction",
                       "presolve_largest_constraint_reduction", "presolve_infeasibility_detections"])
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        flattened = []
        for r in family_rows:
            row = {k: r.get(k) for k in fields}
            row.update({"objective_agreement_rate": r.get("objective_agreement", {}).get("rate"),
                "status_agreement_rate": r.get("status_agreement", {}).get("rate"),
                "feasibility_agreement_rate": r.get("feasibility_agreement", {}).get("rate")})
            for section in profile_sections:
                for q in (50, 90, 95, 99):
                    row[f"{section}_p{q}"] = (r.get("profiles", {}).get(section, {}).get(f"p{q}")
                        if section in TIME_FIELDS else r.get(section, {}).get(f"p{q}"))
            row.update({f"presolve_{k}": v for k, v in r.get("presolve", {}).items()
                        if f"presolve_{k}" in fields})
            flattened.append(row)
        writer.writerows(flattened)
    failure_rows = [r for r in rows if r.get("classification") != "PASS"]
    with (out / "failures.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["instance_id", "family", "split", "status", "classification",
                  "failure_reason", "nirnaya_failure_reason", "reference_status", "reference_failure_reason",
                  "solve_time_seconds", "iterations", "model_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in failure_rows)
    (out / "holdout_summary.json").write_text(json.dumps({"split": "holdout",
        "summary": summarize(split_rows["holdout"]),
        "warning": "Holdout results must not guide solver tuning."}, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    machine = rows[0].get("machine", {}) if rows else {}
    (out / "machine_info.json").write_text(json.dumps(machine, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    preserved_attempts = preserve_regression_cases()
    print(json.dumps({"source": str(source), "observed": len(rows),
                      "summary": str(out / "summary.json"), "failures": len(failure_rows),
                      "preserved_regression_attempts": preserved_attempts}, sort_keys=True))


if __name__ == "__main__":
    main()
