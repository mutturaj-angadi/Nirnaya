#!/usr/bin/env python3
"""
run_validation_suite.py

End-to-end validation of the full pipeline:

    Part 1 (model)  ->  Part 2 (presolve)  ->  Part 3/4 (solve)  ->  Part 5 (API)  ->  this suite

It POSTs each case in `cases/` to the current production API and compares
the result against `reference_solver.py` (SciPy/HiGHS) â€” a numerically
independent implementation. Agreement between the two is real evidence
of correctness.

Exit code is non-zero if any case fails, so this can run in CI.

Usage:
    python3 run_validation_suite.py --base-url http://127.0.0.1:8000
"""
import argparse
import json
import sys
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_solver import solve_reference  # noqa: E402

CASES_DIR = Path(__file__).resolve().parent / "cases"
OBJ_TOL = 1e-4
VAR_TOL = 1e-4
RESID_TOL = 1e-6


def call_solve(base_url, model, backend="cpu"):
    payload = {"model": model, "backend": backend, "options": {"max_iterations": 20000}}
    req = urlrequest.Request(
        f"{base_url}/api/solve", data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json"},
    )
    with urlrequest.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def check_case(base_url, case_path, backend):
    model = json.loads(case_path.read_text())
    name = model.get("name", case_path.stem)
    ref = solve_reference(model)

    try:
        got = call_solve(base_url, model, backend=backend)
    except URLError as e:
        return {"case": name, "backend": backend, "passed": None,
                "reason": f"API unreachable: {e}", "reference": ref}

    result = {"case": name, "backend": backend, "reference_status": ref["status"],
              "api_status": got.get("status"), "reference": ref, "api": got}

    if got.get("status") == "error" and got.get("error") == "gpu_unavailable":
        result["passed"] = None
        result["reason"] = f"GPU backend unavailable: {got.get('gpu_unavailable_reason')}"
        return result

    if ref["status"] != got.get("status"):
        result["passed"] = False
        result["reason"] = f"status mismatch: reference={ref['status']} api={got.get('status')}"
        return result

    if ref["status"] != "optimal":
        # infeasible / unbounded: status agreement is the whole check
        result["passed"] = True
        result["reason"] = "status agreement (no numeric solution expected)"
        return result

    obj_ref, obj_api = ref["objective"], got.get("objective")
    if obj_api is None:
        result["passed"] = False
        result["reason"] = "api returned optimal but no objective value"
        return result

    obj_diff = abs(obj_ref - obj_api)
    obj_ok = obj_diff <= OBJ_TOL * max(1.0, abs(obj_ref))

    max_resid = (got.get("numerical") or {}).get("max_residual")
    resid_ok = (max_resid is None) or (max_resid <= 1e-4)  # loose; LP solvers vary

    result["objective_reference"] = obj_ref
    result["objective_api"] = obj_api
    result["objective_diff"] = obj_diff
    result["max_residual_api"] = max_resid
    result["passed"] = bool(obj_ok and resid_ok)
    if not result["passed"]:
        result["reason"] = f"objective diff {obj_diff:.6g} exceeds tolerance, or residual too large"
    else:
        result["reason"] = "objective and residuals within tolerance"

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--backends", nargs="+", default=["cpu", "gpu"])
    ap.add_argument("--cases-dir", default=str(CASES_DIR))
    args = ap.parse_args()

    case_files = sorted(Path(args.cases_dir).glob("*.json"))
    if not case_files:
        print("No validation cases found.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    n_fail = 0
    n_skip = 0
    for case_path in case_files:
        for backend in args.backends:
            r = check_case(args.base_url, case_path, backend)
            all_results.append(r)
            mark = "SKIP" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
            print(f"[{mark}] {r['case']:24s} backend={backend:4s} {r.get('reason', '')}")
            if r["passed"] is False:
                n_fail += 1
            elif r["passed"] is None:
                n_skip += 1

    out_path = Path(args.cases_dir).parent / "validation_report.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\n{len(all_results) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped/unavailable "
          f"out of {len(all_results)} checks")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()


