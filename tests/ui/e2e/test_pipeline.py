#!/usr/bin/env python3
"""
tests/e2e/test_pipeline.py

End-to-end pipeline tests: Part 1 (model) -> Part 2 (presolve) -> Part 3/4
(solve) -> Part 5 (API) -> Part 6 (this test).

Runs against the live integrated Nirnaya API configured by NIRNAYA_API_URL.
Uses pytest if available; otherwise runs as a plain script.

    pytest tests/e2e/test_pipeline.py
  or
    python3 tests/e2e/test_pipeline.py
"""
import json
import sys
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
import os
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
from reference_solver import solve_reference  # noqa: E402

BASE_URL = os.environ.get("NIRNAYA_API_URL", "http://127.0.0.1:8000")
CASES_DIR = ROOT / "validation" / "cases"


def _post(path, payload):
    req = urlrequest.Request(
        BASE_URL + path, data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json"},
    )
    with urlrequest.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _api_reachable():
    try:
        urlrequest.urlopen(BASE_URL + "/api/health", timeout=3)
        return True
    except URLError:
        return False


def _load(name):
    return json.loads((CASES_DIR / f"{name}.json").read_text())


def _run(name, backend="cpu"):
    model = _load(name)
    return solve_reference(model), _post("/api/solve", {"model": model, "backend": backend, "options": {}})


def test_feasible_lp():
    if not _api_reachable():
        pytest.skip(f"Nirnaya API not reachable at {BASE_URL}")
    ref, got = _run("feasible_lp")
    assert got["status"] == "optimal" == ref["status"]
    assert abs(got["objective"] - ref["objective"]) < 1e-4
    print("PASS test_feasible_lp")


def test_infeasible_lp():
    if not _api_reachable():
        pytest.skip(f"Nirnaya API not reachable at {BASE_URL}")
    ref, got = _run("infeasible_lp")
    assert got["status"] == "infeasible" == ref["status"]
    print("PASS test_infeasible_lp")


def test_unbounded_lp():
    if not _api_reachable():
        pytest.skip(f"Nirnaya API not reachable at {BASE_URL}")
    ref, got = _run("unbounded_lp")
    assert got["status"] == "unbounded" == ref["status"]
    print("PASS test_unbounded_lp")


def test_bounded_production():
    if not _api_reachable():
        pytest.skip(f"Nirnaya API not reachable at {BASE_URL}")
    ref, got = _run("bounded_production")
    assert got["status"] == "optimal" == ref["status"]
    assert abs(got["objective"] - ref["objective"]) < 1e-4 * max(1, abs(ref["objective"]))
    print("PASS test_bounded_production")


def test_sparse_large_lp():
    if not _api_reachable():
        pytest.skip(f"Nirnaya API not reachable at {BASE_URL}")
    ref, got = _run("sparse_large_lp")
    assert got["status"] == "optimal" == ref["status"]
    assert abs(got["objective"] - ref["objective"]) < 1e-3 * max(1, abs(ref["objective"]))
    max_resid = (got.get("numerical") or {}).get("max_residual")
    assert max_resid is not None and max_resid < 1e-4
    print("PASS test_sparse_large_lp")


def test_gpu_backend_reports_honestly():
    """GPU requests must either return real GPU numbers or an explicit
    unavailable reason — never CPU numbers silently relabeled as GPU."""
    if not _api_reachable():
        pytest.skip(f"Nirnaya API not reachable at {BASE_URL}")
    model = _load("feasible_lp")
    try:
        got = _post("/api/solve", {"model": model, "backend": "gpu", "options": {}})
    except HTTPError as exc:
        error = json.loads(exc.read().decode())
        assert exc.code == 409
        assert error["error"]["code"] == "DEVICE_UNAVAILABLE"
        print("PASS test_gpu_backend_reports_honestly: CPU-only solver rejects GPU request")
        return
    if got.get("status") == "error":
        assert got.get("gpu_unavailable_reason"), "GPU error response must explain why"
    else:
        assert got.get("backend") == "gpu"
        assert got.get("device"), "GPU solve must report a real device string"
    print("PASS test_gpu_backend_reports_honestly")


ALL_TESTS = [
    test_feasible_lp, test_infeasible_lp, test_unbounded_lp,
    test_bounded_production, test_sparse_large_lp, test_gpu_backend_reports_honestly,
]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
    sys.exit(1 if failures else 0)
