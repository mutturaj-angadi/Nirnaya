# Validation and measured results

## NOC-10K scientific evaluation

The deterministic corpus, seed/hash schema, exact family allocation, 80/10/10 split, run harness, independent reference labeling, profile aggregation, and explicit measurement limits are documented in [NOC10K.md](NOC10K.md). Its generated manifest contains 10,000 unique IDs/hashes and 8,000/1,000/1,000 split counts. The official corpus run and per-family/holdout reports are emitted under `benchmarks/results/noc10k/`; use an exact run JSONL as the analysis input. Do not treat a partial or interrupted run as a 10,000-case result.

The solver now uses bounded product-form eta basis updates with periodic basis residual checks and refactorization; details and measured NOC-10K evidence are in [SOLVER_ALGORITHM.md](SOLVER_ALGORITHM.md) and [NOC10K.md](NOC10K.md). It retains the same deterministic pivot rule, tolerances, and original-model verification.

## Fresh judge-pass baseline — 2026-09-28

All API-backed checks targeted a fresh canonical launcher process loading the workspace source tree, not the reference backend or an installed wheel. The launcher printed `packages/nirnaya-api/src/nirnaya_api/server.py` and `packages/nirnaya-solver/src/nirnaya_solver/solver.py`. API/UI used `127.0.0.1:8153` and `127.0.0.1:8154`. Machine: Windows 11 build 26200, Python 3.13.5, Intel64 Family 6 Model 170, 18 logical cores. Backend: CPU. Reference validation: SciPy 1.17.0 / HiGHS. CUDA/CuPy unavailable.

| Check | Timestamp (Asia/Kolkata) | Exact command | Result |
|---|---|---|---|
| Fresh API + UI | 2026-09-28 14:11 | `python scripts/launch_demo.py --api-port 8153 --ui-port 8154 --no-browser` | `/health` OK; source paths printed; UI HTTP 200 |
| Live HTTP release matrix | 2026-09-28 14:20:24 | `python scripts/run_release_validation.py --base-url http://127.0.0.1:8153` | 12/12 passed; 0 time limits |
| Independent live reference suite | 2026-09-28 14:20:30 | `python apps/nirnaya-ui/validation/run_validation_suite.py --base-url http://127.0.0.1:8153 --backends cpu` | 5 passed, 0 failed/unavailable |
| UI/API E2E | 2026-09-28 14:28:43 | `python -m pytest apps/nirnaya-ui/tests/e2e tests/ui/e2e -q` | 12 passed in 12.54 s |
| Root pytest | 2026-09-28 14:29:10 | `python -m pytest -q` | 274 passed, 8 hardware skips, 0 errors, 12.64 s |
| In-process benchmark matrix | 2026-09-28 14:21:43 | `python benchmarks/run_benchmark_matrix.py --time-limit 30` | 18 total: 16 optimal, 1 infeasible, 1 unbounded, 0 time limits/numerical/execution errors |
| Rendered browser smoke | 2026-09-28 14:28 | Headless Edge 153 via DevTools Protocol against the configured launcher UI | All views opened; flagship solved; 6 scenarios, 5 validation rows, 18 benchmark rows; GPU unavailable honestly shown; no runtime/console errors; no horizontal overflow at 1440 px or 390 px; offline state cleared results and recovered on reconnect |
| Isolated editable install | 2026-09-28 14:30:47 | `python -m pip install --no-build-isolation --no-deps -e packages/nirnaya-core -e packages/nirnaya-presolve -e packages/nirnaya-gpu -e packages/nirnaya-solver -e packages/nirnaya-api` in a fresh `--system-site-packages` venv | All five editables built and installed; imported API module resolved to this workspace source |

The live sparse model in the release matrix completed optimally in 5.237 s solver time (5.440 s HTTP), 8,436 iterations on CPU, objective `10313.470999999998`, objective delta `3.64e-12` versus the independent reference, and maximum original-model violation `3.20e-14`. The in-process matrix completed the same source-hashed model in 5.141 s solver time and 5.320 s end to end. Its phase timings were 0.233 s Phase I, 4.891 s Phase II, 3.531 s factorization, 0.245 s pricing, and 0.185 s ratio test; timings overlap and should not be summed. These are host measurements, not a speed guarantee. See `benchmarks/results/reproducible/latest.json` and the latest archived release-gate artifact.

The rendered browser pass exposed and verified the fix for an API/UI integration defect: the API lacked CORS access for the separately served dashboard. The launcher now adds its exact UI origin to the API allowlist. The UI benchmark panel distinguishes the in-process matrix from live HTTP release validation and the synthetic validation suite, and shows all status counts. Concurrent heavy API requests still have no admission control and can contend for CPU; the observed timeout remains archived, while that old run's exact process state cannot be proven.

## Historical freeze baseline before eta updates — 2026-09-28

After correcting the sparse dataset's descriptive GPU wording (no coefficients or constraints changed), the refreshed case SHA-256 was `e95d3ed4d19bbe4ea6cb80707f91098a24ac2ef401359ea9dda44b17c7cfc791`. A fresh in-process matrix completed all 18 records (16 optimal, one infeasible, one unbounded). Its sparse case completed with 8,436 iterations, CPU backend, objective `10313.470999999998`, independent HiGHS objective `10313.471000000001`, objective difference `3.64e-12`, and original-model maximum violation `3.20e-14`; solver time was 23.149 s and end-to-end time 23.531 s. Phase I was 0.312 s, Phase II 22.821 s, factorization 16.274 s, pricing 1.035 s, and ratio test 0.776 s (profile components overlap). The 30-second live release run made immediately afterward reached the configured deadline at 7,606 iterations and is preserved in its archived release-gate record. A separate fresh independent validation suite passed all five reference checks, including the sparse case. This timeout occurred on the pre-eta solver revision; see the current revision results below.

## Sparse regression investigation

The current sparse model is the same serialized input in the benchmark, validation, and E2E paths: SHA-256 `e95d3ed4d19bbe4ea6cb80707f91098a24ac2ef401359ea9dda44b17c7cfc791` (1,440 variables, 300 rows, 2,880 nonzeros). Current API defaults are a 30-second solver deadline and 10,000 iterations. The validation harness requests 20,000 iterations through its legacy option alias but leaves the time limit at 30 seconds. This does not explain a time limit at about 5,000 iterations.

The benchmark matrix calls `solve_model` in-process. The validation harness and E2E tests call HTTP at `NIRNAYA_API_URL`; historically their default was a fixed `localhost:8000`, while the demo launcher allowed a different API port without configuring the UI. That made a stale or unrelated listener possible. The old timeout report did not record its server identity, so its exact source cannot now be proven. At the start of this investigation no API listener was present on the usual ports. A fresh launcher started the source-tree packages on port 8123; `/health` and `/api/health` both returned `ok`.

Sequential runs against that fresh API completed the sparse model at optimal status through every path:

| Path | Status | Solve time | Iterations | Details |
|---|---:|---:|---:|---|
| Direct solver, no presolve | optimal | 8.791 s | 8,445 | sparse LU, current source |
| API integration, presolve/postsolve | optimal | 8.751 s wall; 8.376 s solver | 8,436 | maximum original-space violation 3.20e-14 |
| Direct HTTP request | optimal | 8.529 s HTTP; 8.204 s solver | 8,436 | same model hash, CPU, objective 10,313.471 |
| Live validation harness | optimal | 14.989 s solver | 8,436 | reference objective delta 3.64e-12; maximum residual 3.20e-14 |
| Earlier canonical HTTP release matrix | optimal | 11.442 s solver; 11.9 s HTTP | 8,436 | reference objective delta 3.64e-12; maximum violation 3.20e-14 |
| E2E tests | pass | 12 E2E checks in 24.29 s | — | both discovered test trees ran against the configured source API |

These are separate local runs on a shared Windows host, so their times vary with machine load and are not a speed guarantee. Sequential fresh-API release runs ranged from 11.44 to 16.39 seconds for the sparse solve; all completed under the unchanged limit.

One **failed time-limit outcome is retained** at [`concurrent-contention-reproduction.json`](../benchmarks/results/release-gate/runs/concurrent-contention-reproduction.json): an API solve reached 30.001 s and stopped after 5,063 iterations while three sparse workloads were competing against the same API; two client commands were interrupted during that run, and their server-side work may have continued. The sequential rerun completed. This is evidence of CPU contention sensitivity, not a reason to increase the deadline or hide a failure. The current API does not queue or limit simultaneous simplex requests; avoid launching multiple sparse validations concurrently on this host.

The baseline profile above predates bounded product-form eta updates. The current implementation keeps sparse matrices and a sparse LU base, reuses pivot updates with periodic residual checks, and refactorizes at a bounded interval. The current sparse completion and NOC-10K profile are documented in [NOC10K.md](NOC10K.md); the earlier slow completion and timeout remain preserved here as baseline evidence.

## Current quality-gate evidence

- Unfiltered `python -m pytest -q` against the live current source API (`NIRNAYA_API_URL=http://127.0.0.1:8153`): **283 passed, 8 skipped, 0 errors**. The skips require unavailable CUDA/CuPy hardware.
- Both E2E test trees against the fresh source API: **12 passed**.
- Independent live validation harness: **5 passed, 0 failed, 0 unavailable**.
- Fresh live release matrix: **12/12 passed**, including sparse under the unchanged 30-second limit. Earlier sparse timeouts are retained in archived runs as evidence of contention sensitivity.
- In-process production benchmark matrix: 18 records: 16 optimal, one infeasible, one unbounded, zero time limits/numerical errors/execution errors.
- NOC-10K production/reference evaluation: **10,000/10,000 PASS**, full 80/10/10 split, 100% status agreement and 100% objective/original-feasibility agreement over all 9,250 optimal cases; zero timeouts, numerical errors, execution errors or unavailable references.
- Synthetic suite: 7/7 optimal. Refinery scenarios: 6/6 optimal. Dataset audit: 19 files audited without modification; manifest: 18 records.
- API-client unit checks: 6 passed; JavaScript syntax checks passed.
- Editable installation of the five local packages in a temporary venv succeeded with system dependencies available.

The root pytest configuration directs `tmp_path` to ignored, repository-local `.pytest-tmp` so tests do not depend on the user-profile temporary directory. A run with that configuration completed without permission errors. The historical standalone `packages/nirnaya-presolve/tests` remains an older suite that imports the removed `nirnaya_core.Sense` API; it is not part of the root `testpaths` and has not been counted as passed.

Rendered browser verification used headless Microsoft Edge 153 through the DevTools Protocol. It covered landing, connection, model, optimization, solution, presolve, scenarios, validation, benchmark evidence, architecture, security, and GPU views. This local smoke test is not a cross-browser or assistive-technology audit. Consult exact archived per-run JSON and the current release manifest when presenting status.

The standalone historical `packages/nirnaya-presolve/tests` suite was also invoked, but all 13 modules fail collection because they target removed `nirnaya_core.Sense`/`Model` APIs and an obsolete test shim. These legacy tests are not counted as passed; the supported integrated presolve paths are exercised by the root suite, 13 solver/integration checks, NOC-10K, and live API validation.

## Reproduction

Start the canonical demo from the repository root in Terminal 1:

```text
python scripts/launch_demo.py
```

### PowerShell

Offline unit/core mode (API-dependent E2E tests skip when the API is stopped):

```powershell
$env:PYTHONPATH='packages/nirnaya-core/src;packages/nirnaya-presolve/src;packages/nirnaya-gpu/src;packages/nirnaya-solver/src;packages/nirnaya-api/src'
Remove-Item Env:NIRNAYA_API_URL -ErrorAction SilentlyContinue
python -m pytest -q
```

In Terminal 2, full live E2E mode:

```powershell
$env:PYTHONPATH='packages/nirnaya-core/src;packages/nirnaya-presolve/src;packages/nirnaya-gpu/src;packages/nirnaya-solver/src;packages/nirnaya-api/src'
$env:NIRNAYA_API_URL='http://127.0.0.1:8000'
python -m pytest apps/nirnaya-ui/tests/e2e tests/ui/e2e -q
```

### Command Prompt (CMD)

Offline unit/core mode:

```bat
set PYTHONPATH=packages/nirnaya-core/src;packages/nirnaya-presolve/src;packages/nirnaya-gpu/src;packages/nirnaya-solver/src;packages/nirnaya-api/src
set NIRNAYA_API_URL=
python -m pytest -q
```

In Terminal 2, full live E2E mode:

```bat
set PYTHONPATH=packages/nirnaya-core/src;packages/nirnaya-presolve/src;packages/nirnaya-gpu/src;packages/nirnaya-solver/src;packages/nirnaya-api/src
set NIRNAYA_API_URL=http://127.0.0.1:8000
python -m pytest apps/nirnaya-ui/tests/e2e tests/ui/e2e -q
```

Do not paste Markdown headings or lines beginning with `#` into CMD; these are explanatory comments, not commands.

With the live API running, validate independently:

```text
python scripts/run_release_validation.py --base-url http://127.0.0.1:8000
python apps/nirnaya-ui/validation/run_validation_suite.py --base-url http://127.0.0.1:8000 --backends cpu
```

`--timeout` on the release-matrix command is the HTTP client wait, not the solver limit. Each solve continues to use the configured 30-second deadline.
