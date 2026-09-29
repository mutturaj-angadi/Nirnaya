# Nirnaya dashboard, validation, and benchmarks

This static dashboard connects to the integrated Nirnaya API: core model validation, presolve, the CPU two-phase revised-simplex solver, postsolve, and API response. Solver output and live metrics come from that API. Model definitions are bundled for browsing; archived reference result JSON is excluded from the browser bundle.

## Start the integrated demo

Run from the repository root:

```powershell
python scripts/launch_demo.py
```

The launcher starts the source-tree API and UI, waits for the API `/health` response, and opens or prints a UI URL configured for that API process. Use `--api-port` and `--ui-port` to select ports. The older SciPy-based `reference_backend/` service is retained for archive compatibility only; the production launcher never starts or uses it.

Opening `src/index.html` or serving the UI without the API allows model browsing. Optimization results, connection status, solver statistics, presolve values, and feasibility require the live API. When the API is unreachable, the UI reports `API OFFLINE` and clears live result metrics.

## Contents

- `src/`: static HTML, CSS, and JavaScript UI.
- `demo-data/`: illustrative model inputs.
- `validation/`: API validation harness and independent SciPy/HiGHS reference implementation; reference code is validation-only.
- `benchmarks/`: reproducible API benchmark tools and recorded artifacts.
- `reference_backend/`: archived compatibility service, not a production solve path.
- `tests/`: UI API-client and live API E2E checks.

## Independent validation

With the integrated API running:

```powershell
python apps/nirnaya-ui/validation/run_validation_suite.py --base-url http://127.0.0.1:8000 --backends cpu
```

The harness compares status, objective, and the API's original-model residual against an independent SciPy/HiGHS run. Different optimal primal vectors are allowed when each is feasible and objectives agree.

## UI/API checks

```powershell
$env:NIRNAYA_API_URL='http://127.0.0.1:8000'
python -m pytest apps/nirnaya-ui/tests/e2e -q
node --test apps/nirnaya-ui/tests/unit/test_api_client.js
```

E2E tests explicitly skip if the configured integrated API is offline; they do not count that condition as a pass.
