> Historical document: this Part 1-6/reference-backend guidance is superseded. The current production contract and architecture are documented in the repository root README.md, docs/DEMO.md, and docs/VALIDATION.md. Use scripts/launch_demo.py to run the current API and UI. The reference backend is archived and is not a production dependency.

# ARCHITECTURE.md â€” nirnaya-ui-tests (Part 6)

## Position in the Nirnaya stack

```
 Part 1          Part 2         Part 3 / 4          Part 5           Part 6 (this repo)
 model parser -> presolve  ->  CPU/GPU solve  ->  orchestration  -> UI / demo / validation
                                                    API (HTTP)        / benchmarks
```

Part 6 never solves anything itself. Every optimization result the UI
displays came from an HTTP response from whatever is running at the
configured **API Base URL** â€” either a real Parts 1â€“5 deployment, or
`reference_backend/app.py`, a small SciPy-backed stand-in included here so
the UI, validation suite, and benchmark harness are runnable and honestly
demoable on day one.

## Why a reference backend exists

Parts 1â€“5 are separate repositories built in parallel. `nirnaya-ui-tests`
needs to be independently testable and demoable without waiting on that
integration. `reference_backend/app.py` implements the exact contract in
`API_CONTRACT.md` using `scipy.optimize.linprog(method="highs")` for CPU,
and honestly reports GPU as unavailable when no CUDA device is detected
(rather than faking GPU numbers). Swapping it for the real Part 5 API is a
one-line change (the Base URL field) â€” see `INTEGRATION_CHECKLIST.md`.

## Data flow inside this repo

```
demo-data/*.json  ----------------------------> src/js/demo-data.js (embedded)
      |                                                 |
      v                                                 v
validation/reference_solver.py            src/index.html (Model tab)
(SciPy/HiGHS, independent of Nirnaya)                    |
      |                                                  v
      v                                          src/js/app.js -> api-client.js -> POST /api/solve
validation/run_validation_suite.py  <--- compares --------------------- live API response
      |
      v
validation/validation_report.json

benchmarks/run_benchmarks.py --- calls live API ---> benchmarks/results/*.json
      |
      v
benchmarks/generate_report.py -> benchmarks/results/REPORT.md
```

`src/js/demo-data.js` is generated (not hand-written) by
`scripts/build_demo_data_js.py` from the JSON files in `demo-data/`, so the
Model tab works without a local file server (useful when opening
`index.html` directly via `file://`).

## Frontend

Plain HTML/CSS/JS, no build step, no framework â€” deliberately, so a judge
can open `src/index.html` on any machine with no `npm install`. Chart.js
is loaded from a CDN for the standard charts (residuals, variable
distribution, objective progress, CPU vs GPU); the sparse-matrix spy plot
is hand-drawn on a `<canvas>` since it needs precise per-cell control that
a generic chart type doesn't offer.

- `api-client.js` â€” the only file that knows the HTTP contract.
- `charts.js` â€” the only file that touches Chart.js / canvas drawing.
- `app.js` â€” application state, view switching, DOM rendering. Every
  render function accepts the possibility that a field is `null` and
  renders "unavailable" rather than guessing.
- `demo-data.js` â€” generated data, not logic.

## Honesty guarantees enforced structurally

1. **No client-side solving.** There is no LP solver in `src/js/` â€” every
   number on Solution/Diagnostics/GPU/Benchmarks came from a fetch.
2. **GPU unavailability is a first-class response shape**, not an
   afterthought: `status: "error"`, `error: "gpu_unavailable"`,
   `gpu_unavailable_reason: "<real reason>"`. The UI, the e2e test suite,
   and the benchmark recorder all check for and handle this shape
   explicitly (see `test_gpu_backend_reports_honestly` in
   `tests/e2e/test_pipeline.py`).
3. **Every demo model has a computed reference solution** in
   `demo-data/reference/`, produced by running `reference_solver.py`
   against it â€” not hand-typed.
4. **Benchmarks are only ever read from `benchmarks/results/*.json`**,
   files produced by actually calling a live API. Nothing in
   `reference_backend/app.py` or `src/js/` invents a benchmark number.

