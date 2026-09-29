> Historical document: this Part 1-6/reference-backend guidance is superseded. The current production contract and architecture are documented in the repository root README.md, docs/DEMO.md, and docs/VALIDATION.md. Use scripts/launch_demo.py to run the current API and UI. The reference backend is archived and is not a production dependency.

# INTEGRATION_CHECKLIST.md â€” Combining All Six Nirnaya Repositories

Audience: whoever does the final integration pass before submission/judging.
This assumes Parts 1â€“5 exist as separate repos (model parser, presolve,
CPU solver, GPU solver, orchestration API) and this repo, `nirnaya-ui-tests`,
is Part 6.

## 0. What "done" looks like

`python3 validation/run_validation_suite.py --base-url <Part-5-URL>` exits
0, and the dashboard's Benchmarks tab shows real CPU (and, if available,
GPU) numbers pulled from a live run against the real Parts 1â€“5 stack â€” not
the reference backend.

## 1. Confirm the contract (30 min)

1. Open `docs/API_CONTRACT.md` in this repo.
2. Confirm Part 5's API implements:
   - `GET /api/health`
   - `GET /api/system/info`
   - `POST /api/model/validate`
   - `POST /api/solve`
   - `GET /api/benchmarks`
3. If Part 5 uses different field names or a different model JSON schema,
   pick ONE of:
   - (preferred) Add a thin adapter layer inside Part 5's API that
     translates to/from the schema in `API_CONTRACT.md`.
   - Edit `src/js/api-client.js` to match Part 5's actual shape (keep
     changes isolated to that one file; `app.js` should not need to change).
4. Confirm Part 5 returns CORS headers (`Access-Control-Allow-Origin`) so
   the browser-hosted UI can call it directly. If Part 5 can't add CORS,
   serve `src/` from the same origin as Part 5, or add a small reverse
   proxy.

## 2. Wire the model schema through Parts 1â€“2 (1â€“2 hrs)

1. Take `demo-data/refinery_blending.json` (or any file in `demo-data/`)
   and feed it into Part 1's parser exactly as Part 5's `/api/solve` would
   receive it.
2. Confirm Part 1 produces the sparse internal representation Part 2/3/4
   expect, and that variable/constraint names round-trip (the UI keys
   solution values by variable **name**, not index).
3. Confirm Part 2's presolve stats (`eliminated_rows`, `eliminated_cols`,
   `time_ms`) are real measurements, not placeholders â€” the Diagnostics
   tab displays them as-is.

## 3. Wire the CPU and GPU solvers (2â€“4 hrs)

1. Point `/api/solve` with `"backend": "cpu"` at Part 3. Confirm the
   response includes real `solve_time_ms`, `iterations`, `objective`,
   `variables`, and (if Part 3 supports it) `constraint_residuals` and
   `iteration_history`.
2. Point `"backend": "gpu"` at Part 4. Two valid outcomes:
   - GPU present: return real `device`, `solve_time_ms`, etc., with
     `backend: "gpu"`.
   - GPU absent: return `status: "error"`, `error: "gpu_unavailable"`, and
     a real `gpu_unavailable_reason` â€” **do not** silently fall back to
     CPU and label it GPU. The UI and validation suite both specifically
     check for this.
3. Run `python3 tests/e2e/test_pipeline.py` against Part 5's real URL. The
   `test_gpu_backend_reports_honestly` case enforces rule 3.2 above.

## 4. Point everything at the real stack (15 min)

1. In the UI top bar, set **API Base URL** to Part 5's real endpoint and
   click **Connect**.
2. Stop using `reference_backend/app.py` for anything except local
   development/offline demoing â€” it is a SciPy stand-in, not part of the
   submission's solver.
3. Re-run:
   ```bash
   python3 validation/run_validation_suite.py --base-url <Part-5-URL>
   python3 benchmarks/run_benchmarks.py --base-url <Part-5-URL> \
     --models-dir demo-data --out benchmarks/results/final_run.json
   python3 benchmarks/generate_report.py "benchmarks/results/final_run.json" \
     --out benchmarks/results/REPORT.md
   ```
4. Open the UI, go to **Benchmarks â†’ Load from API**, and confirm the
   chart/table reflect the real run.

## 5. Sanity checks before the demo

- [ ] `GET /api/system/info` on the real API returns a real CPU model
      string and an honest GPU `available`/`reason`.
- [ ] All 5 validation cases pass (`run_validation_suite.py` exit code 0).
- [ ] At least one demo model in `demo-data/` solves end-to-end through
      the *real* Parts 1â€“5 stack, not just the reference backend.
- [ ] The Solution tab's variable table matches variable names from your
      actual model file (no index-only fallback needed).
- [ ] `DEMO_SCRIPT.md` steps 1â€“9 all work against the real API.
- [ ] If GPU hardware won't be in the room, decide in advance whether to
      demo GPU at all or explicitly show the honest "unavailable" path â€”
      both are legitimate; don't discover it live.

## 6. Known integration edges to watch for

- **Different constraint sense conventions.** Some solvers use `<=` only
  internally and flip `>=`/`=` rows; make sure Part 5's adapter reports
  residuals in the *original* sense so Diagnostics stays interpretable.
- **Maximize vs minimize sign flips.** Confirm Part 5 returns the
  objective in the model's original sense (as documented in
  `API_CONTRACT.md`), not the solver's internal minimized form.
- **Iteration history size.** If Part 3/4 can emit large iteration logs,
  cap what's returned (e.g. last 500 points) â€” the UI charts fine with a
  subsample but a multi-MB JSON response will slow the browser.
- **Large sparse models.** `validation/cases/sparse_large_lp.json` has
  1,440 variables / 2,880 nonzeros â€” use it to catch scaling issues in
  Part 5's JSON serialization before judges try a bigger model live.

