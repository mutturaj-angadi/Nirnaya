# Benchmark methodology and scorecard

Nirnaya publishes measured outcomes, not an algorithm ranking. A benchmark run is not removed when it times out or fails. The report groups outcomes as optimal, infeasible, unbounded, time limit, numerical failure, or execution error.

## Reproduction

From the workspace root, in an environment with the editable package dependencies installed:

```powershell
python benchmarks/run_benchmark_matrix.py --time-limit 30 --out benchmarks/results/reproducible/latest.json
python benchmarks/build_manifest.py
python benchmarks/audit_datasets.py
```

The matrix runner regenerates the seeded synthetic LP families, then solves the exact serialized model inputs through Nirnaya's production API orchestrator and independently through `scipy.optimize.linprog(method="highs")`. The reference is not imported by the solver or API production path. Each result contains model SHA-256, category/source/license, seed, original and presolved dimensions, status, objective, iterations, phase/factorization/pricing/ratio timings where available, maximum verified violation, CPU backend, reference result, machine information, configuration, and timestamp. The report marks unavailable memory/postsolve sub-timings as null instead of estimating them.

The in-process matrix does not verify listener selection or HTTP serialization. For that, run `python scripts/run_release_validation.py --base-url http://127.0.0.1:8000`; it checks `/health`, then exercises the bounded/feasible/infeasible/unbounded/sparse models, the flagship, and all six refinery scenarios through the configured live API. It records status, objective, original-model violation, solver time, HTTP wall time, iterations, backend, and an independent reference status/objective. Every invocation is archived under `benchmarks/results/release-gate/runs/`; the `latest.json` file is a convenience pointer, not the only retained outcome.

## Fair-comparison limits

Both validators receive the same mathematical model and run in the same process environment on the same machine. Nirnaya's recorded time is the production API/presolve/solver/recovery path; SciPy's time is measured separately and includes its own implementation and default presolve behavior. They do not have a controlled, identical preprocessing stack or solver tolerance semantics. Therefore the report is for correctness and workload characterization, **not a controlled speed comparison**. No superiority claim is made.

The reference implementation currently densifies input matrices and is appropriate only for the included small/medium validation set. The sparse 1,440-variable case is checked by both solvers and its objective and original-model feasibility are recorded. Coordinate-wise primal vectors can differ where the LP has multiple optimal points; reports record the maximum coordinate delta while independently checking each primal's feasibility and objective.

## Scorecard artifacts

- `benchmarks/manifest.json`: dataset catalog plus the latest measured values joined by dataset ID.
- `benchmarks/results/reproducible/latest.json`: full run methodology and every model outcome.
- `benchmarks/results/synthetic/latest.json`: focused seeded family run, with all statuses retained.
- `benchmarks/results/refinery-scenarios/latest.json`: baseline and all documented scenario runs with field-by-field assumption changes.
- `benchmarks/results/sparse-large-lp-production.json`: recorded sparse reference comparison and kernel timing breakdown.
- `benchmarks/results/REPORT.md`: historical archive report; do not confuse it with current-run artifacts above.

Benchmark JSON is regenerated only by actually executing the corresponding runner. A missing solver/reference timing remains null. GPU requests are rejected or recorded as unavailable because LP simplex execution is CPU-only here.

Sparse solves use the CPU and enforce the unchanged 30-second deadline. The exact sparse case completed in the latest sequential source-tree release matrix after the bounded basis-update optimization. Run API validations sequentially on constrained hosts: simultaneous sparse requests can contend and legitimately produce `time_limit`. Such outcomes are retained in the release-gate archive; no timeout is increased or removed from reports. The current timing and archived earlier timeout are both described in [NOC10K.md](NOC10K.md).

## NOC-10K corpus

The separate stratified corpus has its generation, deterministic development/validation/holdout split, reference-label policy, and current evaluation status in [NOC10K.md](NOC10K.md). Reproduce it with:

```powershell
python scripts/generate_noc10k.py
python scripts/run_noc10k.py --workers 1
python scripts/analyze_noc10k.py --input <one-run-jsonl>
```

The 10,000 manifest rows do not imply 10,000 completed solves. Use only the selected run's observed records and retain its timeouts/errors. The runner's default is at most four controlled workers; `--workers 1` is the timing-comparable mode. Holdout metrics are generated separately and must not inform tuning. The production solver remains CPU-only; SciPy/HiGHS appears in the independent labeler only.
