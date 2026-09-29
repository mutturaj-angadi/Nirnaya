> Historical document: this Part 1-6/reference-backend guidance is superseded. The current production contract and architecture are documented in the repository root README.md, docs/DEMO.md, and docs/VALIDATION.md. Use scripts/launch_demo.py to run the current API and UI. The reference backend is archived and is not a production dependency.

# Nirnaya API Contract (consumed by Part 6 â€” `nirnaya-ui-tests`)

This document is the interface boundary between the UI/validation/benchmark
layer (Part 6) and the pipeline implemented by Parts 1â€“5
(model parsing â†’ presolve â†’ CPU solve â†’ GPU solve â†’ orchestration API).
Part 6 never re-implements solver logic; it only talks HTTP to whatever
process serves this contract.

If your Parts 1â€“5 stack exposes a different shape, adapt the thin client in
`src/js/api-client.js` (or point `reference_backend/app.py` at your real
solver code) â€” the rest of the UI does not need to change.

## Base URL

Configurable in the UI ("API Base URL" field, default `http://localhost:8000`).
`reference_backend/app.py` implements this exact contract using SciPy so the
UI is fully demoable even before Parts 1â€“5 are wired in.

## Endpoints

### `GET /api/system/info`
Returns real host information â€” never fabricated. Fields the UI reads:

```json
{
  "cpu": {
    "model": "string or null",
    "physical_cores": 8,
    "logical_cores": 16,
    "available": true
  },
  "gpu": {
    "available": false,
    "device_name": null,
    "reason": "no CUDA-capable device detected / cupy not installed"
  },
  "backend_versions": {"nirnaya_core": "x.y.z", "nirnaya_cpu_solver": "x.y.z", "nirnaya_gpu_solver": "x.y.z"}
}
```
If a field cannot be determined, it MUST be `null` (or `available: false`
with a `reason`) â€” never a guessed value.

### `POST /api/model/validate`
Body: a Nirnaya model JSON (schema below). Returns parse/presolve stats from
Part 1/Part 2 without solving:

```json
{
  "valid": true,
  "num_variables": 8,
  "num_constraints": 9,
  "nonzeros": 41,
  "presolve": {"eliminated_rows": 0, "eliminated_cols": 0, "time_ms": 1.2}
}
```

### `POST /api/solve`
Body:
```json
{
  "model": { "...": "Nirnaya model JSON, see docs/API_CONTRACT.md#model-schema" },
  "backend": "cpu | gpu",
  "options": {
    "max_iterations": 5000,
    "tolerance": 1e-8,
    "time_limit_s": 60,
    "presolve": true,
    "record_iteration_history": true
  }
}
```
Response (all numeric fields are measured, not estimated):
```json
{
  "status": "optimal | infeasible | unbounded | iteration_limit | error",
  "objective": 636.0,
  "variables": {"reformate_reg": 12.3, "...": "..."},
  "iterations": 14,
  "solve_time_ms": 3.8,
  "presolve_time_ms": 0.6,
  "backend": "cpu",
  "device": "Intel(R) Xeon(R) ... / or GPU device string",
  "iteration_history": [{"iter": 0, "objective": 0.0, "residual": 1.0}, "..."],
  "constraint_residuals": {"c1": 0.0, "c2": 1.2e-9},
  "numerical": {"max_residual": 1.2e-9, "duality_gap": 3.1e-10, "condition_estimate": null}
}
```
If a field is not produced by the underlying solver (e.g. GPU is
unavailable, or a solver doesn't report a duality gap), the server MUST
return `null` for it and the UI will render "unavailable" â€” it will not
invent a value.

### `GET /api/solve/{id}/status`
For long-running/async solves. Returns the same shape as `/api/solve` with a
`state: "queued|running|done"` field while running.

### `GET /api/benchmarks`
Returns the contents of `benchmarks/results/*.json` actually produced by
`benchmarks/run_benchmarks.py` â€” never synthesized on the fly.

```json
{
  "runs": [
    {
      "problem": "sparse_large_lp", "num_variables": 450, "num_constraints": 99,
      "nonzeros": 900, "cpu_time_ms": 41.2, "gpu_time_ms": null,
      "gpu_unavailable_reason": "no CUDA device detected",
      "iterations": 37, "objective": 10313.471, "max_residual": 7.1e-15,
      "timestamp": "2026-09-27T10:15:00Z"
    }
  ]
}
```

## Model schema

```json
{
  "name": "string",
  "sense": "minimize | maximize",
  "variables": [{"name": "x1", "lower": 0, "upper": null, "type": "continuous"}],
  "objective": {"offset": 0, "coefficients": {"x1": 3.0}},
  "constraints": [{"name": "c1", "sense": "<= | >= | =", "rhs": 10, "coefficients": {"x1": 1.0}}]
}
```

This is the schema used throughout `demo-data/` and `validation/cases/`.
Parts 1â€“2 are expected to accept this JSON (or a documented transform of it)
and produce the sparse internal representation consumed by Parts 3â€“4.

