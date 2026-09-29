# API Reference

Covers the REST API (`nirnaya_api.server:app`) and the CLI (`nirnaya`).
Both are thin wrappers over the same orchestration path
(`nirnaya_api.orchestrator.solve_model` / `validate_model`), so their
behavior - status codes aside - is identical.

## Conventions

* All request/response bodies are JSON.
* All errors, from any endpoint, share one envelope (see
  [Error format](#error-format)) - except FastAPI's own request-schema
  validation errors, noted below.
* No endpoint executes shell commands, evaluates user-supplied code, or
  reads/writes files outside the process's own managed temp directory.
* Every CLI command accepts `--json` for machine-readable output; without
  it, output is a concise human-readable summary.

## REST endpoints

### `GET /health`

Liveness and dependency status. Always returns `200`.

```json
{
  "status": "ok",
  "checks": {
    "nirnaya_core": true,
    "nirnaya_presolve": true,
    "nirnaya_solver": true,
    "nirnaya_gpu": false
  }
}
```

`status` is `"degraded"` (not an error) when one or more of Parts 1-4 are
not installed; `/health`, `/version`, `/devices`, and `/validate` all still
work in that state. Only `/solve` and `/jobs` require the missing
dependency and will fail with `DEPENDENCY_UNAVAILABLE` if used.

### `GET /version`

```json
{
  "api_version": "0.1.0",
  "json_model_format_version": "1.0",
  "api_schema_version": "1.0",
  "dependencies": {
    "nirnaya_core": "1.2.0",
    "nirnaya_presolve": "1.2.0",
    "nirnaya_solver": "1.2.0",
    "nirnaya_gpu": null
  }
}
```

`dependencies` reports each package's `__version__` if installed, or
`null` if not importable.

### `GET /devices`

```json
[
  {"id": "cpu:0", "kind": "cpu", "name": "CPU (host)", "available": true, "detail": {}},
  {"id": "gpu:0", "kind": "gpu", "name": "NVIDIA A100", "available": true, "detail": {"memory_gb": 40}}
]
```

Always includes at least one CPU device. GPU devices are only listed when
`nirnaya-gpu` is installed and reports them; if GPU listing itself fails,
a synthetic `gpu:unavailable` entry explains why (never a silent empty
list masquerading as "no GPU hardware exists").

### `POST /validate`

Body: a model object (the same shape as `model` in the solve request - see
`docs/JSON_FORMAT.md`). Never solves; never requires `nirnaya-core` to be
installed for basic shape/reference/size checks.

Request:
```json
{
  "variables": {"x": {"lb": 0, "ub": 5}},
  "objective": {"coefficients": {"x": 1}, "sense": "max"}
}
```

Response (always `200`; check `valid`):
```json
{
  "valid": true,
  "format_version": "1.0",
  "num_variables": 1,
  "num_constraints": 0,
  "issues": []
}
```

### `POST /solve`

Synchronous solve.

Request:
```json
{
  "model": { "...": "see docs/JSON_FORMAT.md" },
  "options": {
    "solver": "auto",
    "backend": "auto",
    "device": null,
    "feasibility_tol": 1e-7,
    "optimality_tol": 1e-7,
    "time_limit": 30.0,
    "iteration_limit": 10000,
    "presolve": true,
    "verbose": false,
    "seed": null
  }
}
```
`options` is entirely optional; every field defaults as shown.

Response (`200`):
```json
{
  "status": "optimal",
  "objective_value": 33.33,
  "variable_values": {"bread": 10.0, "milk": 0.0},
  "observability": {
    "solve_time_s": 0.0041,
    "presolve_time_s": 0.0002,
    "iterations": 6,
    "backend": "cpu",
    "device": "cpu:0",
    "solver": "auto"
  },
  "diagnostics": {
    "max_infeasibility": 0.0,
    "duality_gap": null,
    "conditioning": null,
    "extra": {}
  },
  "warnings": []
}
```

`status` is one of: `optimal`, `infeasible`, `unbounded`, `iteration_limit`,
`time_limit`, `numerical_error`, `error`. A non-`optimal` status is still a
`200` response - it's a valid solver outcome, not an API failure. `extra`
diagnostics are only populated when `options.verbose` is `true`.

### `POST /jobs` (asynchronous solve)

Same request body as `POST /solve`. Returns immediately:

```json
{"job_id": "3fbb...", "status": "queued"}
```
Status code `202 Accepted`.

### `GET /jobs/{job_id}`

```json
{
  "job_id": "3fbb...",
  "status": "completed",
  "result": { "...": "same shape as POST /solve response" },
  "error": null
}
```
`status` is one of `queued`, `running`, `completed`, `failed`. When
`failed`, `error` holds the same structured error object described below
and `result` is `null`. Unknown job ids return `404` with a `JOB_NOT_FOUND`
error.

The reference job manager is in-memory and single-process (see
`nirnaya_api/jobs.py`); this document describes the contract, not that
implementation detail - a production deployment may swap in a durable
queue behind the same endpoints.

## Error format

Every error response - regardless of endpoint - has this shape:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Model failed schema validation.",
    "details": {"issues": ["objective.coefficients.z: unknown variable"]}
  }
}
```

| HTTP status | `code`                    | Meaning                                             |
|-------------|---------------------------|------------------------------------------------------|
| 400         | `VALIDATION_ERROR`        | Malformed/inconsistent options or business-level input |
| 413         | `PAYLOAD_TOO_LARGE`       | Request body exceeded the configured size limit       |
| 422         | `RESOURCE_LIMIT_EXCEEDED` | Model exceeds variable/constraint/nonzero limits      |
| 404         | `NOT_FOUND` / `JOB_NOT_FOUND` | No such resource / job                            |
| 409         | `DEVICE_UNAVAILABLE`      | An explicitly-requested device/backend doesn't exist  |
| 503         | `DEPENDENCY_UNAVAILABLE`  | A required Part 1-4 package isn't installed           |
| 500         | `INTERNAL_ERROR`          | Unexpected internal error (never leaks internals)     |

**Exception:** malformed request *bodies* that fail FastAPI's own pydantic
schema validation (e.g. `POST /solve` with `model` missing entirely, or a
field of the wrong type) are rejected by FastAPI before reaching
`nirnaya-api`'s own error handling, and use FastAPI's standard `422`
response shape (`{"detail": [...]}`) rather than the envelope above. Once a
request is well-formed enough to be dispatched to a handler, every error is
the structured envelope.

`SolveTimeoutError` is the one exception that is not surfaced as an HTTP
error: a solve that hits the hard wall-clock limit returns a normal `200`
`/solve` response with `status: "time_limit"` and a warning explaining
that the limit was exceeded - a timeout is a legitimate solver outcome,
not an API failure.

## Security

* **Payload limits** - request bodies over `max_payload_bytes` (default
  5 MiB) are rejected with `413` before JSON parsing occurs.
* **Model size limits** - variable count, constraint count, and total
  nonzero coefficients are each capped (defaults: 200,000 / 200,000 /
  5,000,000) and rejected with `422` `RESOURCE_LIMIT_EXCEEDED`.
* **Time limits** - a caller's requested `time_limit` is clamped to
  `max_time_limit_s` (default 600s), and every solve is additionally
  wrapped in a hard wall-clock ceiling (`hard_wall_clock_s`, default 900s)
  enforced by `nirnaya-api` itself, independent of the solver's own
  handling.
* **Resource limits** - the async job queue caps the number of pending
  jobs (default 100) and runs jobs in a bounded thread pool (default 4
  workers), rejecting further submissions with `422` when full.
* **Safe temporary files** - any temp file `nirnaya-api` creates uses a
  securely-generated name, `0600` permissions, and is always removed, even
  on error (`nirnaya_api.security.safe_temp_file`).
* **No shell execution** - the CLI and REST layers only ever call Python
  functions in-process; no subprocess or shell invocation occurs anywhere
  in this package, and model `options` are passed through as inert data,
  never evaluated.

These defaults live in `nirnaya_api.security.SecurityLimits` and can be
overridden by constructing your own `SecurityLimits` and passing it to
`solve_model`/`validate_model` (Python API), or by adjusting
`nirnaya_api.server._limits` before mounting the app in your own
deployment.

## CLI reference

```
nirnaya solve MODEL_PATH [options]
nirnaya validate MODEL_PATH
nirnaya benchmark [--sizes 10,100,1000] [--backend auto]
nirnaya devices
nirnaya version
```

All commands accept `--json`. `solve` accepts the same tuning knobs as the
Python API (`--solver`, `--backend`, `--device`, `--time-limit`,
`--iteration-limit`, `--feasibility-tol`, `--optimality-tol`,
`--no-presolve`, `--verbose`, `--seed`).

Exit codes:
* `0` - success (`solve`: status is `optimal`; `validate`: model is valid)
* `1` - a `NirnayaAPIError` occurred (validation, resource limit, etc.);
  printed as `error: [CODE] message` or, with `--json`, the same error
  envelope as the REST API, on stderr.
* `2` - `solve` completed without error but the result status was not
  `optimal` (infeasible, unbounded, time/iteration limit, etc.)
