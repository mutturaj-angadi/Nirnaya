# Integration Contract: nirnaya-api <-> Parts 1-4

`nirnaya-api` (Part 5) implements no solving logic. Everything it needs
from `nirnaya-core`, `nirnaya-presolve`, `nirnaya-solver`, and `nirnaya-gpu`
is defined here and adapted in exactly one file:
`nirnaya_api/_integration.py`, behind the `SolverBackend` protocol. If any
of Parts 1-4 changes its public API, this is the only file in `nirnaya-api`
that should need to change.

This document is the authoritative description of what nirnaya-api calls
and expects back. `RealSolverBackend` in `_integration.py` is the reference
adapter implementing it; it lazily imports each package so `import
nirnaya_api` never fails just because the solver stack isn't installed.

## Why an adapter layer, not direct calls everywhere

* A single seam makes it possible to test the orchestration layer
  (validation, security limits, options plumbing, observability,
  CLI/REST behavior) without requiring Parts 1-4 to be installed -
  see `nirnaya_api.testing.FakeSolverBackend`, used only by this
  package's own test suite via explicit dependency injection
  (`nirnaya_api._integration.set_backend`), never in production code paths.
* It keeps "what Parts 1-4 look like" separate from "how nirnaya-api
  presents itself", so either side can evolve independently.

## 1. `nirnaya-gpu`: device discovery

**Expected call:** `nirnaya_gpu.list_devices() -> list[Device]`

Each `Device` is expected to expose (via attribute access):

| Attribute    | Type   | Required | Notes                              |
|---------------|--------|----------|--------------------------------------|
| `id`          | str    | recommended | e.g. `"gpu:0"`; falls back to `f"gpu:{index}"` if absent |
| `name`        | str    | no       | Human-readable device name           |
| `available`   | bool   | no       | Defaults to `True` if absent         |
| `detail`      | dict   | no       | Free-form extra info (memory, compute capability, ...) |

**Contract:**
* If `nirnaya-gpu` is not installed, `nirnaya-api` treats this as "no GPU
  devices" (not an error) - CPU-only operation is always fully supported.
* If `nirnaya_gpu.list_devices()` raises, `nirnaya-api` surfaces that as a
  reported reason (a synthetic `gpu:unavailable` entry / warning), not a
  crash, and `backend="auto"` still falls back to CPU cleanly.
* `nirnaya-api` never assumes device IDs, counts, or naming beyond the
  table above.

## 2. `nirnaya-core`: model construction

**Expected call:** `nirnaya_core.Model.from_dict(model_dict) -> CoreModel`

* `model_dict` is the plain-dict form of the JSON model format (see
  `docs/JSON_FORMAT.md`), already schema-validated by `nirnaya-api`
  (non-empty variables, valid cross-references, `lb <= ub`, etc.) before
  this call - `nirnaya-core` does not need to re-validate wire-format
  shape, only build a solver-ready internal representation from it.
* On semantic construction failure (e.g. numerically degenerate input),
  `nirnaya-core` is expected to raise `nirnaya_core.errors.ModelError`
  (or a subclass); `nirnaya-api` translates this into its own
  `ValidationError` rather than letting it propagate as an internal error.
* The returned `CoreModel` is treated by `nirnaya-api` as an opaque object
  - it is only ever passed to `nirnaya-presolve` and `nirnaya-solver`, never
  introspected.

## 3. `nirnaya-presolve`: presolve pass

**Expected call:**
```python
nirnaya_presolve.presolve(core_model, feasibility_tol=1e-7) -> (reduced_model, info)
```

* `reduced_model` is another opaque object of the same kind `nirnaya-solver`
  accepts (may be the same object, unmodified, if no reductions apply).
* `info` is a dict; `nirnaya-api` reads `info["presolve_time_s"]` if
  present, and otherwise times the call itself and fills that key in. Any
  other keys are ignored by `nirnaya-api` today (reserved for future
  observability fields, e.g. `reductions_applied`).
* Only called when `SolveOptions.presolve` is `True` (the default).

## 4. `nirnaya-solver`: solving

**Expected call:**
```python
nirnaya_solver.solve(
    core_model,
    backend="cpu" | "gpu",
    device="cpu:0" | "gpu:0" | ...,
    solver="auto" | "simplex" | "interior-point" | "mip",
    feasibility_tol=1e-7,
    optimality_tol=1e-7,
    time_limit=30.0 | None,
    iteration_limit=10000 | None,
    verbose=False,
    seed=None | int,
) -> dict
```

`backend` and `device` are always concrete, already-resolved values by the
time `nirnaya-api` calls this - `nirnaya-solver` never needs to perform its
own device fallback; that is `nirnaya-api`'s job
(`nirnaya_api.backends.select_backend`).

**Expected return dict:**

| Key                 | Type            | Required | Notes                                                     |
|----------------------|-----------------|----------|--------------------------------------------------------------|
| `status`             | str             | **yes**  | One of: `optimal`, `infeasible`, `unbounded`, `iteration_limit`, `time_limit`, `numerical_error`, `error` |
| `objective_value`    | float or null   | **yes**  | `null` unless `status == "optimal"` (or otherwise meaningful) |
| `variable_values`    | dict[str,float] | **yes**  | Empty dict when no solution point exists                     |
| `iterations`         | int             | **yes**  |                                                                |
| `solve_time_s`       | float           | **yes**  | Time spent inside the solver itself                           |
| `diagnostics`        | dict            | no       | See below                                                     |

`diagnostics` (all optional):

| Key                 | Type    | Notes                                             |
|----------------------|---------|-----------------------------------------------------|
| `max_infeasibility`  | float   | Largest constraint/bound violation at the returned point |
| `duality_gap`        | float   | If applicable to the algorithm family used            |
| `conditioning`       | float   | Condition-number style indicator, if available        |
| `extra`              | dict    | Free-form; only surfaced to callers when `verbose=True` |

**Contract:**
* `nirnaya-solver` is expected to respect `time_limit` and
  `iteration_limit` itself where possible (returning `status:
  "time_limit"` / `"iteration_limit"` rather than running indefinitely).
  `nirnaya-api` additionally wraps every call in its own hard wall-clock
  ceiling (`SecurityLimits.hard_wall_clock_s`, default 900s) as a
  last-resort safety net - see `docs/API.md#security` - which is not a
  substitute for cooperative time-limit handling inside the solver.
* Any exception other than a clean status return is treated by
  `nirnaya-api` as an unexpected internal error (never leaked verbatim to
  API/CLI callers).

## Versioning

* `nirnaya_api.version.JSON_MODEL_FORMAT_VERSION` tracks the JSON model
  format (`docs/JSON_FORMAT.md`), independent of package versions.
* This contract document is not currently machine-checked against
  installed package versions; `nirnaya version` / `GET /version` report
  each package's own `__version__` so operators can correlate manually.
  If Parts 1-4 adopt a formal contract-version marker in the future,
  `RealSolverBackend` is where a compatibility check would be added.

## What is explicitly out of scope for this contract

* `nirnaya-api` never inspects or depends on internal data structures of
  `CoreModel` / the reduced model - they are opaque, pass-through objects.
* `nirnaya-api` never implements alternative solving logic of its own in
  production code paths; see `nirnaya_api/testing.py` for the one narrow,
  explicitly test-only exception, which is not part of this contract and
  is not selectable through any public API, CLI flag, or REST parameter.
