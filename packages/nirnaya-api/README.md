# nirnaya-api

The application/service layer around the Nirnaya solver stack.

`nirnaya-api` does **not** implement any solving logic. It orchestrates:

| Package             | Role                                   |
|----------------------|-----------------------------------------|
| `nirnaya-core`       | Problem model primitives                |
| `nirnaya-presolve`   | Presolve / reduction passes             |
| `nirnaya-solver`     | The actual numerical solver             |
| `nirnaya-gpu`        | Device discovery / GPU backends         |

...and exposes exactly three interfaces on top of them:

1. **A Python API** — build a model, call `.solve()`, inspect a `Solution`.
2. **A CLI** — `nirnaya solve|validate|benchmark|devices|version`.
3. **A REST API** — a FastAPI service with sync and async solve endpoints.

See `docs/INTEGRATION_CONTRACT.md` for exactly what `nirnaya-api` expects from
Parts 1-4, `docs/JSON_FORMAT.md` for the on-the-wire model format, and
`docs/API.md` for the CLI/REST reference.

## Install

```bash
pip install -e .            # Python API + CLI + REST server, without Parts 1-4
pip install -e ".[core]"    # also pulls in nirnaya-core/presolve/solver/gpu
pip install -e ".[dev]"     # test dependencies (pytest, httpx, ...)
```

`nirnaya-api` imports Parts 1-4 lazily: `import nirnaya_api` always works,
and `/health`, `/version`, `/devices`, `nirnaya validate`, and model-building
all work even if those packages are missing. Only an actual solve requires
them, and it fails with a clear `DEPENDENCY_UNAVAILABLE` error (not a crash)
if they're absent.

## Quick start: Python API

```python
from nirnaya_api import Model

m = Model("diet")
m.add_variable("bread", lb=0, ub=10)
m.add_variable("milk", lb=0, ub=10)
m.add_constraint("budget", {"bread": 2, "milk": 3}, sense="<=", rhs=20)
m.set_objective({"bread": 4, "milk": 3}, sense="max")

result = m.solve(
    solver="auto",        # auto | simplex | interior-point | mip
    backend="auto",       # auto | cpu | gpu  (auto falls back to CPU gracefully)
    time_limit=10.0,
    iteration_limit=5_000,
    verbose=False,
)

print(result.status)              # SolveStatus.OPTIMAL
print(result.objective_value)
print(result.variable_values)     # {"bread": ..., "milk": ...}
print(result.backend, result.device, result.solve_time_s)
```

## Quick start: CLI

```bash
$ nirnaya validate examples/model.json
valid: True
variables: 2
constraints: 1

$ nirnaya solve examples/model.json --backend auto --time-limit 5
status:          optimal
objective_value: 33.33...
solver:          auto
backend/device:  cpu/cpu:0
solve_time_s:    0.004123
presolve_time_s: 0.000210
iterations:      6
variables:
  bread = 10.0
  milk = 0.0

$ nirnaya solve examples/model.json --json | jq .status
"optimal"

$ nirnaya devices
cpu:0            kind=cpu available=True  name=CPU (host)

$ nirnaya version
nirnaya-api 0.1.0
json_model_format_version: 1.0
api_schema_version: 1.0
  nirnaya_core: not installed
  nirnaya_presolve: not installed
  nirnaya_solver: not installed
  nirnaya_gpu: not installed
```

## Quick start: REST API

```bash
uvicorn nirnaya_api.server:app --host 0.0.0.0 --port 8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/solve \
  -H 'content-type: application/json' \
  -d @- <<'JSON'
{
  "model": {
    "variables": {"bread": {"lb": 0, "ub": 10}, "milk": {"lb": 0, "ub": 10}},
    "constraints": {"budget": {"coefficients": {"bread": 2, "milk": 3}, "sense": "<=", "rhs": 20}},
    "objective": {"coefficients": {"bread": 4, "milk": 3}, "sense": "max"}
  },
  "options": {"backend": "auto", "time_limit": 5}
}
JSON
```

See `docs/API.md` for the full endpoint reference and error format.

## Design principles

* **One orchestration path.** `nirnaya_api.orchestrator.solve_model` is the
  single function the Python API, CLI, and REST API all call. There is no
  parallel logic per interface.
* **No second solver.** The only place `nirnaya-api` talks to the actual
  solving stack is `nirnaya_api._integration`. A trivial test double
  (`nirnaya_api.testing.FakeSolverBackend`) exists solely so this
  orchestration layer's own test suite doesn't require Parts 1-4 to be
  installed - it is not reachable through any public option and is not an
  alternate way to solve real problems.
* **Graceful GPU->CPU fallback.** `backend="auto"` (the default) prefers a
  GPU device via `nirnaya-gpu` when one is available and healthy, and falls
  back to CPU with a reported warning otherwise. `backend="gpu"` is a strict
  request that raises rather than silently downgrading.
* **Security by default.** Every entry point enforces payload size limits,
  model size limits, and wall-clock limits, and never shells out or
  evaluates user-supplied code. See `docs/API.md#security`.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Unit tests for validation, CLI behavior, REST behavior, security limits,
and backend/device selection run against `FakeSolverBackend` and therefore
do not require Parts 1-4 to be installed. GPU-hardware tests are marked
`@pytest.mark.gpu` and skip automatically when no GPU/`nirnaya-gpu` is
present.

## Repository layout

```
src/nirnaya_api/
  __init__.py        public Python API surface
  model.py            Model / Variable / Constraint / Objective (Python API)
  options.py          SolveOptions
  solution.py         Solution / SolveStatus / NumericalDiagnostics
  schema.py           pydantic schemas: JSON model format + REST bodies
  orchestrator.py     validate -> select backend -> presolve -> solve
  backends.py         device discovery + GPU->CPU fallback
  _integration.py     adapter to nirnaya-core/presolve/solver/gpu
  testing.py          FakeSolverBackend test double (test-only)
  security.py         payload/size/time limits, safe temp files
  jobs.py             in-memory async job manager
  cli.py              `nirnaya` command-line interface
  server.py           FastAPI REST app
  errors.py           structured error hierarchy
  version.py          version constants
docs/
  API.md
  JSON_FORMAT.md
  INTEGRATION_CONTRACT.md
tests/
examples/
  model.json
```
