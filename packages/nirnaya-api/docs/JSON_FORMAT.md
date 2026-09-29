# JSON Model Format

Version: `1.0` (`nirnaya_api.version.JSON_MODEL_FORMAT_VERSION`)

This is the single on-the-wire representation of an optimization model used
by the Python API (`Model.to_dict()` / `Model.from_dict()`), the CLI
(`nirnaya solve model.json`), and the REST API (`POST /solve`'s `model`
field, `POST /validate`'s body). It is validated by
`nirnaya_api.schema.ModelSchema`.

## Shape

```json
{
  "format_version": "1.0",
  "name": "diet-problem",
  "variables": {
    "<variable name>": {
      "lb": 0.0,
      "ub": 10.0,
      "kind": "continuous"
    }
  },
  "constraints": {
    "<constraint name>": {
      "coefficients": { "<variable name>": 1.0 },
      "sense": "<=",
      "rhs": 20.0
    }
  },
  "objective": {
    "coefficients": { "<variable name>": 1.0 },
    "sense": "min",
    "offset": 0.0
  },
  "options": {}
}
```

## Fields

### Top level

| Field             | Type   | Required | Notes                                                        |
|--------------------|--------|----------|----------------------------------------------------------------|
| `format_version`   | string | no       | Defaults to the current format version; informational only.    |
| `name`             | string | no       | Free-form model name.                                           |
| `variables`        | object | **yes**  | Must be non-empty.                                              |
| `constraints`      | object | no       | Defaults to `{}`.                                                |
| `objective`        | object | **yes**  |                                                                  |
| `options`          | object | no       | Reserved for solver-specific hints; passed through as inert data, never interpreted or executed by `nirnaya-api` itself. |

### Variable

| Field  | Type   | Default    | Notes                                                    |
|--------|--------|------------|------------------------------------------------------------|
| `lb`   | number | `0.0`      |                                                              |
| `ub`   | number | `+Infinity`| Must be `>= lb`.                                             |
| `kind` | string | `"continuous"` | One of `continuous`, `integer`, `binary` (for MIP models). |

### Constraint

| Field           | Type              | Required | Notes                                             |
|------------------|-------------------|----------|-----------------------------------------------------|
| `coefficients`   | object (name -> number) | **yes** | Must be non-empty; every key must name a declared variable. |
| `sense`          | string            | **yes**  | One of `<=`, `>=`, `=`.                              |
| `rhs`            | number            | **yes**  |                                                       |

Represents: `sum(coefficients[v] * v for v in coefficients) <sense> rhs`.

### Objective

| Field           | Type              | Default | Notes                                        |
|------------------|-------------------|---------|-------------------------------------------------|
| `coefficients`   | object (name -> number) | `{}` | Every key must name a declared variable.       |
| `sense`          | string            | `"min"` | `min` or `max`.                                 |
| `offset`         | number            | `0.0`   | Constant added to the reported objective value. |

## Validation rules (enforced before any solve)

1. `variables` must be non-empty.
2. Every coefficient key in every constraint and in the objective must
   name a variable declared in `variables`.
3. For every variable, `lb <= ub`.
4. Every constraint's `coefficients` must be non-empty.
5. Model size (variable count, constraint count, total nonzero
   coefficients) must stay within the configured
   `nirnaya_api.security.SecurityLimits`.

Violating (1)-(4) produces a `VALIDATION_ERROR`; violating (5) produces a
`RESOURCE_LIMIT_EXCEEDED` error. `nirnaya validate` / `POST /validate`
report all such issues without ever invoking a solver.

## Example

See `examples/model.json`:

```json
{
  "format_version": "1.0",
  "name": "diet-problem",
  "variables": {
    "bread": { "lb": 0.0, "ub": 10.0, "kind": "continuous" },
    "milk":  { "lb": 0.0, "ub": 10.0, "kind": "continuous" }
  },
  "constraints": {
    "budget": {
      "coefficients": { "bread": 2.0, "milk": 3.0 },
      "sense": "<=",
      "rhs": 20.0
    }
  },
  "objective": {
    "coefficients": { "bread": 4.0, "milk": 3.0 },
    "sense": "max",
    "offset": 0.0
  },
  "options": {}
}
```

## Solution format

A solved model produces a `Solution` (Python API) / the `POST /solve`
response body (REST) / `nirnaya solve --json` output (CLI) - all the same
shape, defined by `nirnaya_api.solution.Solution`:

```json
{
  "status": "optimal",
  "objective_value": 33.33,
  "variable_values": { "bread": 10.0, "milk": 0.0 },
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
`time_limit`, `numerical_error`, `error`. `diagnostics.extra` is only
populated when the solve was run with `verbose: true`.

## Notes on numeric edge cases

* An unspecified `ub` defaults to positive infinity. Python's own
  `json.dumps`/`json.loads` (used internally) round-trip this as the
  non-standard `Infinity` token; if you need strictly RFC 8259-compliant
  JSON for interop with another tool, give every variable an explicit,
  large finite `ub` instead of omitting it.
* `format_version` and `options` are accepted but `options` is not
  currently interpreted by `nirnaya-api` itself - it exists as a
  forward-compatible place for solver-specific hints once Parts 1-4 define
  them, and is never treated as executable in any form.

## Compatibility

`format_version` is bumped only on backward-incompatible changes to this
document. `nirnaya-api` does not currently reject a model based on its
declared `format_version` (it always validates against the current
`ModelSchema`); the field is informational for now and is reserved for
stricter version negotiation later.
