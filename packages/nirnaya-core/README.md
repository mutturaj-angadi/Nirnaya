# Nirnaya Core

**Part 1 of Nirnaya** — an indigenous, GPU-accelerated mathematical
optimization solver being developed for **SIH 2026 Problem Statement
26119** (Mangalore Refinery and Petrochemicals Limited, MRPL).

`nirnaya-core` is the mathematical foundation and the stable interface
layer that Parts 2–6 (presolve, the simplex/interior-point engine, the
GPU/native backend, and reporting) are built on top of. **It ships no
solver.** What it ships is a correct, well-tested, strongly-typed
representation of a linear program, plus the data structures a solver's
output must conform to.

---

## 1. What Nirnaya Core does

Nirnaya Core lets you:

1. **Build** a linear program with a small, ergonomic, name-addressed API
   (`Model.add_variable`, `Model.add_constraint`, `Model.set_objective`).
2. **Validate** that model exhaustively — every structural problem is
   caught and reported together, with a stable machine-readable `code` per
   problem, before anything touches a solver.
3. **Compile** the validated model into `ProblemData`: an immutable,
   purely numerical standard-form snapshot (dense objective/bound vectors,
   sparse constraint matrices, explicit dtype) that has no further
   dependency on the mutable `Model` object graph.
4. **Serialize** either the model or the compiled `ProblemData` to
   deterministic JSON, and deserialize it back losslessly (including
   `+inf`/`-inf` bounds).
5. **Describe** what a solver's output looks like (`SolverResult`,
   `SolverStatus`, `OptimizationStatistics`) so that every future backend
   — a reference CPU simplex, a GPU interior-point method, a native
   C++/CUDA/HIP kernel — produces the same shape of answer.

It does **not** solve anything. There is no simplex tableau, no
interior-point iteration, no GPU kernel here. That is deliberate: Part 1's
job is to make Parts 2–6 unable to disagree about what a "variable," a
"constraint," a "sparse matrix," or a "solver result" is.

---

## 2. Architecture

```
nirnaya-core/
├── src/nirnaya_core/
│   ├── model/        # Variable, Constraint, LinearObjective, Model, ProblemData, enums
│   ├── matrix/        # SparseMatrix (CSR-backed, deterministic serialization)
│   ├── numeric/        # NumericalConfig — explicit tolerances & dtype policy
│   ├── solution/       # SolverResult, OptimizationStatistics
│   ├── status/         # SolverStatus enum
│   ├── validation/     # Validation error types + validate_problem_data()
│   ├── io/             # Deterministic JSON (de)serialization
│   └── exceptions/     # Machine-readable exception hierarchy (NirnayaError)
├── tests/               # ~150 unit tests
├── examples/            # Runnable example scripts
└── docs/
    └── INTEGRATION_CONTRACT.md   # What Parts 2-6 may depend on
```

### Layering

```
        Model  (mutable, name-addressed, ergonomic)
          │  .build()  [validates, then compiles]
          ▼
   ProblemData  (immutable, numerical, standard form)
          │  handed to a solver backend (Part 2+)
          ▼
   SolverResult  (status + primal/dual/reduced costs/residuals/statistics)
```

* **`Model`** is where you *describe* a problem. It is mutable, uses
  string names, and validates eagerly at the component level (a
  `Variable` can never have `lower > upper`; a `Constraint` can never have
  a non-finite coefficient) and exhaustively at the model level
  (`Model.validate()` / `Model.build()`).
* **`ProblemData`** is where a solver *reads* a problem. It is immutable
  (arrays are marked read-only after construction), index-addressed, and
  contains nothing but numbers — no back-reference to the `Model` that
  produced it. This is intentional: a solver backend should never need to
  import the model-builder layer.
* **`SolverResult`** is where a solver *writes* an answer. Its arrays are
  index-aligned to `ProblemData.variable_names` /
  `constraint_names_eq + constraint_names_ub`, so a caller can always map
  a result back to named quantities.

### Why this split?

A later native/GPU backend (C++/CUDA/HIP) needs a numerical contract that
does not depend on Python object identity, mutability, or garbage
collection. `ProblemData` is designed so its fields — dense numpy arrays
plus one sparse matrix type with a documented COO/CSR shape — map
directly onto what a native kernel would want as input buffers, without
having to reverse-engineer that shape out of a mutable builder API.

---

## 3. Mathematical representation

Nirnaya Core represents a **linear program** in the form:

```
minimize / maximize      c^T x + objective_constant

subject to                A_eq x  = b_eq
                           A_ub x <= b_ub
                           l <= x <= u
```

* `x` is the vector of decision variables (`Variable`), in the order they
  were added to the `Model`.
* `c` is the (dense) objective coefficient vector; `objective_constant`
  is folded in additively and does not affect the optimal `x`.
* `A_eq` / `b_eq` hold every constraint declared with
  `ConstraintSense.EQ`.
* `A_ub` / `b_ub` hold every constraint declared with `ConstraintSense.LE`
  directly, and every `ConstraintSense.GE` constraint **normalized** by
  negation: `a^T x >= b` is stored as `-a^T x <= -b`. This means a solver
  backend only ever has to implement one inequality direction.
* `l`, `u` are dense per-variable bound vectors. `-inf` / `+inf` (or any
  magnitude at or beyond `NumericalConfig.infinity`) means "no bound" —
  see [§5](#5-numerical-philosophy).
* All matrices are stored **sparse** (`SparseMatrix`, backed by
  `scipy.sparse.csr_matrix` internally), never dense, unless a caller
  explicitly asks for a dense view via `.to_dense()`.

`Constraint.residual(values)` and `LinearObjective.evaluate(values)` let
you evaluate `a^T x - rhs` and `c^T x + constant` respectively against a
`{variable_name: value}` mapping — useful for sanity-checking a candidate
point, or for a solver backend to self-check its own answer before
returning a `SolverResult`.

---

## 4. Public APIs

The full contract is documented in
[`docs/INTEGRATION_CONTRACT.md`](docs/INTEGRATION_CONTRACT.md). Summary:

| Type | Module | Role |
|---|---|---|
| `Model` | `nirnaya_core.model` | Mutable LP builder |
| `Variable` | `nirnaya_core.model` | A decision variable `l <= x <= u` |
| `Constraint` | `nirnaya_core.model` | A row `a^T x {<=,==,>=} rhs` |
| `LinearObjective` | `nirnaya_core.model` | `sense  c^T x + constant` |
| `ProblemData` | `nirnaya_core.model` | Immutable standard-form snapshot |
| `VariableType` | `nirnaya_core.model` | `CONTINUOUS` / `INTEGER` / `BINARY` |
| `ConstraintSense` | `nirnaya_core.model` | `LE` / `EQ` / `GE` |
| `ObjectiveSense` | `nirnaya_core.model` | `MINIMIZE` / `MAXIMIZE` |
| `SparseMatrix` | `nirnaya_core.matrix` | CSR-backed sparse matrix, deterministic (de)serialization |
| `NumericalConfig` | `nirnaya_core.numeric` | Explicit tolerances, dtype, iteration/time limits |
| `SolverStatus` | `nirnaya_core.status` | Canonical solve outcomes |
| `SolverResult` | `nirnaya_core.solution` | Full solve output (primal/dual/reduced costs/residuals/stats) |
| `OptimizationStatistics` | `nirnaya_core.solution` | Iteration count, timing, gaps, infeasibilities |
| `NirnayaError` and subclasses | `nirnaya_core.exceptions` / `nirnaya_core.validation` | Machine-readable errors with stable `.code` |
| `nirnaya_core.io.*` | `nirnaya_core.io` | Deterministic JSON (de)serialization |

Every public dataclass exposes `to_dict()` / `from_dict()`, and the model
and result types additionally have `*_to_json()` / `*_from_json()`
helpers in `nirnaya_core.io`.

---

## 5. Numerical philosophy

* **No silent dense fallback.** `SparseMatrix` is the only matrix type in
  the public API; converting to dense (`.to_dense()`) is always explicit
  and is intended for small models, debugging, and tests — never used
  internally as a default path.
* **No silent dtype narrowing.** `NumericalConfig.dtype` is the single
  source of truth for the floating-point precision of an entire
  `ProblemData` snapshot (`float64` by default; `float32` supported for
  future GPU paths). Every array in a `ProblemData` is constructed at
  that dtype explicitly; nothing implicitly upcasts or downcasts along
  the way.
* **NaN is never valid data.** Every numerical field — objective and
  constraint coefficients, RHS values, the objective constant — rejects
  `NaN` and non-finite values at construction time, in the component's
  own `__post_init__`. Variable **bounds** are the one deliberate
  exception: `+inf` / `-inf` are meaningful there ("no bound").
  `NumericalConfig.infinity` (default `1e20`) additionally lets a solver
  backend treat any sufficiently large finite bound as effectively
  unbounded, via `NumericalConfig.is_effectively_infinite(value)` —
  useful for kernels that cannot represent IEEE infinities.
  `NumericalConfig.is_effectively_infinite` and `SparseMatrix`/`Variable`/
  `Constraint`/`LinearObjective` validation together mean a
  `ProblemData` snapshot can never silently contain a `NaN`.
* **Tolerances are explicit, never magic numbers.** `NumericalConfig`
  bundles `feasibility_tol`, `optimality_tol`, `bound_tol`, `pivot_tol`,
  `zero_tol`, `infinity`, `max_iterations`, and `time_limit_seconds` into
  one immutable, validated, serializable object. A solver backend should
  accept a `NumericalConfig` rather than inventing its own defaults.
* **Immutability at the boundary.** `ProblemData` freezes its numpy
  arrays (`array.setflags(write=False)`) after construction, so a solver
  cannot accidentally mutate the snapshot it was handed. `SparseMatrix`
  and every model component (`Variable`, `Constraint`, `LinearObjective`)
  are frozen dataclasses.
* **Deterministic by construction.** Sparse matrix serialization always
  emits COO triplets sorted by `(row, col)`; JSON serialization always
  sorts keys. The same logical model always serializes to the same
  bytes, which matters for hashing, diffing, and caching solver runs.
* **Errors are structured, not stringly-typed.** Every validation failure
  is a typed exception with a stable `.code` (e.g. `"E_DIM_MISMATCH"`,
  `"E_INVALID_BOUNDS"`) and a JSON-serializable `.details` payload, and
  `Model.build()` never returns a partially-valid result — it raises a
  single aggregated `ModelValidationError` listing every problem found in
  one pass.

---

## 6. How Parts 2–6 should integrate with this module

1. **Depend on `ProblemData`, not on `Model` internals.** A solver
   backend's entry point should be a function/class that accepts a
   `ProblemData` (plus, optionally, its own solver-specific options) and
   returns a `SolverResult`. Building the `Model` and calling `.build()`
   is the caller's job, not the solver's.
2. **Accept a `NumericalConfig`.** `ProblemData.config` carries the
   tolerances the model was compiled under; a solver should honor
   `feasibility_tol` / `optimality_tol` / `max_iterations` /
   `time_limit_seconds` from that object rather than hardcoding its own.
3. **Re-validate at the boundary if you didn't produce the `ProblemData`
   yourself.** If a `ProblemData` arrived over a wire (deserialized JSON,
   a message queue, a GPU-host transfer), call
   `nirnaya_core.validation.validate_problem_data(data)` before solving.
   `Model.build()` already does this internally for data it produces.
4. **Return a `SolverResult`, always.** Even for `INFEASIBLE` /
   `UNBOUNDED` / error terminations, construct a `SolverResult` with the
   appropriate `SolverStatus` — don't invent a parallel result type.
   `primal` / `dual_values` / `reduced_costs` / `constraint_residuals` may
   be `None` when not applicable, but when present they must be
   index-aligned to `ProblemData.variable_names` /
   `constraint_names_eq + constraint_names_ub` as documented on
   `SolverResult`.
5. **Use `nirnaya_core.matrix.SparseMatrix` as your matrix type**, or
   convert at your backend's boundary (`.to_csr()` / `.to_csc()` /
   `.to_dense()`). Do not silently assume dense storage anywhere in a
   presolve or kernel-launch path; large refinery-scale LPs will not fit
   that assumption.
6. **Match on `.code`, not on exception message text**, when handling
   `NirnayaError` subclasses — messages may change between versions,
   codes will not (see `docs/INTEGRATION_CONTRACT.md`).
7. **New variable/constraint/status kinds are additive.** If Parts 2–6
   need a new `VariableType` (e.g. semi-continuous) or a new
   `SolverStatus` (e.g. a GPU-specific numerical failure mode), add a new
   enum member rather than repurposing an existing one — this keeps old
   serialized documents readable by new code.

See [`docs/INTEGRATION_CONTRACT.md`](docs/INTEGRATION_CONTRACT.md) for the
authoritative, versioned statement of what downstream code may rely on.

---

## 7. Example usage

```python
from nirnaya_core import (
    Model,
    ObjectiveSense,
    ConstraintSense,
    SolverResult,
    SolverStatus,
    OptimizationStatistics,
)

# 1. Build a small refinery blending-style LP.
model = Model(name="toy_blend")
model.add_variable("crude_a", lower=0.0, upper=100.0)
model.add_variable("crude_b", lower=0.0, upper=100.0)

model.set_objective(
    ObjectiveSense.MAXIMIZE,
    {"crude_a": 12.5, "crude_b": 9.0},   # profit per unit
)

model.add_constraint(
    "refinery_capacity",
    {"crude_a": 1.0, "crude_b": 1.0},
    ConstraintSense.LE,
    150.0,
)
model.add_constraint(
    "min_throughput",
    {"crude_a": 1.0, "crude_b": 1.0},
    ConstraintSense.GE,
    40.0,
)

# 2. Validate explicitly (build() also does this internally).
model.validate()  # raises ModelValidationError with every problem found, or returns None

# 3. Compile to a solver-ready, immutable numerical snapshot.
data = model.build()
print(data.variable_names)        # ('crude_a', 'crude_b')
print(data.c)                     # [12.5  9.0]
print(data.a_ub.to_dense())       # dense view, for inspection only

# 4. Serialize deterministically for storage/interchange.
from nirnaya_core.io import model_to_json, model_from_json
text = model_to_json(model)
restored = model_from_json(text)

# 5. What a solver backend (Part 2+) is expected to hand back:
result = SolverResult(
    status=SolverStatus.OPTIMAL,
    variable_names=data.variable_names,
    constraint_names=data.constraint_names_eq + data.constraint_names_ub,
    primal=(100.0, 50.0),
    statistics=OptimizationStatistics(iterations=4, primal_objective=1700.0),
)
print(result.primal_value("crude_a"))  # 100.0
```

Runnable, more extensive versions of this live in
[`examples/`](examples/).

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Core runtime dependencies are `numpy` and `scipy`
only.

## Running the tests

```bash
pytest
# or, with coverage:
pytest --cov=nirnaya_core --cov-report=term-missing
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
