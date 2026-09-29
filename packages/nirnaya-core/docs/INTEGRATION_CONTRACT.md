# Nirnaya Core — Integration Contract

**Audience:** Parts 2–6 of the Nirnaya project (presolve, the LP solver
engine, GPU/native acceleration, reporting/UX, and anything else built on
top of Part 1).

**Status:** Governs `nirnaya-core` version `0.1.0` (see
`nirnaya_core.__version__`). This document is the authoritative statement
of what downstream code may depend on. If something you need isn't listed
here as stable, treat it as an implementation detail that may change
without notice, and raise it with the Part 1 maintainers instead of
depending on it directly.

---

## 1. Stability levels

Everything in this package falls into one of three buckets:

| Level | Meaning | Examples |
|---|---|---|
| **Stable public API** | Covered by semantic versioning. Breaking changes require a major version bump and a migration note in `CHANGELOG` (once one exists). Safe to import and depend on from Parts 2–6. | `Model`, `ProblemData`, `SolverResult`, `SolverStatus`, `NumericalConfig`, `SparseMatrix`, all `nirnaya_core.exceptions`/`nirnaya_core.validation` error types and their `.code` values, `nirnaya_core.io.*` |
| **Additive-only** | New members may be added in a minor version; existing members are never renamed, renumbered, or removed within a major version. | `VariableType`, `ConstraintSense`, `ObjectiveSense`, `SolverStatus` enum members; error `.code` strings |
| **Internal / unstable** | May change in any release. Do not import or depend on. | Anything under a module-private name (leading underscore), internal helper functions, the exact scipy.sparse internals behind `SparseMatrix`, private validation helper functions not re-exported from `nirnaya_core.validation` |

If a name is exported from a package's top-level `__init__.py` (i.e. it
appears in that module's `__all__`), it is stable public API. If you have
to reach into a submodule that isn't re-exported to get something, that's
a signal it's internal.

---

## 2. The `Model` → `ProblemData` → `SolverResult` contract

This is the core hand-off surface between Part 1 and everything after it.

### 2.1 `Model` (build-time, mutable)

* Owned and used primarily by callers constructing a problem (UI layers,
  file importers, Part 6 reporting tools that need to reconstruct a
  model for display). **Solver backends (Part 2+) should not need to
  import or manipulate `Model` directly** — they consume `ProblemData`.
* `Model.validate()` raises `nirnaya_core.validation.ModelValidationError`
  (never a bare `ValueError`) listing every problem found via its
  `.errors` list. `Model.build()` calls `validate()` internally and will
  never return a `ProblemData` for an invalid model.
* Variable/constraint ordering is **insertion order**. This order is
  preserved into `ProblemData.variable_names` /
  `constraint_names_eq` / `constraint_names_ub`, so index `i` in any
  numerical array always corresponds to the same name across the whole
  pipeline for a given build.

### 2.2 `ProblemData` (solve-time, immutable)

This is the primary artifact solver backends should target.

* **Guaranteed immutable.** All numpy array fields (`c`, `b_eq`, `b_ub`,
  `lower`, `upper`) are marked read-only (`array.flags.writeable is
  False`) after construction. `SparseMatrix` fields are frozen
  dataclasses over an internally-copied CSR matrix. A solver may safely
  hold a reference to a `ProblemData` across iterations without
  defensive copying.
* **Guaranteed shape-consistent.** `ProblemData.__post_init__` enforces:
  `c`, `lower`, `upper`, `var_types` all have length `n_variables`;
  `a_eq` is `(n_eq, n_variables)`, `b_eq` has length `n_eq`; `a_ub` is
  `(n_ub, n_variables)`, `b_ub` has length `n_ub`. A `ProblemData` you
  successfully construct is shape-valid by construction — you do not
  need to re-check shapes, only (if you did not produce it via
  `Model.build()`) numerical validity via
  `nirnaya_core.validation.validate_problem_data()`.
* **Guaranteed sign convention.** Every row of `a_ub` is in `<=` form.
  There is no `>=` row anywhere in a `ProblemData`; `ConstraintSense.GE`
  constraints were negated during `Model.build()`. A solver only ever
  needs to implement `<=` and `==`.
* **dtype is a promise, not a suggestion.** Every array in `ProblemData`
  is constructed at `ProblemData.config.dtype` (`float64` by default).
  Do not assume `float64` unconditionally — read `config.dtype`, or
  `config.to_dict()["dtype"]` after deserialization.
* **Index alignment.** `variable_names[i]` describes column `i` of `c`,
  `a_eq`, `a_ub`, and entry `i` of `lower`, `upper`, `var_types`.
  `constraint_names_eq[i]` describes row `i` of `a_eq` / entry `i` of
  `b_eq`. `constraint_names_ub[i]` describes row `i` of `a_ub` / entry
  `i` of `b_ub`. These two constraint name lists are **separate
  namespaces from each other in row-index terms** (row 0 of `a_eq` and
  row 0 of `a_ub` are unrelated), but constraint *names* are unique
  across both (enforced by `Model`).

### 2.3 `SolverResult` (post-solve, immutable)

* Solver backends **must** return a `SolverResult`, never raise for a
  "normal" non-optimal outcome (infeasible, unbounded, iteration limit).
  Exceptions are reserved for genuine programming errors or malformed
  input (e.g. a `ProblemData` that fails
  `validate_problem_data`) — not for legitimate solve outcomes.
* `status` must be a real `SolverStatus`, matching what actually
  happened. Do not report `OPTIMAL` unless `primal` (and ideally
  `dual_values` / `reduced_costs`) are populated and satisfy
  `ProblemData.config`'s tolerances.
* When `primal` is present, its length must equal
  `len(variable_names)`, and the values are index-aligned to
  `variable_names` exactly as described in §2.2 for `ProblemData` — in
  practice, callers should set `variable_names =
  problem_data.variable_names` and `constraint_names =
  problem_data.constraint_names_eq + problem_data.constraint_names_ub`
  when constructing the result, so the two structures share one index
  space.
* `constraint_residuals`, when present, follow the residual sign
  convention documented on `Constraint.residual`: `a^T x - rhs`. For the
  `a_ub` block (all rows `<=`), a positive residual means violated; for
  the `a_eq` block, any nonzero residual means violated (within
  `feasibility_tol`).
* `OptimizationStatistics` fields are all optional — a backend that
  doesn't track `dual_objective`, for instance, should leave it `None`
  rather than fabricating a value.

---

## 3. Numerical contract

* **Tolerances live in `NumericalConfig`, not in solver code.** A
  backend must accept a `NumericalConfig` (typically
  `problem_data.config`) and use its `feasibility_tol`,
  `optimality_tol`, `bound_tol`, `pivot_tol`, `zero_tol`,
  `max_iterations`, and `time_limit_seconds` rather than hardcoding
  independent constants. If a backend needs an additional tolerance not
  present on `NumericalConfig`, propose adding a field rather than
  inventing a parallel config object.
* **Effective infinity.** Use
  `config.is_effectively_infinite(value)` rather than `value ==
  math.inf` when deciding whether a bound is "real" — some numerical
  kernels (GPU, fixed-point) represent "no bound" as a large finite
  sentinel rather than IEEE infinity, and `config.infinity` (default
  `1e20`) is the agreed threshold.
* **No backend may silently change dtype.** If a GPU kernel internally
  needs `float32` even though `ProblemData.config.dtype` is `float64`
  (or vice versa), that conversion must be explicit and logged/flagged
  in the returned `SolverResult.statistics.extra`, not silent.
* **NaN must never appear in a `SolverResult`'s numerical fields.**
  `OptimizationStatistics` rejects NaN at construction (see its
  `__post_init__`); `SolverResult`'s array fields will coerce via
  `float()`, which will raise if a backend tries to pass a Python `nan`
  object through non-strictly — but backends should treat this as a hard
  invariant, not rely on incidental enforcement.

---

## 4. Sparse matrix contract (`SparseMatrix`)

* Backed internally by `scipy.sparse.csr_matrix`, exposed via
  `.to_csr()` / `.to_csc()` / `.to_dense()` / `.matvec()` /
  `.to_coo_triplets()`. These conversions are the stable surface — do
  not depend on `SparseMatrix.data` being exactly a `csr_matrix` object
  in future versions (a native/GPU backend replacing scipy internally
  should keep these accessor methods working).
* `.to_dense()` is explicit and is not called anywhere in this package's
  own internal code paths by default — it exists for small-model
  debugging, tests, and display, not as a hidden fallback. Treat any use
  of `.to_dense()` in a hot path (a real solver iteration) as a
  performance bug.
* Construction via `SparseMatrix.from_triplets` sums duplicate
  `(row, col)` entries (standard COO semantics) and eliminates explicit
  zeros. `SparseMatrix.to_coo_triplets()` always returns entries sorted
  by `(row, col)` — this ordering is guaranteed and is what
  serialization relies on for determinism.
* All matrix coefficients are validated finite (no NaN/Inf) at
  construction; there is no code path that produces a `SparseMatrix`
  containing non-finite data.

---

## 5. Error handling contract

* Every exception raised by this package is a `nirnaya_core.exceptions.NirnayaError`
  or a subclass. Catch `NirnayaError` if you want "anything this package
  raised"; catch a specific subclass (or match `.code`) for finer
  control.
* **Match on `.code`, not on `str(exception)` or `type(exception).__name__`
  free-form parsing.** `.code` values (e.g. `"E_DIM_MISMATCH"`,
  `"E_VALIDATION"`, `"E_MODEL_INVALID"`) are part of the additive-only
  contract in §1: they will not be renamed, and new codes are only ever
  added under new subclasses.
* `Model.validate()` / `Model.build()` / `validate_problem_data()` raise
  exactly one exception type on failure —
  `nirnaya_core.validation.ModelValidationError` — with every
  individual problem available via `.errors` (a list of the specific
  subclasses: `DimensionMismatchError`, `InvalidBoundsError`,
  `EmptyModelError`, `DuplicateNameError`, `NaNInfError`,
  `InvalidObjectiveError`, `UnsupportedVariableTypeError`,
  `UnknownVariableReferenceError`, `InvalidConstraintSenseError`,
  `InvalidConstraintDimensionError`). Do not assume validation stops at
  the first error — it does not, by design, so that a caller (e.g. a
  UI in Part 6) can show every problem to the user at once.
* All exceptions expose `.to_dict()` (JSON-serializable: `error_type`,
  `code`, `message`, `details`). Use this if propagating validation
  failures across a process boundary (e.g. a solver microservice
  returning a 4xx with error detail).

---

## 6. Serialization contract (`nirnaya_core.io`)

* `model_to_json` / `problem_data_to_json` / `solver_result_to_json`
  produce **deterministic** output: identical logical content always
  produces byte-identical JSON (sorted keys, sorted sparse triplets,
  fixed float formatting). This is safe to hash for caching or change
  detection.
* Every serialized document is wrapped in an envelope:
  `{"schema_version": <int>, "kind": <"Model"|"ProblemData"|"SolverResult">, "data": {...}}`.
  Readers (`model_from_json` etc.) reject documents with the wrong
  `kind` or an unsupported `schema_version`, raising
  `SerializationError`. **`SCHEMA_VERSION` bumps are breaking changes**
  to the on-disk format and follow the same stability rules as code
  (§1) — a new major version of `nirnaya-core` may introduce
  `schema_version = 2` with a documented migration, but a reader built
  against `schema_version = 1` is not expected to silently accept
  `2`.
* `+inf` / `-inf` values (legal in variable bounds) are serialized as
  the sentinel strings `"Infinity"` / `"-Infinity"` rather than
  non-standard JSON tokens, so output is valid per RFC 8259. Do not
  reinterpret these sentinel strings as ordinary variable/constraint
  names — a name that happens to be exactly `"Infinity"` will round-trip
  incorrectly through this layer (a known, accepted limitation; avoid
  naming a variable or constraint `"Infinity"` or `"-Infinity"`).

---

## 7. Extension points reserved for Parts 2–6

These are explicitly *not* implemented in Part 1, but the data model was
shaped to accommodate them without a breaking change:

* **Integer/binary (MIP) solving.** `VariableType.INTEGER` and
  `VariableType.BINARY` already exist and validate structurally; no LP
  relaxation, branch-and-bound, or cutting-plane logic exists in Part 1.
  A MIP-capable backend can consume `ProblemData.var_types` directly.
* **GPU/native backends.** `SparseMatrix`'s public surface
  (`.to_csr()`, `.to_csc()`, `.matvec()`, `.to_coo_triplets()`) is
  designed so a GPU-resident sparse format can sit behind the same
  interface, and `NumericalConfig.dtype` supports `float32` today for
  that purpose.
* **Presolve.** Nothing in Part 1 removes redundant rows, tightens
  bounds, or detects fixed variables. A presolve stage in Parts 2–6
  should accept a `ProblemData` and return a new, smaller `ProblemData`
  (plus whatever mapping it needs to translate a reduced solution back
  to the original variable/constraint space) — this package does not
  currently define that mapping type, and Parts 2–6 should propose one
  rather than smuggling it through `OptimizationStatistics.extra`
  long-term.
* **Additional `SolverStatus` / error codes.** New members may be added
  (additive-only, §1) as new backends need them (e.g. a GPU-specific
  numerical failure mode). Propose additions rather than overloading
  `NUMERICAL_ERROR` or `UNKNOWN` for everything.

---

## 8. What is explicitly out of scope for Part 1

To set expectations: the following are **not** provided by
`nirnaya-core` and should not be assumed present when integrating.

* Any actual LP/MIP solving algorithm (simplex, interior-point,
  branch-and-bound).
* Presolve, scaling, or basis management.
* A CLI, HTTP service, or RPC layer.
* GPU kernels or CUDA/HIP bindings.
* Multi-objective, quadratic, or nonlinear objective support.
* Warm-starting / basis reuse across solves.
* A file-format importer for third-party LP formats (MPS, LP format,
  etc.) — `nirnaya_core.io` only covers this package's own JSON schema.
