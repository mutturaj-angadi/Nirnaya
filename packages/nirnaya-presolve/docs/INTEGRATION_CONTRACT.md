# Integration Contract: how Part 3 (solver) consumes `nirnaya-presolve`

This document specifies exactly how a downstream solver ("Part 3")
consumes a presolved model produced by `nirnaya-presolve`, and how it gets
back to the original-space solution. It assumes `nirnaya-core`'s public
API as documented in `docs/ASSUMED_CORE_API.md`; if the real core API
differs, only variable/constraint *identity* (by `name: str`) matters for
this contract, and that is expected to be stable.

## 1. Running presolve

```python
from nirnaya_presolve import Presolver, PresolveTolerances

presolver = Presolver(tolerances=PresolveTolerances())   # or customize
result = presolver.run(original_model)   # original_model: nirnaya_core.LPModel
```

`presolver.run` never raises for an ordinary infeasible/unbounded/degenerate
input — those are reported *in* the result, not as exceptions. It may raise
`nirnaya_presolve.ModelBuildError` only if reconstructing the reduced model
via `nirnaya_core`'s public constructors fails, which indicates a
presolve bug, not a property of the input model.

## 2. Checking the outcome before solving anything

```python
if result.infeasible:
    # Presolve proved infeasibility outright (empty row, bound conflict,
    # activity-bound infeasibility, etc). result.infeasible_reason is a
    # human-readable explanation. Do NOT attempt to solve
    # result.reduced_model -- it is None in this case.
    ...
elif result.unbounded:
    # Presolve proved the objective is unbounded (an empty-column variable
    # with a favorable objective coefficient and no finite limiting bound).
    # result.unbounded_reason explains why. result.reduced_model is None.
    ...
else:
    # result.reduced_model is a valid nirnaya_core.LPModel, ready to solve.
    ...
```

Part 3 **must** check `result.infeasible` / `result.unbounded` before
touching `result.reduced_model`; it is `None` in both cases specifically
so that a solver that forgets this check fails loudly (`AttributeError`)
rather than silently solving `None`.

Note: presolve's infeasibility/unboundedness detection is *sound but
incomplete* — if it says infeasible/unbounded, that's certain (subject to
the configured tolerances); if it says neither, the model may still turn
out to be infeasible or unbounded, and Part 3's solver must still detect
that itself in the ordinary way and report it via `Solution.status`.

## 3. Solving the reduced model

`result.reduced_model` is an ordinary `nirnaya_core.LPModel`: same
`Variable`/`Constraint`/`objective` shape as any other model Part 3
already knows how to solve. It has:

* possibly fewer variables and constraints than the original (eliminated
  ones are gone entirely — not present with trivial/fixed bounds);
* possibly different (scaled) coefficients, bounds, and objective
  coefficients than the corresponding original entries, if scaling was
  enabled (`Presolver(enable_scaling=True)`, the default);
* variable and constraint **names preserved exactly** from the original
  model for everything that survives (presolve never renames).

Part 3 solves it however it normally solves an `LPModel` and produces an
`nirnaya_core.Solution` whose `values` dict is keyed by
`result.reduced_model`'s variable names (i.e. reduced-space names, which
are a subset of original names) and whose `status` is one of
`SolveStatus.OPTIMAL / INFEASIBLE / UNBOUNDED / ERROR`.

```python
reduced_solution = my_solver.solve(result.reduced_model)
```

## 4. Recovering the original-space solution

```python
original_solution = result.recover_solution(reduced_solution)
```

Contract for `recover_solution`:

* **Input:** an `nirnaya_core.Solution` whose `values` cover *every*
  variable name in `result.reduced_model.variable_names()`. (Values for
  names outside that set are ignored if present.)
* **If `reduced_solution.status != SolveStatus.OPTIMAL`:** no
  reconstruction is attempted — a `Solution` with the same status, the
  same `objective_value`, and an **empty** `values` dict is returned. Part
  3 should treat a non-optimal reduced solution as authoritative for
  status/infeasibility purposes and not expect recovered values.
* **If the reduced solution is missing a value for some active reduced
  variable:** raises `nirnaya_presolve.RecoveryError` — this indicates a
  mismatch between the model Part 3 actually solved and
  `result.reduced_model` (e.g. it solved a stale/different model). Not
  silently tolerated.
* **On success:** returns an `nirnaya_core.Solution` whose `values` dict
  has exactly one entry per variable in the **original** model
  (`values.keys() == set(original_model.variable_names())`), and whose
  `objective_value` is recomputed from the *original* model's objective
  coefficients and offset applied to the recovered values (i.e. it is not
  simply forwarded from the reduced solve — this is a self-check: if
  presolve has a bug, the recomputed value will disagree with what a
  from-scratch solve of the original model would report, rather than
  silently propagating a wrong number).
* Un-scaling and all eliminated-variable back-substitution (fixed
  variables, singleton-column and doubleton-equality aggregations) are
  handled internally; Part 3 never needs to look at `result.scaling`,
  `result.fixed_variables`, or `result.eliminated_variables` to get a
  correct answer — those are exposed for diagnostics/logging/UI, not
  because Part 3 needs to reimplement any of this logic itself.

## 5. Fields available for diagnostics (not required for correctness)

| Field | Type | Meaning |
|---|---|---|
| `result.statistics` | `PresolveStatistics` | Counts of every reduction applied, round count, timing. `.summary()` gives a one-line string. |
| `result.trace` | `tuple[TransformationRecord]` | Every transformation applied, in order, with a human-readable description — useful for logging/debugging a specific model. |
| `result.fixed_variables` | `dict[str, float]` | Original-space variables presolve fixed to a constant. |
| `result.eliminated_variables` | `dict[str, EliminatedVariable]` | Original-space variables presolve expressed as an affine function of other (possibly also eliminated) variables, with a human-readable `.relation` string. |
| `result.reduced_to_original_variables` / `.original_to_reduced_variables` | `dict[str, str \| None]` | Name-preserving mapping; the `None` values on the `original_to_reduced_*` side mark variables/constraints that did not survive to the reduced model. |
| `result.scaling` | `dict` | `{"row_scale": {...}, "col_scale": {...}}`, cumulative scaling factors actually applied (only entries that changed from 1.0 are listed). |

## 6. What Part 3 must NOT do

* Must not mutate `result.reduced_model` or any object reachable from
  `result` — presolve's internal state assumes these are immutable value
  objects (this matches nirnaya-core's own immutability contract).
* Must not attempt its own variable-name translation, scaling reversal, or
  fixed-variable back-substitution — use `result.recover_solution` for
  all of that, so there is exactly one place (this package) responsible
  for the original<->reduced correspondence.
* Must not assume reduced-model variable/constraint *order* matches the
  original model's order — only *names* are stable. Index-based
  correspondence between original and reduced models is not part of this
  contract.
* Must not skip the `result.infeasible` / `result.unbounded` check (§2).

## 7. Minimal end-to-end example

```python
from nirnaya_presolve import Presolver
from nirnaya_core import SolveStatus

result = Presolver().run(original_model)

if result.infeasible or result.unbounded:
    print(result.summary())
else:
    reduced_solution = my_solver.solve(result.reduced_model)
    if reduced_solution.status == SolveStatus.OPTIMAL:
        original_solution = result.recover_solution(reduced_solution)
        print(original_solution.objective_value, original_solution.values)
    else:
        print("Reduced problem was not solved to optimality:", reduced_solution.status)
```
