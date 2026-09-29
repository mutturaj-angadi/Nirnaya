# nirnaya-presolve

LP preprocessing / presolve engine for **Nirnaya** — Part 2 of the
project. Transforms an input `nirnaya_core.LPModel` into a numerically
improved, reduced model, while preserving mathematical equivalence to the
extent it is safe to do so, and provides an exact `recover_solution`
postsolve path back to the original variable space.

> **Integration note:** this module was built without access to the real
> `nirnaya-core` package. It is written against a documented, minimal
> assumed public API (`docs/ASSUMED_CORE_API.md`), with a test-only
> stand-in implementation of that API under `tests/_core_shim/` used to
> run the test suite today. See that document and
> `docs/INTEGRATION_CONTRACT.md` for exactly what needs reconciling once
> the real `nirnaya-core` is available.

## What it does

Given an LP, applies (to a fixed point, then scales once):

- fixed-variable elimination
- empty row / empty column detection
- singleton row detection (-> bound tightening)
- singleton column elimination (free variable, equality row)
- doubleton equality aggregation (variable substitution)
- activity-based bound propagation / implied bounds
- redundant constraint detection
- duplicate (proportional) constraint merging
- coefficient cleanup (numerically negligible terms only, with a proven
  worst-case-impact bound — never applied to an unbounded direction)
- row/column geometric-mean scaling
- infeasibility detection (empty rows, bound conflicts, activity bounds)
- unboundedness detection (empty columns with a one-sided objective and no
  limiting bound)

Every transformation is logged to `PresolveResult.trace` in application
order. All tolerances are configurable via `PresolveTolerances` and are
individually documented with the numerical assumption they encode.

## Quick start

```python
from nirnaya_presolve import Presolver

result = Presolver().run(model)          # model: nirnaya_core.LPModel

if result.infeasible or result.unbounded:
    print(result.summary())
else:
    solution = my_solver.solve(result.reduced_model)
    original_solution = result.recover_solution(solution)
```

See `docs/INTEGRATION_CONTRACT.md` for the full contract Part 3 (the
solver) is expected to follow.

## Package layout

```
src/nirnaya_presolve/
  adapter.py            # sole translation boundary to/from nirnaya_core
  exceptions.py
  presolve/
    tolerances.py        # PresolveTolerances
    state.py              # internal mutable sparse LP representation
    passes.py             # every transformation, independently testable
    engine.py              # Presolver: fixed-point orchestration
    result.py               # PresolveResult (public output)
    postsolve.py             # recover_solution implementation
    trace.py                  # TransformationRecord / TraceLog
    statistics.py             # PresolveStatistics
docs/
  ASSUMED_CORE_API.md    # the nirnaya-core surface this was built against
  INTEGRATION_CONTRACT.md # exact contract for the Part 3 solver
tests/
  _core_shim/nirnaya_core/  # TEST-ONLY stand-in for nirnaya-core
  conftest.py                # wires the shim in only if real core is absent
  test_*.py                   # one file per transformation, plus regression/
                               #   postsolve/integration tests
```

## Running the tests

```bash
pip install -e ".[test]"
pytest
```
