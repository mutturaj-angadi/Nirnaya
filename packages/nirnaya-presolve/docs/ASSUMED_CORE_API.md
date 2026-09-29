# Assumed `nirnaya-core` Public API

> **Status: ASSUMPTION — to be reconciled when nirnaya-core (Part 1) is available.**
>
> `nirnaya-presolve` was built without access to the actual `nirnaya-core`
> source. To make progress, this document fixes a minimal, reasonable public
> API that `nirnaya-core` is assumed to expose, and `nirnaya-presolve` is
> written *only* against this surface (no private/internal structures, no
> duplicated math layer). A lightweight, clearly-labeled stand-in package
> (`tests/_core_shim/nirnaya_core`) implements exactly this surface so the
> test suite can run today. It is **test-only scaffolding, not a deliverable**
> — when real `nirnaya-core` lands, delete the shim, `pip install
## nirnaya-core`, and re-run the test suite. If real names/signatures differ,
> only `nirnaya_presolve/adapter.py` (the single translation boundary between
> core objects and presolve's internal sparse representation) should need to
> change — see `docs/INTEGRATION_CONTRACT.md`.

## Package: `nirnaya_core`

### Enums

```python
class Sense(Enum):
    LE = "<="   # a·x <= rhs
    GE = ">="   # a·x >= rhs
    EQ = "=="   # a·x == rhs

class ObjectiveSense(Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"

class SolveStatus(Enum):
    OPTIMAL
    INFEASIBLE
    UNBOUNDED
    ERROR
```

### `Variable` (immutable value object)

```python
class Variable:
    name: str
    lower: float            # may be -inf
    upper: float            # may be +inf
    is_integer: bool = False
```

### `Constraint` (immutable value object)

```python
class Constraint:
    name: str
    coefficients: Mapping[str, float]   # sparse: var name -> coeff (zeros omitted)
    sense: Sense
    rhs: float
```

### `LPModel` (immutable; built via constructor, never mutated in place)

```python
class LPModel:
    def __init__(
        self,
        variables: Sequence[Variable],
        constraints: Sequence[Constraint],
        objective: Mapping[str, float],     # var name -> coeff, sparse
        objective_sense: ObjectiveSense,
        objective_offset: float = 0.0,
        name: str = "",
    ): ...

    variables: Sequence[Variable]
    constraints: Sequence[Constraint]
    objective: Mapping[str, float]
    objective_sense: ObjectiveSense
    objective_offset: float
    name: str

    def variable_names(self) -> List[str]: ...
    def constraint_names(self) -> List[str]: ...
    def get_variable(self, name: str) -> Variable: ...
    def get_constraint(self, name: str) -> Constraint: ...
```

### `Solution` (immutable value object, produced by a solver)

```python
class Solution:
    def __init__(
        self,
        values: Mapping[str, float],   # var name -> value
        objective_value: float,
        status: SolveStatus,
    ): ...

    values: Mapping[str, float]
    objective_value: float
    status: SolveStatus
```

## What `nirnaya-presolve` relies on, precisely

1. `LPModel`, `Variable`, `Constraint`, `Solution` are constructed via their
   public `__init__`/attributes only — never subclassed, never mutated.
2. Variable/constraint identity is by `name: str` (unique within a model).
   `nirnaya-presolve` never assumes a stable integer index from core; any
   indexing is internal to presolve and discarded at the boundary.
3. Sparse dict-of-float representation for `Constraint.coefficients` and
   `LPModel.objective` (missing key == coefficient 0).
4. `±inf` (Python `float('inf')` / `float('-inf')`) is the representation of
   an unbounded variable bound.
5. No dependency on a core solver, core presolve, or core numerical routines.
   `nirnaya-presolve` reimplements none of core's math *model* layer — it
   only reads models through the accessors above and produces new models
   through the same constructors.

If the real `nirnaya-core` differs (e.g. dense arrays, a `Bounds` object
instead of `lower`/`upper` floats, integer-indexed variables), update
`nirnaya_presolve/adapter.py::model_from_core` and
`adapter.py::model_to_core` to translate — the rest of the package operates
on presolve's own internal representation and is insulated from this choice.
