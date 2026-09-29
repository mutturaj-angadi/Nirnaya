"""
nirnaya-presolve
=================

LP preprocessing / presolve engine for the Nirnaya project (Part 2).

Consumes an `nirnaya_core.LPModel` (public API only — see
docs/ASSUMED_CORE_API.md) and produces a `PresolveResult` holding a reduced
`LPModel`, full transformation metadata, and the mappings needed to recover
an original-space solution from a reduced-space one via
`recover_solution`.

Typical usage
-------------

    from nirnaya_presolve import Presolver, PresolveTolerances

    presolver = Presolver(tolerances=PresolveTolerances())
    result = presolver.run(model)

    if result.infeasible:
        ...
    else:
        reduced_solution = my_solver.solve(result.reduced_model)
        original_solution = result.recover_solution(reduced_solution)
"""

from .presolve.tolerances import PresolveTolerances
from .presolve.trace import TransformationRecord
from .presolve.statistics import PresolveStatistics
from .presolve.result import PresolveResult, EliminatedVariable
from .presolve.engine import Presolver
from .exceptions import PresolveError, ModelBuildError

__all__ = [
    "Presolver",
    "PresolveTolerances",
    "PresolveResult",
    "EliminatedVariable",
    "TransformationRecord",
    "PresolveStatistics",
    "PresolveError",
    "ModelBuildError",
]

__version__ = "0.1.0"
