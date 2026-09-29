"""
nirnaya_api
===========

Application / service layer for the Nirnaya optimization stack.

This package does **not** implement any solving logic itself. It is a thin,
well-tested orchestration layer on top of:

    * nirnaya-core      (problem model primitives)
    * nirnaya-presolve  (presolve / reduction passes)
    * nirnaya-solver    (the actual numerical solver)
    * nirnaya-gpu       (device discovery / GPU backends)

It exposes exactly three interfaces to that stack:

    1. A Python API      -> :mod:`nirnaya_api.model`
    2. A CLI              -> ``nirnaya`` (see :mod:`nirnaya_api.cli`)
    3. A REST API         -> :mod:`nirnaya_api.server` (FastAPI app)

Quick start (Python API)
-------------------------

>>> from nirnaya_api import Model
>>> m = Model("toy")
>>> x = m.add_variable("x", lb=0, ub=10)
>>> y = m.add_variable("y", lb=0, ub=10)
>>> m.add_constraint("c1", {"x": 1, "y": 1}, sense="<=", rhs=12)
>>> m.set_objective({"x": 1, "y": 2}, sense="max")
>>> result = m.solve(solver="auto", backend="auto", time_limit=5.0)
>>> result.status
'optimal'
"""

from .version import __version__
from .model import Model, Variable, Constraint, Objective
from .solution import Solution, SolveStatus
from .options import SolveOptions
from .errors import (
    NirnayaAPIError,
    ValidationError,
    DependencyUnavailableError,
    ResourceLimitError,
    SolveTimeoutError,
)
from .backends import DeviceInfo, list_devices, select_backend

__all__ = [
    "__version__",
    "Model",
    "Variable",
    "Constraint",
    "Objective",
    "Solution",
    "SolveStatus",
    "SolveOptions",
    "NirnayaAPIError",
    "ValidationError",
    "DependencyUnavailableError",
    "ResourceLimitError",
    "SolveTimeoutError",
    "DeviceInfo",
    "list_devices",
    "select_backend",
]
