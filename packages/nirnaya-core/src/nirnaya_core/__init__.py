"""
Nirnaya Core — mathematical foundation and stable interfaces for the
Nirnaya GPU-accelerated optimization solver (SIH 2026, PS 26119, MRPL).

This package (Part 1 of the Nirnaya project) defines:

* the LP model representation (:mod:`nirnaya_core.model`),
* sparse matrix storage (:mod:`nirnaya_core.matrix`),
* explicit numerical configuration (:mod:`nirnaya_core.numeric`),
* solver status and result data structures
  (:mod:`nirnaya_core.status`, :mod:`nirnaya_core.solution`),
* validation (:mod:`nirnaya_core.validation`),
* deterministic serialization (:mod:`nirnaya_core.io`),
* a machine-readable exception hierarchy (:mod:`nirnaya_core.exceptions`).

It deliberately ships **no solver**. See ``docs/INTEGRATION_CONTRACT.md``
for the interfaces Parts 2-6 are expected to build against, and
``README.md`` for an overview and usage examples.

The most commonly used names are re-exported here for convenience::

    from nirnaya_core import Model, ConstraintSense, ObjectiveSense
"""

from nirnaya_core.exceptions import (
    ConfigurationError,
    ImmutabilityError,
    NirnayaError,
    SerializationError,
    ValidationError,
)
from nirnaya_core.matrix import SparseMatrix, SparseMatrixError
from nirnaya_core.model import (
    Constraint,
    ConstraintSense,
    LinearObjective,
    Model,
    ObjectiveSense,
    ProblemData,
    Variable,
    VariableType,
)
from nirnaya_core.numeric import DEFAULT_CONFIG, DEFAULT_DTYPE, NumericalConfig
from nirnaya_core.solution import OptimizationStatistics, SolverResult
from nirnaya_core.status import SolverStatus

__version__ = "0.1.0"

#: The Nirnaya project part this package implements. Referenced by Parts
#: 2-6 for compatibility checks.
NIRNAYA_PART = 1

__all__ = [
    "__version__",
    "NIRNAYA_PART",
    # model
    "Model",
    "Variable",
    "Constraint",
    "LinearObjective",
    "ProblemData",
    "VariableType",
    "ConstraintSense",
    "ObjectiveSense",
    # matrix
    "SparseMatrix",
    "SparseMatrixError",
    # numeric
    "NumericalConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_DTYPE",
    # status / solution
    "SolverStatus",
    "SolverResult",
    "OptimizationStatistics",
    # exceptions
    "NirnayaError",
    "ValidationError",
    "SerializationError",
    "ConfigurationError",
    "ImmutabilityError",
]
