"""
Standalone validation functions for :class:`~nirnaya_core.model.ProblemData` snapshots.

:meth:`nirnaya_core.model.Model.validate` covers the *model-building* layer
(name-addressed, before compilation). This module covers the *numerical*
layer: a :class:`~nirnaya_core.model.ProblemData` that arrived from
deserialization, a hand-rolled solver, or any other source that bypassed
:class:`~nirnaya_core.model.Model` entirely. Both layers raise the same
:class:`~nirnaya_core.validation.errors.ModelValidationError` shape so callers
have one exception type to handle regardless of which layer caught the
problem.
"""

from __future__ import annotations

from typing import List

import numpy as np

from nirnaya_core.model.problem_data import ProblemData
from nirnaya_core.validation.errors import (
    DimensionMismatchError,
    EmptyModelError,
    InvalidBoundsError,
    InvalidConstraintDimensionError,
    ModelValidationError,
    NaNInfError,
    ValidationError,
)


def validate_problem_data(data: ProblemData) -> None:
    """Validate a compiled :class:`~nirnaya_core.model.ProblemData` snapshot.

    Checks (in addition to what :class:`~nirnaya_core.model.ProblemData`'s own
    ``__post_init__`` already enforces structurally):

    * at least one variable is present,
    * no NaN in ``c``, ``b_eq``, ``b_ub``, or the sparse matrices,
    * bounds are consistent (``lower <= upper`` elementwise, no NaN),
    * matrix column counts match ``n_variables`` and row counts match the
      corresponding name lists.

    Raises:
        ModelValidationError: aggregating every problem found.
    """
    errors: List[ValidationError] = []

    if data.n_variables == 0:
        errors.append(EmptyModelError("ProblemData has zero variables"))

    if data.a_eq.n_cols != data.n_variables:
        errors.append(
            DimensionMismatchError(
                f"a_eq has {data.a_eq.n_cols} columns but there are {data.n_variables} variables"
            )
        )
    if data.a_ub.n_cols != data.n_variables:
        errors.append(
            DimensionMismatchError(
                f"a_ub has {data.a_ub.n_cols} columns but there are {data.n_variables} variables"
            )
        )
    if data.a_eq.n_rows != len(data.constraint_names_eq):
        errors.append(
            InvalidConstraintDimensionError(
                f"a_eq has {data.a_eq.n_rows} rows but constraint_names_eq has "
                f"{len(data.constraint_names_eq)} entries"
            )
        )
    if data.a_ub.n_rows != len(data.constraint_names_ub):
        errors.append(
            InvalidConstraintDimensionError(
                f"a_ub has {data.a_ub.n_rows} rows but constraint_names_ub has "
                f"{len(data.constraint_names_ub)} entries"
            )
        )

    if not np.all(np.isfinite(data.c)):
        errors.append(NaNInfError("Objective vector c contains NaN or Inf"))
    if not np.all(np.isfinite(data.b_eq)):
        errors.append(NaNInfError("b_eq contains NaN or Inf"))
    if not np.all(np.isfinite(data.b_ub)):
        errors.append(NaNInfError("b_ub contains NaN or Inf"))
    if data.a_eq.nnz and not np.all(np.isfinite(data.a_eq.data.data)):
        errors.append(NaNInfError("a_eq contains NaN or Inf entries"))
    if data.a_ub.nnz and not np.all(np.isfinite(data.a_ub.data.data)):
        errors.append(NaNInfError("a_ub contains NaN or Inf entries"))

    nan_lower = np.isnan(data.lower)
    nan_upper = np.isnan(data.upper)
    if np.any(nan_lower) or np.any(nan_upper):
        errors.append(NaNInfError("Variable bounds contain NaN"))
    else:
        bad = data.lower > data.upper
        if np.any(bad):
            bad_indices = np.nonzero(bad)[0].tolist()
            bad_names = [data.variable_names[i] for i in bad_indices]
            errors.append(
                InvalidBoundsError(
                    f"Variables with lower > upper: {bad_names}",
                    details={"variables": bad_names},
                )
            )

    if errors:
        raise ModelValidationError(errors, details={"source": "ProblemData"})


def is_problem_data_valid(data: ProblemData) -> bool:
    """Return True iff :func:`validate_problem_data` would raise nothing."""
    try:
        validate_problem_data(data)
    except ModelValidationError:
        return False
    return True


__all__ = ["validate_problem_data", "is_problem_data_valid"]
