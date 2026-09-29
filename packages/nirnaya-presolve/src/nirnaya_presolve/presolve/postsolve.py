"""
Postsolve: turn a `Solution` in *reduced* variable space back into a
`Solution` in *original* variable space.

Algorithm
---------
1. Start from the reduced solution's values, keyed by reduced-model
   variable names (which are always a subset of the original names, since
   presolve never renames a variable).
2. Un-scale: for every reduced variable that scaling touched, multiply its
   value by the cumulative column scale factor recorded during presolve
   (`x_original_frame = x_scaled_frame * col_scale`).
3. Replay `st.eliminated` in *reverse* order (i.e. last eliminated first).
   Each record is one of:
     - fixed:        value[name] = fixed_value
     - aggregation:   value[name] = alpha * value[other_name] + beta
     - multi_affine:  value[name] = beta + sum(coeff * value[other])
   Because later eliminations can reference variables eliminated earlier
   (e.g. a doubleton aggregation whose surviving variable was itself later
   fixed), replaying in reverse guarantees every "other" reference has
   already been resolved by the time it's needed -- earlier eliminations
   were computed using the *pre-elimination* model, i.e. they only ever
   reference variables that were still active at that time, which (walking
   backwards) are exactly the ones resolved so far.
4. The result covers every original variable name exactly once.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from nirnaya_core import SolverResult, SolverStatus

from ..exceptions import RecoveryError


def recover_solution(state, reduced_solution: SolverResult) -> SolverResult:
    if reduced_solution.status is not SolverStatus.OPTIMAL or reduced_solution.primal is None:
        # Nothing to recover for infeasible/unbounded/error reduced
        # solutions -- pass the status through untouched with an empty
        # value map rather than fabricate values.
        return SolverResult(status=reduced_solution.status,
                            variable_names=state.var_names,
                            constraint_names=reduced_solution.constraint_names,
                            message=reduced_solution.message,
                            statistics=reduced_solution.statistics)

    values: Dict[str, float] = {}

    # 1 & 2: seed with reduced values, un-scaled back to the pre-scaling frame.
    for c in state.active_col_indices():
        name = state.var_names[c]
        reduced_values = dict(zip(reduced_solution.variable_names, reduced_solution.primal))
        if name not in reduced_values:
            raise RecoveryError(
                f"Reduced solution is missing a value for active reduced "
                f"variable {name!r}."
            )
        values[name] = reduced_values[name] * state.col_scale[c]

    # 3: replay eliminations in reverse order.
    for record in reversed(state.eliminated):
        if record.mode == "fixed":
            values[record.name] = record.fixed_value
        elif record.mode == "aggregation":
            if record.other_name not in values:
                raise RecoveryError(
                    f"Cannot recover {record.name!r}: dependency "
                    f"{record.other_name!r} not yet resolved."
                )
            values[record.name] = record.alpha * values[record.other_name] + record.beta
        elif record.mode == "multi_affine":
            total = record.beta
            for other_name, coeff in record.terms.items():
                if other_name not in values:
                    raise RecoveryError(
                        f"Cannot recover {record.name!r}: dependency "
                        f"{other_name!r} not yet resolved."
                    )
                total += coeff * values[other_name]
            values[record.name] = total
        else:  # pragma: no cover - defensive
            raise RecoveryError(f"Unknown elimination mode {record.mode!r}")

    missing = set(state.var_names) - set(values.keys())
    if missing:
        raise RecoveryError(f"Failed to recover values for: {sorted(missing)}")

    # Recompute the objective in original space from first principles
    # (original objective coefficients + offset) rather than trusting the
    # reduced solver's reported value, so recover_solution is self-checking
    # against silent presolve bugs.
    minimize = state.original_sense.value == "min"
    sign = 1.0 if minimize else -1.0
    obj = state._original_objective_offset_for_report
    for name, coeff in state._original_objective_for_report.items():
        obj += coeff * values[name]

    names = tuple(state.var_names)
    primal = tuple(values[name] for name in names)
    stats = replace(reduced_solution.statistics, primal_objective=float(obj))
    return SolverResult(status=reduced_solution.status, variable_names=names,
                        constraint_names=tuple(state.row_names),
                        primal=primal, statistics=stats)
