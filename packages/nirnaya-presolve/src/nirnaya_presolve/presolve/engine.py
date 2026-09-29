"""The Presolver: orchestrates all transformation passes to a fixed point
and packages the outcome into a PresolveResult."""

from __future__ import annotations

import time

from nirnaya_core import Model

from ..adapter import model_from_core, model_to_core
from .tolerances import PresolveTolerances
from .result import PresolveResult
from . import passes as P


# Structural passes run repeatedly to a fixed point. Order matters somewhat
# for efficiency (cheap/high-yield passes first) but not for correctness --
# each pass only ever tightens the feasible region's *representation*, never
# changes the feasible region itself (verified per-pass in passes.py
# docstrings), so any order converges to the same fixed point.
_STRUCTURAL_PASSES = [
    P.pass_empty_rows,
    P.pass_empty_columns,
    P.pass_fixed_variables,
    P.pass_singleton_rows,
    P.pass_bound_propagation,
    P.pass_doubleton_aggregation,
    P.pass_singleton_column,
    P.pass_redundant_constraints,
    P.pass_duplicate_constraints,
    P.pass_coefficient_cleanup,
]


class Presolver:
    """Runs the full presolve pipeline on an `nirnaya_core.Model`.

    Parameters
    ----------
    tolerances:
        A `PresolveTolerances` instance controlling how aggressively (and
        how safely) each pass behaves. Defaults are conservative.
    enable_scaling:
        Whether to apply row/column scaling as a final conditioning step.
        Scaling never changes feasibility; disable it only if the
        downstream solver already does its own scaling and you want to
        avoid doing it twice.
    passes:
        Optional override of the structural pass list (mainly for testing
        individual passes in isolation via the engine, or for a caller
        that wants to disable a specific reduction). Defaults to all
        structural passes.
    """

    def __init__(self, tolerances: PresolveTolerances = None, enable_scaling: bool = True,
                 passes=None):
        self.tolerances = tolerances or PresolveTolerances()
        self.enable_scaling = enable_scaling
        self.passes = list(passes) if passes is not None else list(_STRUCTURAL_PASSES)

    def run(self, model: Model) -> PresolveResult:
        start = time.monotonic()
        st = model_from_core(model, self.tolerances)

        rounds = 0
        for rounds in range(1, self.tolerances.max_rounds + 1):
            any_changed = False
            for pass_fn in self.passes:
                changed = pass_fn(st)
                any_changed = any_changed or changed
                if st.infeasible or st.unbounded:
                    break
            if st.infeasible or st.unbounded:
                break
            if not any_changed:
                break
        st.stats.rounds_executed = rounds

        if not st.infeasible and not st.unbounded and self.enable_scaling:
            P.pass_scaling(st)

        st.stats.reduced_num_variables = st.n_active_vars()
        st.stats.reduced_num_constraints = st.n_active_rows()
        st.stats.wall_time_seconds = time.monotonic() - start

        reduced_model = None
        if not st.infeasible and not st.unbounded:
            reduced_model = model_to_core(st, model.config)

        return PresolveResult(_state=st, reduced_model=reduced_model)
