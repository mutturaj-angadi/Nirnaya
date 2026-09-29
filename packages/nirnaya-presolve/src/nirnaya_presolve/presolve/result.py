"""PresolveResult: the full, public output of a presolve run.

Holds the reduced model, every mapping needed to go back to original
variable/constraint space, and the statistics/trace describing what
presolve did. `recover_solution` is the sanctioned way for a downstream
solver (Part 3) to turn a solution of `reduced_model` back into a solution
of the original model -- see docs/INTEGRATION_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from nirnaya_core import Model, SolverResult

from .postsolve import recover_solution as _recover_solution


@dataclass(frozen=True)
class EliminatedVariable:
    """Describes how a non-fixed eliminated variable's value is recovered
    from the values of the variable(s) it was expressed in terms of."""
    name: str
    mode: str  # "aggregation" | "multi_affine"
    relation: str  # human-readable, e.g. "x1 = 2.0 * x2 + -1.0"
    depends_on: tuple  # names of variables this one's value depends on


class PresolveResult:
    def __init__(self, _state, reduced_model: Optional[Model]):
        self._state = _state
        self.reduced_model = reduced_model

        self.infeasible: bool = _state.infeasible
        self.infeasible_reason: str = _state.infeasible_reason
        self.unbounded: bool = _state.unbounded
        self.unbounded_reason: str = _state.unbounded_reason

        self.statistics = _state.stats
        self.trace = _state.trace.records
        self.tolerances = _state.tol

        self.fixed_variables: Dict[str, float] = {}
        self.eliminated_variables: Dict[str, EliminatedVariable] = {}
        for rec in _state.eliminated:
            if rec.mode == "fixed":
                self.fixed_variables[rec.name] = rec.fixed_value
            elif rec.mode == "aggregation":
                self.eliminated_variables[rec.name] = EliminatedVariable(
                    name=rec.name,
                    mode="aggregation",
                    relation=f"{rec.name} = {rec.alpha} * {rec.other_name} + {rec.beta}",
                    depends_on=(rec.other_name,),
                )
            elif rec.mode == "multi_affine":
                terms_str = " + ".join(f"{c}*{n}" for n, c in rec.terms.items())
                self.eliminated_variables[rec.name] = EliminatedVariable(
                    name=rec.name,
                    mode="multi_affine",
                    relation=f"{rec.name} = {rec.beta} + {terms_str}",
                    depends_on=tuple(rec.terms.keys()),
                )

        active_var_names = [_state.var_names[c] for c in _state.active_col_indices()]
        active_row_names = [_state.row_names[r] for r in _state.active_row_indices()]

        # Variable/constraint identity is preserved by name across presolve
        # (presolve never renames), so the "mapping" from reduced to
        # original space is the identity on the surviving names; what
        # actually differs is *which* names survive, captured here.
        self.reduced_to_original_variables: Dict[str, str] = {
            n: n for n in active_var_names
        }
        self.original_to_reduced_variables: Dict[str, Optional[str]] = {
            n: (n if n in set(active_var_names) else None) for n in _state.var_names
        }
        self.reduced_to_original_constraints: Dict[str, str] = {
            n: n for n in active_row_names
        }
        self.original_to_reduced_constraints: Dict[str, Optional[str]] = {
            n: (n if n in set(active_row_names) else None) for n in _state.row_names
        }

        self.scaling: Dict[str, Dict[str, float]] = {
            "row_scale": {
                _state.row_names[r]: _state.row_scale[r]
                for r in range(len(_state.row_names))
                if _state.row_scale[r] != 1.0
            },
            "col_scale": {
                _state.var_names[c]: _state.col_scale[c]
                for c in range(len(_state.var_names))
                if _state.col_scale[c] != 1.0
            },
        }

    def recover_solution(self, reduced_solution: SolverResult) -> SolverResult:
        """Reconstruct a SolverResult in the ORIGINAL variable space.

        Raises `RecoveryError` if `reduced_solution` doesn't cover every
        variable of `reduced_model`, or if this result represents an
        infeasible/unbounded presolve outcome (there is no reduced_model
        to have solved in the first place).
        """
        from ..exceptions import RecoveryError

        if self.infeasible or self.unbounded:
            raise RecoveryError(
                "Cannot recover a solution: presolve determined the model is "
                f"{'infeasible' if self.infeasible else 'unbounded'} "
                f"({self.infeasible_reason or self.unbounded_reason})."
            )
        return _recover_solution(self._state, reduced_solution)

    def summary(self) -> str:
        if self.infeasible:
            return f"INFEASIBLE: {self.infeasible_reason}"
        if self.unbounded:
            return f"UNBOUNDED: {self.unbounded_reason}"
        return self.statistics.summary()

    def __repr__(self):
        return f"PresolveResult({self.summary()})"
