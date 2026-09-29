"""
`_PresolveState`: presolve's own internal, mutable, sparse working
representation of an LP.

This is *not* a duplicate of nirnaya-core's data model — it is a private
implementation detail of this package, used only between
`adapter.model_from_core` and `adapter.model_to_core`. Nothing outside
`nirnaya_presolve` ever sees it. Internally the objective is always kept in
minimization form (maximize is handled by negating objective coefficients
on the way in and back out) so every pass only has to reason about one
sense.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .trace import TraceLog
from .statistics import PresolveStatistics


class _EliminationRecord:
    """How to recover an eliminated variable's value in postsolve.

    Exactly one of three modes applies:
      * fixed:        value = fixed_value
      * aggregation:  value = alpha * value_of[other_name] + beta
      * multi_affine: value = beta + sum(coeff * value_of[other] for
                       other, coeff in terms.items())   (singleton-column
                       elimination, where the row had more than one other
                       variable)

    In all cases, any referenced "other" variable may itself have been
    eliminated earlier or later; postsolve resolves names in *reverse*
    elimination order, so by the time a record is resolved every variable
    it refers to has already been assigned a value.
    """

    __slots__ = ("name", "mode", "fixed_value", "alpha", "beta", "other_name",
                 "terms", "reason")

    def __init__(self, name, mode, fixed_value=None, alpha=None, beta=None,
                 other_name=None, terms=None, reason=""):
        assert mode in ("fixed", "aggregation", "multi_affine")
        self.name = name
        self.mode = mode
        self.fixed_value = fixed_value
        self.alpha = alpha
        self.beta = beta
        self.other_name = other_name
        self.terms = terms  # Dict[str, float], only for multi_affine
        self.reason = reason


class _PresolveState:
    def __init__(self, tolerances):
        self.tol = tolerances
        self.trace = TraceLog()
        self.stats = PresolveStatistics()

        # index-based sparse storage
        self.var_names: List[str] = []
        self.var_lower: List[float] = []
        self.var_upper: List[float] = []
        self.var_obj: List[float] = []          # always minimize-sense
        self.var_active: List[bool] = []
        self.var_is_integer: List[bool] = []

        self.row_names: List[str] = []
        self.row_sense: List[str] = []          # 'L' | 'G' | 'E'
        self.row_rhs: List[float] = []
        self.row_active: List[bool] = []

        # sparse: row index -> {col index: coeff}, col index -> {row index: coeff}
        self.rows: List[Dict[int, float]] = []
        self.cols: List[Dict[int, float]] = []

        # cumulative scaling factors, indexed like var_names / row_names.
        # x_original = x_scaled * col_scale[c]; row r's scaled coefficients
        # are row_scale[r] * original_coeff * col_scale[c].
        self.col_scale: List[float] = []
        self.row_scale: List[float] = []

        self.obj_offset: float = 0.0
        self.original_sense = None  # ObjectiveSense, remembered for output
        self.name: str = ""

        # Immutable snapshot of the *original* objective (in the model's
        # original sense, untouched by any presolve substitution), kept
        # purely so postsolve can recompute the objective value in
        # original space independently of whatever the reduced solver
        # reports -- a cheap self-check against presolve bugs.
        self._original_objective_for_report: Dict[str, float] = {}
        self._original_objective_offset_for_report: float = 0.0

        self.name_to_var: Dict[str, int] = {}
        self.name_to_row: Dict[str, int] = {}

        self.eliminated: "list[_EliminationRecord]" = []  # in elimination order
        self.fixed_original_values: Dict[str, float] = {}

        self.infeasible = False
        self.infeasible_reason = ""
        self.unbounded = False
        self.unbounded_reason = ""

    # ------------------------------------------------------------------
    # construction helpers (used only by the adapter, while building)
    # ------------------------------------------------------------------

    def add_variable(self, name, lower, upper, obj_coeff, is_integer=False):
        idx = len(self.var_names)
        self.var_names.append(name)
        self.var_lower.append(lower)
        self.var_upper.append(upper)
        self.var_obj.append(obj_coeff)
        self.var_active.append(True)
        self.var_is_integer.append(is_integer)
        self.cols.append({})
        self.col_scale.append(1.0)
        self.name_to_var[name] = idx
        return idx

    def add_row(self, name, coeffs: Dict[int, float], sense, rhs):
        idx = len(self.row_names)
        self.row_names.append(name)
        self.row_sense.append(sense)
        self.row_rhs.append(rhs)
        self.row_active.append(True)
        clean = {c: v for c, v in coeffs.items() if v != 0.0}
        self.rows.append(clean)
        for c, v in clean.items():
            self.cols[c][idx] = v
        self.row_scale.append(1.0)
        self.name_to_row[name] = idx
        return idx

    # ------------------------------------------------------------------
    # mutation helpers used by passes
    # ------------------------------------------------------------------

    def set_row_coeff(self, r, c, value):
        """Set (or remove, if ~0) the coefficient of column c in row r,
        keeping row/col adjacency in sync."""
        if abs(value) <= self.tol.coeff_zero_tol:
            self.rows[r].pop(c, None)
            self.cols[c].pop(r, None)
        else:
            self.rows[r][c] = value
            self.cols[c][r] = value

    def row_terms(self, r):
        return ((c, v) for c, v in self.rows[r].items() if self.var_active[c])

    def col_terms(self, c):
        return ((r, v) for r, v in self.cols[c].items() if self.row_active[r])

    def active_row_indices(self):
        return (r for r in range(len(self.row_names)) if self.row_active[r])

    def active_col_indices(self):
        return (c for c in range(len(self.var_names)) if self.var_active[c])

    def remove_row(self, r):
        if not self.row_active[r]:
            return
        self.row_active[r] = False
        for c in list(self.rows[r].keys()):
            self.cols[c].pop(r, None)

    def remove_col(self, c):
        if not self.var_active[c]:
            return
        self.var_active[c] = False
        for r in list(self.cols[c].keys()):
            self.rows[r].pop(c, None)

    def mark_infeasible(self, reason):
        if not self.infeasible:
            self.infeasible = True
            self.infeasible_reason = reason
            self.stats.infeasible = True
            self.stats.infeasible_reason = reason

    def mark_unbounded(self, reason):
        if not self.unbounded:
            self.unbounded = True
            self.unbounded_reason = reason
            self.stats.unbounded = True
            self.stats.unbounded_reason = reason

    def fix_variable(self, c, value, reason, tie_to_original_bounds_check=True):
        """Eliminate variable c by fixing it to `value`. Substitutes its
        contribution into the objective and every row it still appears in,
        then deactivates the column."""
        name = self.var_names[c]
        lo, hi = self.var_lower[c], self.var_upper[c]
        if tie_to_original_bounds_check:
            if value < lo - self.tol.infeasibility_tol or value > hi + self.tol.infeasibility_tol:
                self.mark_infeasible(
                    f"Fixing variable {name!r} to {value} violates its bounds [{lo}, {hi}]"
                )
                return

        obj_c = self.var_obj[c]
        self.obj_offset += obj_c * value

        for r, a in list(self.cols[c].items()):
            if not self.row_active[r]:
                continue
            self.row_rhs[r] -= a * value
            self.rows[r].pop(c, None)
        self.cols[c] = {}
        self.remove_col(c)

        self.eliminated.append(
            _EliminationRecord(name=name, mode="fixed", fixed_value=value, reason=reason)
        )
        self.fixed_original_values[name] = value

    def aggregate_variable(self, c, alpha, other_c, beta, reason):
        """Eliminate variable c via x_c = alpha * x_other + beta.

        Substitutes into the objective and every remaining row containing
        c (adding alpha * a_rc to the coefficient of `other_c` in each such
        row, and folding beta * a_rc into that row's rhs).
        """
        name = self.var_names[c]
        other_name = self.var_names[other_c]

        obj_c = self.var_obj[c]
        if obj_c != 0.0:
            self.var_obj[other_c] += alpha * obj_c
            self.obj_offset += beta * obj_c

        for r, a in list(self.cols[c].items()):
            if not self.row_active[r]:
                continue
            self.row_rhs[r] -= beta * a
            new_other_coeff = self.rows[r].get(other_c, 0.0) + alpha * a
            self.rows[r].pop(c, None)
            self.cols[c].pop(r, None)
            self.set_row_coeff(r, other_c, new_other_coeff)

        self.remove_col(c)

        self.eliminated.append(
            _EliminationRecord(
                name=name, mode="aggregation", alpha=alpha, beta=beta,
                other_name=other_name, reason=reason,
            )
        )

    def _record_singleton_column_relation(self, var_name, row_name, pivot, rhs, other_terms):
        """Record x_var = rhs/pivot - sum_k (coeff_k/pivot) * x_other_k for
        a singleton-column elimination (see passes.pass_singleton_column).
        `other_terms` maps *column index* -> coefficient (still valid at
        call time, before the row is removed)."""
        beta = rhs / pivot
        terms = {self.var_names[k]: -(coeff / pivot) for k, coeff in other_terms.items()}
        self.eliminated.append(
            _EliminationRecord(
                name=var_name, mode="multi_affine", beta=beta, terms=terms,
                reason=f"singleton column, row {row_name!r}",
            )
        )

    # ------------------------------------------------------------------

    def row_activity_bounds(self, r):
        """Return (min_activity, max_activity) of sum a_rc * x_c over
        active columns of row r, given current variable bounds. +-inf
        propagate naturally when a coefficient meets an infinite bound."""
        lo_sum = 0.0
        hi_sum = 0.0
        for c, a in self.row_terms(r):
            lb, ub = self.var_lower[c], self.var_upper[c]
            if a > 0:
                term_lo = a * lb
                term_hi = a * ub
            else:
                term_lo = a * ub
                term_hi = a * lb
            lo_sum += term_lo
            hi_sum += term_hi
        return lo_sum, hi_sum

    def n_active_vars(self):
        return sum(1 for _ in self.active_col_indices())

    def n_active_rows(self):
        return sum(1 for _ in self.active_row_indices())
