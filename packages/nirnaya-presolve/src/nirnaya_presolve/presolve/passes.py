"""
Individual presolve transformations.

Every function here takes the internal `_PresolveState` and mutates it in
place, returning True if it changed anything (so the engine's fixed-point
loop knows whether to keep iterating). Every mutation that removes/fixes/
aggregates something is (a) logged to `st.trace` and (b) reflected in
`st.stats`. Infeasibility is signalled via `st.mark_infeasible(...)` and
callers should stop iterating once `st.infeasible` is set.

Internally the objective is always in *minimize* form (see adapter.py), so
these passes don't need to special-case objective sense.
"""

from __future__ import annotations

import sys
from typing import Dict

from .state import _PresolveState


# ---------------------------------------------------------------------
# 1. Empty row detection
# ---------------------------------------------------------------------

def pass_empty_rows(st: _PresolveState) -> bool:
    """A row with no active nonzero coefficients is either trivially
    satisfied (0 meets rhs within tolerance -> redundant, drop it) or a
    certificate of infeasibility (0 cannot meet rhs -> infeasible)."""
    changed = False
    for r in list(st.active_row_indices()):
        if any(True for _ in st.row_terms(r)):
            continue
        rhs = st.row_rhs[r]
        sense = st.row_sense[r]
        tol = st.tol.infeasibility_tol
        ok = (
            (sense == "E" and abs(rhs) <= tol)
            or (sense == "L" and rhs >= -tol)
            or (sense == "G" and rhs <= tol)
        )
        name = st.row_names[r]
        if ok:
            st.remove_row(r)
            st.stats.rows_removed_empty += 1
            st.trace.add(
                "empty_row",
                f"Row {name!r} has no active variables and is trivially "
                f"satisfied (0 {sense} {rhs}); removed as redundant.",
                constraints=(name,),
            )
            changed = True
        else:
            st.mark_infeasible(
                f"Row {name!r} has no active variables but requires "
                f"0 {sense} {rhs}, which is violated."
            )
            return True
    return changed


# ---------------------------------------------------------------------
# 2. Empty column detection
# ---------------------------------------------------------------------

def pass_empty_columns(st: _PresolveState) -> bool:
    """A variable that appears in no active row is only constrained by its
    own bounds and objective coefficient. If its objective coefficient is
    zero, any feasible value does equally well: fix it at a finite bound
    (preferring 0 if 0 is within bounds, for a "clean" solution), or 0 if
    both bounds are infinite. If its objective coefficient is nonzero, the
    minimizing choice runs the variable to +-infinity in the improving
    direction unless a finite bound stops it (in which case fix it there);
    if no finite bound stops it, the model is unbounded."""
    changed = False
    for c in list(st.active_col_indices()):
        if any(True for _ in st.col_terms(c)):
            continue
        name = st.var_names[c]
        lo, hi = st.var_lower[c], st.var_upper[c]
        obj = st.var_obj[c]

        if obj == 0.0:
            if lo <= 0.0 <= hi:
                value = 0.0
            elif lo > float("-inf"):
                value = lo
            elif hi < float("inf"):
                value = hi
            else:
                value = 0.0  # fully free, zero-cost: any value works
            reason = "empty column, zero objective coefficient: value is arbitrary"
        elif obj > 0.0:
            # minimizing: want x as small as possible
            if lo > float("-inf"):
                value = lo
                reason = "empty column, positive objective coefficient: minimized at lower bound"
            else:
                st.mark_unbounded(
                    f"Variable {name!r} has no constraints, a positive "
                    f"objective coefficient, and no finite lower bound: "
                    f"objective is unbounded below."
                )
                return True
        else:
            # obj < 0: minimizing wants x as large as possible
            if hi < float("inf"):
                value = hi
                reason = "empty column, negative objective coefficient: minimized at upper bound"
            else:
                st.mark_unbounded(
                    f"Variable {name!r} has no constraints, a negative "
                    f"objective coefficient, and no finite upper bound: "
                    f"objective is unbounded below."
                )
                return True

        st.stats.variables_eliminated_empty_column += 1
        st.trace.add(
            "empty_column",
            f"Variable {name!r} appears in no active constraint; fixed to "
            f"{value} ({reason}).",
            variables=(name,),
            detail={"value": value, "reason": reason},
        )
        st.fix_variable(c, value, reason=reason, tie_to_original_bounds_check=False)
        changed = True
    return changed


# ---------------------------------------------------------------------
# 3. Fixed-variable elimination
# ---------------------------------------------------------------------

def pass_fixed_variables(st: _PresolveState) -> bool:
    """lower == upper (within bound_tol) => the variable's value is
    determined; substitute it everywhere and remove the column."""
    changed = False
    for c in list(st.active_col_indices()):
        lo, hi = st.var_lower[c], st.var_upper[c]
        if lo == float("-inf") or hi == float("inf"):
            continue
        if abs(hi - lo) <= st.tol.bound_tol:
            name = st.var_names[c]
            value = (lo + hi) / 2.0
            st.stats.variables_fixed += 1
            st.trace.add(
                "fixed_variable",
                f"Variable {name!r} has lower == upper == {value} "
                f"(within tolerance); fixed and eliminated.",
                variables=(name,),
                detail={"value": value},
            )
            st.fix_variable(c, value, reason="lower bound equals upper bound")
            changed = True
            if st.infeasible:
                return True
    return changed


# ---------------------------------------------------------------------
# 4. Singleton row -> bound tightening
# ---------------------------------------------------------------------

def pass_singleton_rows(st: _PresolveState) -> bool:
    """A row with exactly one active nonzero coefficient a*x <=/>=/== rhs
    is really just a bound on x: x <=/>=/== rhs/a (flip sense if a < 0).
    Tighten the variable's bound accordingly and drop the row."""
    changed = False
    for r in list(st.active_row_indices()):
        terms = list(st.row_terms(r))
        if len(terms) != 1:
            continue
        c, a = terms[0]
        if abs(a) <= st.tol.coeff_zero_tol:
            continue
        rname = st.row_names[r]
        vname = st.var_names[c]
        implied = st.row_rhs[r] / a
        sense = st.row_sense[r]
        if a < 0:
            sense = {"L": "G", "G": "L", "E": "E"}[sense]

        old_lo, old_hi = st.var_lower[c], st.var_upper[c]
        new_lo, new_hi = old_lo, old_hi
        if sense in ("G", "E"):
            new_lo = max(new_lo, implied)
        if sense in ("L", "E"):
            new_hi = min(new_hi, implied)

        if new_lo > new_hi + st.tol.infeasibility_tol:
            st.mark_infeasible(
                f"Singleton row {rname!r} implies {vname!r} in "
                f"[{new_lo}, {new_hi}], an empty interval."
            )
            return True

        if new_lo != old_lo or new_hi != old_hi:
            st.stats.bounds_tightened += 1
        st.var_lower[c] = new_lo
        st.var_upper[c] = new_hi

        st.trace.add(
            "singleton_row",
            f"Row {rname!r} is a singleton on {vname!r}; converted to bound "
            f"[{new_lo}, {new_hi}] and removed.",
            variables=(vname,),
            constraints=(rname,),
            detail={"lower": new_lo, "upper": new_hi},
        )
        st.remove_row(r)
        st.stats.rows_removed_singleton += 1
        changed = True
    return changed


# ---------------------------------------------------------------------
# 5. Bound propagation (activity-based bound tightening)
# ---------------------------------------------------------------------

def pass_bound_propagation(st: _PresolveState) -> bool:
    """For each row and each variable in it, use the row's sense/rhs and
    the *other* variables' current bounds to derive the tightest possible
    bound on that variable, and tighten if it improves on the current one.

    Standard LP presolve formula: for row sum a_k x_k <= rhs (after
    flipping >= / == appropriately), isolate x_j:

        a_j x_j <= rhs - sum_{k != j} a_k x_k

    The right-hand side is minimized by taking, for each other term, its
    *minimum* possible value (i.e. using the activity lower bound with
    term j excluded), giving the tightest valid upper bound on a_j x_j.
    """
    changed = False
    tol = st.tol
    for r in list(st.active_row_indices()):
        sense = st.row_sense[r]
        rhs = st.row_rhs[r]
        terms = list(st.row_terms(r))
        if len(terms) < 2:
            continue  # singleton/empty rows handled by their own passes

        lo_sum, hi_sum = st.row_activity_bounds(r)

        # quick infeasibility check on the whole row
        if sense in ("L", "E") and lo_sum > rhs + tol.infeasibility_tol:
            st.mark_infeasible(
                f"Row {st.row_names[r]!r}: minimum possible activity "
                f"{lo_sum} exceeds rhs {rhs}."
            )
            return True
        if sense in ("G", "E") and hi_sum < rhs - tol.infeasibility_tol:
            st.mark_infeasible(
                f"Row {st.row_names[r]!r}: maximum possible activity "
                f"{hi_sum} is below required rhs {rhs}."
            )
            return True

        for c, a in terms:
            if abs(a) <= tol.coeff_zero_tol:
                continue
            lb, ub = st.var_lower[c], st.var_upper[c]
            if a > 0:
                term_lo = a * lb
                term_hi = a * ub
            else:
                term_lo = a * ub
                term_hi = a * lb
            rest_lo = lo_sum - term_lo   # min activity of all *other* terms
            rest_hi = hi_sum - term_hi   # max activity of all *other* terms

            if float("inf") in (abs(rest_lo), abs(rest_hi)) and (
                rest_lo in (float("inf"), float("-inf"))
                and rest_hi in (float("inf"), float("-inf"))
            ):
                # both sides unbounded -> this row cannot tighten x_c at all
                if rest_lo == float("-inf") and sense in ("L", "E"):
                    continue
                if rest_hi == float("inf") and sense in ("G", "E"):
                    continue

            new_lo, new_hi = lb, ub

            # sum <= rhs  =>  a*x_c <= rhs - rest_lo
            if sense in ("L", "E") and rest_lo > float("-inf"):
                bound_val = (rhs - rest_lo) / a
                if a > 0:
                    new_hi = min(new_hi, bound_val)
                else:
                    new_lo = max(new_lo, bound_val)

            # sum >= rhs  =>  a*x_c >= rhs - rest_hi
            if sense in ("G", "E") and rest_hi < float("inf"):
                bound_val = (rhs - rest_hi) / a
                if a > 0:
                    new_lo = max(new_lo, bound_val)
                else:
                    new_hi = min(new_hi, bound_val)

            if new_lo > new_hi + tol.infeasibility_tol:
                st.mark_infeasible(
                    f"Bound propagation from row {st.row_names[r]!r} implies "
                    f"{st.var_names[c]!r} in [{new_lo}, {new_hi}], empty."
                )
                return True

            new_lo = min(new_lo, new_hi) if new_lo > new_hi else new_lo

            improved = (new_lo > lb + tol.bound_tol) or (new_hi < ub - tol.bound_tol)
            if improved:
                st.var_lower[c] = new_lo
                st.var_upper[c] = new_hi
                st.stats.bounds_tightened += 1
                st.trace.add(
                    "bound_propagation",
                    f"Row {st.row_names[r]!r} tightens {st.var_names[c]!r} "
                    f"from [{lb}, {ub}] to [{new_lo}, {new_hi}].",
                    variables=(st.var_names[c],),
                    constraints=(st.row_names[r],),
                    detail={"old": [lb, ub], "new": [new_lo, new_hi]},
                )
                changed = True
    return changed


# ---------------------------------------------------------------------
# 6. Redundant constraint detection
# ---------------------------------------------------------------------

def pass_redundant_constraints(st: _PresolveState) -> bool:
    """If a row's activity bound already guarantees the constraint holds
    for *every* value the variables could take, the constraint carries no
    information and can be dropped. For inequality rows this means the
    binding side of the activity interval already clears rhs; for equality
    rows this would require the activity interval to have collapsed to a
    single point equal to rhs (extremely rare, but checked for
    completeness -- typically an equality row only ever gets removed by
    singleton/aggregation passes, not this one)."""
    changed = False
    tol = st.tol
    for r in list(st.active_row_indices()):
        terms = list(st.row_terms(r))
        if len(terms) < 2:
            continue
        sense = st.row_sense[r]
        rhs = st.row_rhs[r]
        lo_sum, hi_sum = st.row_activity_bounds(r)

        redundant = False
        if sense == "L" and hi_sum <= rhs + tol.redundancy_tol:
            redundant = True
        elif sense == "G" and lo_sum >= rhs - tol.redundancy_tol:
            redundant = True
        elif sense == "E" and abs(hi_sum - lo_sum) <= tol.redundancy_tol and \
                abs(lo_sum - rhs) <= tol.redundancy_tol:
            redundant = True

        if redundant:
            name = st.row_names[r]
            st.trace.add(
                "redundant_constraint",
                f"Row {name!r} is always satisfied given current variable "
                f"bounds (activity in [{lo_sum}, {hi_sum}], rhs {rhs}); removed.",
                constraints=(name,),
                detail={"activity_lo": lo_sum, "activity_hi": hi_sum, "rhs": rhs},
            )
            st.remove_row(r)
            st.stats.rows_removed_redundant += 1
            changed = True
    return changed


# ---------------------------------------------------------------------
# 7. Duplicate constraint handling
# ---------------------------------------------------------------------

def _row_signature(st, r):
    """A hashable, scale-invariant signature for a row's coefficient
    pattern: sorted (col, ratio-to-first-nonzero) pairs, used to group
    candidate-duplicate rows cheaply before doing a precise numeric
    comparison."""
    terms = sorted(st.row_terms(r), key=lambda t: t[0])
    if not terms:
        return None
    _, pivot = terms[0]
    return tuple((c, round(a / pivot, 9)) for c, a in terms)


def pass_duplicate_constraints(st: _PresolveState) -> bool:
    """Detect rows that are scalar multiples of each other (same variables,
    proportional coefficients) and merge them: for two rows r1 = k * r2,
    combine their senses/rhs into the single tightest equivalent
    constraint, or detect a direct contradiction as infeasibility."""
    changed = False
    tol = st.tol
    buckets: Dict[tuple, list] = {}
    for r in st.active_row_indices():
        sig = _row_signature(st, r)
        if sig is None:
            continue
        buckets.setdefault(sig, []).append(r)

    for sig, rows in buckets.items():
        if len(rows) < 2:
            continue
        # rows here share identical (col, ratio) signatures => truly
        # proportional with ratio 1 relative to each row's own pivot.
        # Compute each row's absolute scale k_r = coeff of first var.
        keep = rows[0]
        keep_terms = dict(st.row_terms(keep))
        keep_first_col = min(keep_terms)
        keep_pivot = keep_terms[keep_first_col]

        for other in rows[1:]:
            if not st.row_active[other] or not st.row_active[keep]:
                continue
            other_terms = dict(st.row_terms(other))
            other_first_col = min(other_terms)
            other_pivot = other_terms[other_first_col]
            k = other_pivot / keep_pivot  # other = k * keep (coefficient-wise)
            # The rounded signature is only a candidate bucket. Treating
            # merely close rows as identical can remove real feasible-region
            # information and, in particular, falsely declare consistent
            # equality systems infeasible. Merge only coefficient vectors
            # that are proportional to floating-point roundoff.
            if not _same_proportional_coefficients(keep_terms, other_terms, k):
                continue

            keep_rhs_scaled = st.row_rhs[keep] * k
            other_rhs = st.row_rhs[other]
            keep_sense = st.row_sense[keep]
            other_sense = st.row_sense[other]
            if k < 0:
                keep_sense = {"L": "G", "G": "L", "E": "E"}[keep_sense]

            # Opposing inequalities on the same form usually define a
            # nonzero interval, which cannot be represented by one row.
            # Preserve both rows; only a genuinely empty interval is
            # infeasible, and a tolerance-tight interval can become equality.
            if {keep_sense, other_sense} == {"L", "G"}:
                lo = keep_rhs_scaled if keep_sense == "G" else other_rhs
                hi = keep_rhs_scaled if keep_sense == "L" else other_rhs
                if lo > hi + tol.infeasibility_tol:
                    st.mark_infeasible(
                        f"Rows {st.row_names[keep]!r} and {st.row_names[other]!r} "
                        "are proportional but their bounds contradict each other."
                    )
                    return True
                if hi - lo > tol.infeasibility_tol:
                    continue

            # Now both constraints are expressed as: (k*keep_row) <sense> rhs
            # keep_sense/keep_rhs_scaled  vs  other_sense/other_rhs
            combined = _combine_duplicate(
                keep_sense, keep_rhs_scaled, other_sense, other_rhs, tol
            )
            rname_keep, rname_other = st.row_names[keep], st.row_names[other]
            if combined is None:
                st.mark_infeasible(
                    f"Rows {rname_keep!r} and {rname_other!r} are proportional "
                    f"but their bounds contradict each other."
                )
                return True

            new_sense_for_keep, new_rhs_for_keep = combined
            # translate back: new_rhs_for_keep is in the (k*keep) frame;
            # convert to keep's own frame by dividing by k
            final_rhs = new_rhs_for_keep / k
            final_sense = new_sense_for_keep
            if k < 0:
                final_sense = {"L": "G", "G": "L", "E": "E"}[final_sense]

            st.row_sense[keep] = final_sense
            st.row_rhs[keep] = final_rhs
            st.trace.add(
                "duplicate_constraint",
                f"Row {rname_other!r} is a scalar multiple (k={k}) of "
                f"{rname_keep!r}; merged into {rname_keep!r} and removed.",
                constraints=(rname_keep, rname_other),
                detail={"scale": k},
            )
            st.remove_row(other)
            st.stats.rows_removed_duplicate += 1
            changed = True
    return changed


def _same_proportional_coefficients(first, second, scale) -> bool:
    if first.keys() != second.keys():
        return False
    eps = 32.0 * sys.float_info.epsilon
    for col, value in first.items():
        expected = scale * value
        actual = second[col]
        if abs(actual - expected) > eps * max(abs(actual), abs(expected), 1e-300):
            return False
    return True


def _combine_duplicate(sense1, rhs1, sense2, rhs2, tol):
    """Combine two constraints of the *same* linear form (c^T x) that
    differ only in sense/rhs into a single equivalent constraint, or
    return None if they contradict. Assumes rhs1, rhs2 already refer to
    the same coefficient scaling."""
    t = tol.infeasibility_tol
    if sense1 == "E" and sense2 == "E":
        return ("E", rhs1) if abs(rhs1 - rhs2) <= t else None
    if sense1 == "E":
        ok = (sense2 == "L" and rhs1 <= rhs2 + t) or (sense2 == "G" and rhs1 >= rhs2 - t)
        return ("E", rhs1) if ok else None
    if sense2 == "E":
        ok = (sense1 == "L" and rhs2 <= rhs1 + t) or (sense1 == "G" and rhs2 >= rhs1 - t)
        return ("E", rhs2) if ok else None
    if sense1 == sense2 == "L":
        return ("L", min(rhs1, rhs2))
    if sense1 == sense2 == "G":
        return ("G", max(rhs1, rhs2))
    # one L one G -> becomes a tight equality if they meet, else stays as two
    # bounds (min L, max G); only contradictory if lower > upper
    lo = rhs1 if sense1 == "G" else rhs2
    hi = rhs1 if sense1 == "L" else rhs2
    if lo > hi + t:
        return None
    if abs(lo - hi) <= t:
        return ("E", (lo + hi) / 2.0)
    # Can't represent "L and G with a gap" as one row without losing
    # information (that's just... still two different bounds) — keep the
    # tighter of the two as a single-sense row is unsound, so refuse to
    # merge (caller should not have called us with a genuine two-sided
    # pair reduced to this case in practice, but be safe).
    return None


# ---------------------------------------------------------------------
# 8. Singleton column elimination (free variable, single equality row)
# ---------------------------------------------------------------------

def pass_singleton_column(st: _PresolveState) -> bool:
    """A variable that (a) appears in exactly one active row, (b) that row
    is an equality, and (c) the variable is completely free (lower=-inf,
    upper=+inf) can always be eliminated: solve the row for it and
    substitute. Because the variable is unbounded, no information is lost
    (there is no bound on it to turn into a new constraint on the rest of
    the row), which is what keeps this reduction unconditionally safe.
    Restricting to free variables (instead of the general singleton-column
    case) is a deliberate numerical-safety choice — see module docstring
    in docs/ASSUMED_CORE_API.md / INTEGRATION_CONTRACT.md for rationale."""
    changed = False
    for c in list(st.active_col_indices()):
        rows_here = list(st.col_terms(c))
        if len(rows_here) != 1:
            continue
        r, a = rows_here[0]
        if st.row_sense[r] != "E":
            continue
        if st.var_lower[c] != float("-inf") or st.var_upper[c] != float("inf"):
            continue
        if abs(a) < st.tol.min_pivot:
            continue

        vname = st.var_names[c]
        rname = st.row_names[r]
        # row: a*x_c + sum_{k!=c} a_k x_k == rhs
        # => x_c = rhs/a - sum_{k!=c} (a_k/a) x_k
        # This is a multi-term affine relation, not a 2-term aggregation,
        # so we substitute directly into the objective (only place x_c can
        # still appear, since it's a singleton column) and record a
        # multi-term relation for postsolve.
        obj_c = st.var_obj[c]
        rhs = st.row_rhs[r]
        other_terms = {k: v for k, v in st.rows[r].items() if k != c}

        if obj_c != 0.0:
            for k, ak in other_terms.items():
                st.var_obj[k] += -obj_c * (ak / a)
            st.obj_offset += obj_c * (rhs / a)

        st.trace.add(
            "singleton_column",
            f"Variable {vname!r} appears only in equality row {rname!r} and "
            f"is free; eliminated via {vname!r} = ({rhs}/{a}) - sum(other "
            f"coeffs/{a} * other var).",
            variables=(vname,),
            constraints=(rname,),
            detail={
                "row": rname,
                "pivot": a,
                "rhs": rhs,
                "other_terms": {st.var_names[k]: v for k, v in other_terms.items()},
            },
        )
        # record as a generic multi-term relation via a dedicated record type
        st._record_singleton_column_relation(vname, rname, a, rhs, other_terms)

        st.remove_row(r)
        st.stats.rows_removed_aggregation += 1
        st.remove_col(c)
        st.stats.variables_eliminated_singleton_column += 1
        changed = True
    return changed


# ---------------------------------------------------------------------
# 9. Doubleton equality aggregation
# ---------------------------------------------------------------------

def pass_doubleton_aggregation(st: _PresolveState) -> bool:
    """An equality row with exactly two active terms, a*x_i + b*x_j == rhs,
    lets us eliminate one variable in favor of the other:

        x_i = (rhs - b*x_j) / a = alpha * x_j + beta

    We eliminate whichever of the two has the larger-magnitude coefficient
    (dividing by the larger pivot is more numerically stable) and fold the
    row's implied bound on the eliminated variable into a tightened bound
    on the surviving variable, so no feasible-region information is lost."""
    changed = False
    tol = st.tol
    for r in list(st.active_row_indices()):
        if st.row_sense[r] != "E":
            continue
        terms = list(st.row_terms(r))
        if len(terms) != 2:
            continue
        (c1, a1), (c2, a2) = terms
        if abs(a1) < abs(a2):
            (c1, a1), (c2, a2) = (c2, a2), (c1, a1)
        if abs(a1) < tol.min_pivot:
            continue

        rname = st.row_names[r]
        n1, n2 = st.var_names[c1], st.var_names[c2]
        rhs = st.row_rhs[r]

        alpha = -a2 / a1
        beta = rhs / a1
        # x_c1 = alpha * x_c2 + beta

        # Fold x_c1's current bounds into an implied bound on x_c2:
        # lo1 <= alpha*x2 + beta <= hi1  (if alpha>0)   [flip if alpha<0]
        lo1, hi1 = st.var_lower[c1], st.var_upper[c1]
        lo2, hi2 = st.var_lower[c2], st.var_upper[c2]
        if alpha > 0:
            implied_lo2 = (lo1 - beta) / alpha if lo1 != float("-inf") else float("-inf")
            implied_hi2 = (hi1 - beta) / alpha if hi1 != float("inf") else float("inf")
        elif alpha < 0:
            implied_lo2 = (hi1 - beta) / alpha if hi1 != float("inf") else float("-inf")
            implied_hi2 = (lo1 - beta) / alpha if lo1 != float("-inf") else float("inf")
        else:
            # alpha == 0 means a2 == 0, contradicts len(terms)==2 with a2 nonzero
            implied_lo2, implied_hi2 = lo2, hi2

        new_lo2 = max(lo2, implied_lo2)
        new_hi2 = min(hi2, implied_hi2)
        if new_lo2 > new_hi2 + tol.infeasibility_tol:
            st.mark_infeasible(
                f"Doubleton row {rname!r} aggregating {n1!r} and {n2!r} "
                f"implies {n2!r} in an empty interval "
                f"[{new_lo2}, {new_hi2}]."
            )
            return True

        st.var_lower[c2] = new_lo2
        st.var_upper[c2] = new_hi2

        st.trace.add(
            "doubleton_aggregation",
            f"Equality row {rname!r}: eliminated {n1!r} = {alpha}*{n2!r} + "
            f"{beta}; bound of {n2!r} tightened to [{new_lo2}, {new_hi2}].",
            variables=(n1, n2),
            constraints=(rname,),
            detail={"alpha": alpha, "beta": beta, "other": n2},
        )

        st.remove_row(r)
        st.stats.rows_removed_aggregation += 1
        st.aggregate_variable(c1, alpha, c2, beta, reason=f"doubleton row {rname!r}")
        st.stats.variables_eliminated_aggregation += 1
        changed = True
    return changed


# ---------------------------------------------------------------------
# 10. Coefficient cleanup
# ---------------------------------------------------------------------

def pass_coefficient_cleanup(st: _PresolveState) -> bool:
    """Drop a coefficient a_rc from row r only if doing so can change that
    row's activity by no more than coeff_drop_tol for *every* value x_c
    could take -- i.e. only when x_c's bounds are both finite and
    max(|a_rc*lb|, |a_rc*ub|) <= coeff_drop_tol. This bounds the worst-case
    constraint violation introduced, and is skipped whenever either bound
    is infinite (an unbounded direction is never safe to ignore, no matter
    how small the coefficient)."""
    changed = False
    for r in list(st.active_row_indices()):
        for c, a in list(st.row_terms(r)):
            if a == 0.0:
                continue
            lb, ub = st.var_lower[c], st.var_upper[c]
            if lb == float("-inf") or ub == float("inf"):
                continue
            worst = max(abs(a * lb), abs(a * ub))
            if worst <= st.tol.coeff_drop_tol:
                st.set_row_coeff(r, c, 0.0)
                st.stats.coefficients_dropped += 1
                st.trace.add(
                    "coefficient_cleanup",
                    f"Dropped negligible coefficient of {st.var_names[c]!r} "
                    f"({a}) in row {st.row_names[r]!r}: worst-case impact "
                    f"{worst} <= tolerance.",
                    variables=(st.var_names[c],),
                    constraints=(st.row_names[r],),
                    detail={"coefficient": a, "worst_case_impact": worst},
                )
                changed = True
    return changed


# ---------------------------------------------------------------------
# 11. Row / column scaling
# ---------------------------------------------------------------------

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def pass_scaling(st: _PresolveState, rounds: int = 2) -> bool:
    """Simple iterative geometric-mean scaling: repeatedly rescale each
    active row, then each active column, by 1/sqrt(min_abs * max_abs) of
    its nonzero entries, clipped to [scale_min, scale_max]. Purely a
    numerical-conditioning step -- never changes feasibility, and factors
    are tracked in st.row_scale / st.col_scale so postsolve can undo it
    exactly. No-ops on an empty/trivial model."""
    tol = st.tol
    any_change = False
    for _ in range(rounds):
        round_changed = False
        for r in list(st.active_row_indices()):
            mags = [abs(a) for _, a in st.row_terms(r) if a != 0.0]
            if not mags:
                continue
            factor = _clip(1.0 / ((min(mags) * max(mags)) ** 0.5), tol.scale_min, tol.scale_max)
            if abs(factor - 1.0) < 1e-12:
                continue
            for c, a in list(st.rows[r].items()):
                st.rows[r][c] = a * factor
                st.cols[c][r] = a * factor
            st.row_rhs[r] *= factor
            st.row_scale[r] *= factor
            round_changed = True

        for c in list(st.active_col_indices()):
            mags = [abs(a) for _, a in st.col_terms(c) if a != 0.0]
            obj = st.var_obj[c]
            if obj != 0.0:
                mags.append(abs(obj))
            if not mags:
                continue
            factor = _clip(1.0 / ((min(mags) * max(mags)) ** 0.5), tol.scale_min, tol.scale_max)
            if abs(factor - 1.0) < 1e-12:
                continue
            for r, a in list(st.cols[c].items()):
                st.cols[c][r] = a * factor
                st.rows[r][c] = a * factor
            st.var_obj[c] *= factor
            if st.var_lower[c] != float("-inf"):
                st.var_lower[c] /= factor
            if st.var_upper[c] != float("inf"):
                st.var_upper[c] /= factor
            st.col_scale[c] *= factor
            round_changed = True

        any_change = any_change or round_changed
        if not round_changed:
            break

    if any_change:
        st.trace.add(
            "scaling",
            "Applied geometric-mean row/column scaling for numerical "
            "conditioning (feasibility-preserving; factors recorded for "
            "postsolve).",
            detail={
                "rows_scaled": sum(1 for f in st.row_scale if f != 1.0),
                "cols_scaled": sum(1 for f in st.col_scale if f != 1.0),
            },
        )
    return any_change
