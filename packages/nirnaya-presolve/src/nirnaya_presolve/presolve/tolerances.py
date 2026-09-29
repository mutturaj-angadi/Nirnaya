"""
Numerical tolerances used throughout presolve.

Every transformation that could be affected by floating-point noise is
gated by one of these tolerances rather than an exact (==, <, >) comparison
against zero. Defaults are deliberately conservative (i.e. biased toward
*not* applying a reduction, and *not* declaring infeasibility, when a
comparison is close to the boundary) — see the docstring on each field for
the numerical assumption it encodes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresolveTolerances:
    #: Two floats are considered equal if their absolute difference is
    #: below this. Used for lower == upper (fixed variable) checks, and
    #: general "is this essentially zero" checks on values (not coefficients).
    feasibility_tol: float = 1e-8

    #: Used specifically when comparing a variable's lower and upper bound
    #: to decide whether the variable is "fixed". Slightly looser than
    #: feasibility_tol is deliberately *not* used here — fixing must be
    #: conservative, since it is an irreversible elimination.
    bound_tol: float = 1e-9

    #: A matrix coefficient with absolute value below this (in absolute
    #: terms, not relative) is treated as structurally zero when deciding
    #: row/column activity (e.g. empty row/column detection, singleton
    #: detection). Does NOT by itself delete coefficients from the model;
    #: see coeff_drop_tol for that.
    coeff_zero_tol: float = 1e-11

    #: Coefficient-cleanup pass: a coefficient a_ij may be dropped from row i
    #: only if, for every value x_j can take within its current bounds,
    #: |a_ij * x_j| <= coeff_drop_tol. This bounds the worst-case constraint
    #: violation introduced by dropping the term. Variables with an infinite
    #: bound on the relevant side are never eligible (worst case is
    #: unbounded), so cleanup never silently hides an unbounded direction.
    coeff_drop_tol: float = 1e-10

    #: A constraint is declared violated (and the model infeasible) only if
    #: the violation exceeds this. Applied to: activity-bound infeasibility,
    #: bound-propagation producing lower > upper, and duplicate/redundant
    #: constraint conflicts. Kept looser than coeff_zero_tol on purpose:
    #: an actual infeasibility call is expensive to get wrong (false
    #: positives kill otherwise-solvable models), so we require a real,
    #: not-explainable-by-noise violation.
    infeasibility_tol: float = 1e-7

    #: A constraint is declared redundant (always satisfied, safe to drop)
    #: only if its activity bound clears the rhs by at least this margin.
    redundancy_tol: float = 1e-9

    #: Candidate-row hash bucketing tolerance after coefficient
    #: normalization. Candidate rows are separately confirmed proportional
    #: to machine precision before any reduction is applied.
    duplicate_rel_tol: float = 1e-9

    #: Scaling: after geometric-mean scaling, coefficients are clipped to
    #: this [min, max] band relative to 1.0 to avoid scaling factors that
    #: are themselves numerically extreme. Purely a scaling-quality knob;
    #: never affects feasibility.
    scale_min: float = 1e-4
    scale_max: float = 1e4

    #: Presolve passes iterate to a fixed point; this bounds the number of
    #: full rounds to guarantee termination even on pathological inputs.
    max_rounds: int = 50

    #: Doubleton-equality aggregation and singleton-column elimination
    #: divide by a pivot coefficient; a pivot with absolute value below this
    #: is treated as (numerically) singular and the reduction is skipped
    #: rather than risking catastrophic cancellation.
    min_pivot: float = 1e-8
