"""Sparse-matrix, two-phase revised simplex implementation."""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from nirnaya_core import (
    Model, ObjectiveSense, ProblemData, SolverResult, SolverStatus,
    OptimizationStatistics, VariableType, SparseMatrix,
)


def _model_data(model: Model | ProblemData) -> ProblemData:
    if isinstance(model, Model) and not model.variables:
        # Presolve may prove every variable fixed/eliminated. That reduced
        # constant LP is valid even though the modeling API rejects an
        # originally empty model, so construct its numerical snapshot here.
        if model.objective is None:
            raise ValueError("model has no objective")
        data = ProblemData(variable_names=(), constraint_names_eq=(), constraint_names_ub=(),
            sense=model.objective.sense, c=np.zeros(0, dtype=model.config.dtype),
            objective_constant=model.objective.constant,
            a_eq=SparseMatrix.empty(0, 0, dtype=model.config.dtype), b_eq=np.zeros(0, dtype=model.config.dtype),
            a_ub=SparseMatrix.empty(0, 0, dtype=model.config.dtype), b_ub=np.zeros(0, dtype=model.config.dtype),
            lower=np.zeros(0, dtype=model.config.dtype), upper=np.zeros(0, dtype=model.config.dtype),
            var_types=(), config=model.config)
    else:
        data = model.build() if isinstance(model, Model) else model
    if not isinstance(data, ProblemData):
        raise TypeError("solve expects nirnaya_core.Model or ProblemData")
    if any(t is not VariableType.CONTINUOUS for t in data.var_types):
        raise ValueError("nirnaya-solver supports continuous variables only")
    return data


def _finite_bounds(data: ProblemData) -> tuple[np.ndarray, np.ndarray]:
    cfg = data.config
    lo, hi = data.lower.astype(data.config.dtype, copy=True), data.upper.astype(data.config.dtype, copy=True)
    for i in range(data.n_variables):
        if cfg.is_effectively_infinite(lo[i]): lo[i] = -np.inf
        if cfg.is_effectively_infinite(hi[i]): hi[i] = np.inf
    return lo, hi


class _BasisFactorization:
    """Sparse LU plus bounded product-form eta updates for a changing basis.

    For a pivot at basis row ``r`` with direction ``d = B^-1 a_enter``, the
    new basis is ``B E`` where ``E`` is identity with column ``r`` replaced
    by ``d``. FTRAN applies ``E^-1`` updates in insertion order; BTRAN
    applies ``E^-T`` updates in reverse order before the base transpose
    solve. The chain is periodically refactorized and checked against the
    current sparse basis to bound accumulated roundoff.
    """

    REFACTOR_INTERVAL = 32
    RESIDUAL_CHECK_INTERVAL = 16

    def __init__(self, A, basis, cfg, metrics=None):
        self.A = A
        self.cfg = cfg
        self.metrics = metrics
        self.etas: list[tuple[int, np.ndarray]] = []
        self.work = np.empty(A.shape[0], dtype=A.dtype)
        self._last_checked_updates = 0
        self.lu = None
        self.refactor(basis)

    def refactor(self, basis) -> None:
        started = time.perf_counter()
        self.lu = spla.splu(self.A[:, basis].tocsc())
        if self.metrics is not None:
            self.metrics["factorization_seconds"] += time.perf_counter() - started
            self.metrics["factorization_count"] += 1
        self.etas.clear()
        self._last_checked_updates = 0

    def solve(self, rhs, trans="N"):
        update_started = time.perf_counter()
        if trans in ("T", "H"):
            transformed = np.array(rhs, dtype=self.A.dtype, copy=True)
            for row, direction in reversed(self.etas):
                old = transformed[row]
                transformed[row] = (old - (float(np.dot(direction, transformed))
                    - direction[row] * old)) / direction[row]
            if self.metrics is not None:
                self.metrics["basis_update_seconds"] += time.perf_counter() - update_started
            solve_started = time.perf_counter()
            answer = self.lu.solve(transformed, trans=trans)
        else:
            solve_started = time.perf_counter()
            answer = self.lu.solve(rhs)
            if self.metrics is not None:
                self.metrics["factor_solve_seconds"] += time.perf_counter() - solve_started
            update_started = time.perf_counter()
            for row, direction in self.etas:
                theta = answer[row] / direction[row]
                np.multiply(direction, theta, out=self.work)
                np.subtract(answer, self.work, out=answer)
                answer[row] = theta
            if self.metrics is not None:
                self.metrics["basis_update_seconds"] += time.perf_counter() - update_started
            return answer
        if self.metrics is not None:
            self.metrics["factor_solve_seconds"] += time.perf_counter() - solve_started
        return answer

    def needs_residual_check(self) -> bool:
        return len(self.etas) - self._last_checked_updates >= self.RESIDUAL_CHECK_INTERVAL

    def check_residuals(self, basis, xb, y, b, basic_costs) -> bool:
        """Refactor if update-chain residuals exceed existing solver tolerances."""
        started = time.perf_counter()
        B = self.A[:, basis].tocsc()
        primal_error = float(np.max(np.abs(B @ xb - b), initial=0.0))
        dual_error = float(np.max(np.abs(B.T @ y - basic_costs), initial=0.0))
        self._last_checked_updates = len(self.etas)
        if self.metrics is not None:
            self.metrics["basis_residual_check_seconds"] += time.perf_counter() - started
        if primal_error > self.cfg.feasibility_tol or dual_error > self.cfg.optimality_tol:
            self.refactor(basis)
            return True
        return False

    def pivot(self, row: int, direction, new_basis) -> None:
        started = time.perf_counter()
        pivot = float(direction[row])
        scale = max(1.0, float(np.max(np.abs(direction), initial=0.0)))
        unsafe_pivot = abs(pivot) <= max(100.0 * self.cfg.pivot_tol, 1e-12) * scale
        if unsafe_pivot or len(self.etas) + 1 >= self.REFACTOR_INTERVAL:
            self.refactor(new_basis)
        else:
            self.etas.append((row, np.array(direction, dtype=self.A.dtype, copy=True)))
            if self.metrics is not None:
                self.metrics["basis_update_count"] += 1
        if self.metrics is not None:
            self.metrics["basis_update_seconds"] += time.perf_counter() - started


def _solve_standard(A, b, c, cfg, deadline, iteration_limit, initial_basis=None, eligible=None,
                    metrics=None, phase="phase"):
    """Primal revised simplex on Ax=b, x>=0 with deterministic Bland pivots."""
    dtype = cfg.dtype
    m, n = A.shape
    if m == 0:
        if np.any(c < -cfg.optimality_tol): return SolverStatus.UNBOUNDED, np.zeros(n, dtype=dtype), [], 0
        return SolverStatus.OPTIMAL, np.zeros(n, dtype=dtype), [], 0
    basis = np.asarray(initial_basis if initial_basis is not None else range(n-m, n), dtype=np.intp)
    # Pricing repeatedly multiplies by A transpose; keep its CSR index
    # structure instead of recreating a transpose wrapper in every iteration.
    AT = A.T.tocsr()
    basic_mask = np.zeros(n, dtype=bool)
    basic_mask[basis] = True
    allowed_mask = np.ones(n, dtype=bool) if eligible is None else np.isin(np.arange(n), np.fromiter(eligible, dtype=int))
    total = 0
    factor = None
    while total < iteration_limit:
        if deadline is not None and time.monotonic() >= deadline:
            return SolverStatus.TIME_LIMIT, None, basis, total
        try:
            if factor is None:
                factor = _BasisFactorization(A, basis, cfg, metrics)
            xb = factor.solve(b)
            y = factor.solve(c[basis], trans="T")
            if factor.needs_residual_check() and factor.check_residuals(basis, xb, y, b, c[basis]):
                xb = factor.solve(b)
                y = factor.solve(c[basis], trans="T")
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return SolverStatus.NUMERICAL_ERROR, None, basis, total
        if not np.all(np.isfinite(xb)) or not np.all(np.isfinite(y)):
            return SolverStatus.NUMERICAL_ERROR, None, basis, total
        if np.min(xb, initial=0.0) < -cfg.feasibility_tol:
            return SolverStatus.NUMERICAL_ERROR, None, basis, total
        t0 = time.perf_counter()
        reduced = c - np.asarray(AT @ y).ravel()
        improving = (reduced < -cfg.optimality_tol) & ~basic_mask
        improving &= allowed_mask
        candidates_entering = np.flatnonzero(improving)
        if metrics is not None: metrics["pricing_seconds"] += time.perf_counter() - t0
        if candidates_entering.size == 0:
            x = np.zeros(n, dtype=dtype); x[basis] = np.maximum(xb, 0)
            return SolverStatus.OPTIMAL, x, basis, total
        # Bland's rule: choose the smallest-index eligible entering variable.
        entering = int(candidates_entering[0])
        try:
            d = factor.solve(A[:, entering].toarray().ravel())
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return SolverStatus.NUMERICAL_ERROR, None, basis, total
        t0 = time.perf_counter()
        eligible_rows = np.flatnonzero(d > cfg.pivot_tol)
        if not eligible_rows.size:
            return SolverStatus.UNBOUNDED, None, basis, total
        ratios = np.maximum(xb[eligible_rows], 0.0) / d[eligible_rows]
        theta = float(np.min(ratios))
        tied = eligible_rows[ratios <= theta + cfg.feasibility_tol]
        leave_row = int(tied[np.argmin(basis[tied])])
        if metrics is not None: metrics["ratio_test_seconds"] += time.perf_counter() - t0
        basic_mask[basis[leave_row]] = False
        basis[leave_row] = entering
        basic_mask[entering] = True
        try:
            factor.pivot(leave_row, d, basis)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return SolverStatus.NUMERICAL_ERROR, None, basis, total
        total += 1
    return SolverStatus.ITERATION_LIMIT, None, basis, total


def _remove_artificial_basics(A, b, basis, artificial_start, cfg, deadline, iteration_limit):
    """Pivot zero-valued Phase-I artificials out or remove redundant rows.

    Phase-II artificial columns must not remain basic: with zero Phase-II
    costs they could otherwise increase and mask a violation of the original
    equations. Degenerate pivots preserve primal feasibility because each
    outgoing artificial has value zero.
    """
    A = A.tocsc()
    basis = np.asarray(basis, dtype=np.intp).copy()
    b = np.asarray(b).copy()
    iterations = 0
    while np.any(basis >= artificial_start):
        if iterations >= iteration_limit:
            return SolverStatus.ITERATION_LIMIT, A, b, basis, iterations
        if deadline is not None and time.monotonic() >= deadline:
            return SolverStatus.TIME_LIMIT, A, b, basis, iterations
        row = int(np.flatnonzero(basis >= artificial_start)[0])
        try:
            factor = spla.splu(A[:, basis].tocsc())
            xb = factor.solve(b)
            if abs(xb[row]) > cfg.feasibility_tol:
                return SolverStatus.NUMERICAL_ERROR, A, b, basis
            unit = np.zeros(A.shape[0], dtype=A.dtype)
            unit[row] = 1.0
            tableau_row = np.asarray(A.T.tocsr() @ factor.solve(unit, trans="T")).ravel()
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return SolverStatus.NUMERICAL_ERROR, A, b, basis
        nonbasic_allowed = np.ones(A.shape[1], dtype=bool)
        nonbasic_allowed[basis] = False
        nonbasic_allowed[artificial_start:] = False
        candidates = np.flatnonzero(nonbasic_allowed & (np.abs(tableau_row) > cfg.pivot_tol))
        if candidates.size:
            # The first eligible column is a deterministic degenerate pivot.
            basis[row] = int(candidates[0])
        else:
            # With no non-artificial pivot in this equation, the row is
            # redundant in the feasible system and can be removed safely.
            keep = np.ones(A.shape[0], dtype=bool)
            keep[row] = False
            A = A[keep, :].tocsc()
            b = b[keep]
            basis = np.delete(basis, row)
        iterations += 1
    return SolverStatus.OPTIMAL, A, b, basis, iterations


def solve(model: Model | ProblemData, *, max_iterations: Optional[int] = None,
          iteration_limit: Optional[int] = None, time_limit: Optional[float] = None,
          feasibility_tol: Optional[float] = None, optimality_tol: Optional[float] = None,
          bound_tol: Optional[float] = None, pivot_tol: Optional[float] = None,
          zero_tol: Optional[float] = None, backend: str = "cpu",
          device: Optional[str] = None, **unknown_options) -> SolverResult:
    """Solve a continuous LP. Backend is CPU only; GPU requests fail clearly."""
    started = time.monotonic()
    data = _model_data(model)
    if unknown_options:
        raise TypeError(f"Unknown solver option(s): {sorted(unknown_options)}")
    if backend not in ("cpu", "auto"):
        raise ValueError("This solver release implements CPU execution only")
    overrides = {k: v for k, v in {
        "feasibility_tol": feasibility_tol, "optimality_tol": optimality_tol,
        "bound_tol": bound_tol, "pivot_tol": pivot_tol, "zero_tol": zero_tol,
    }.items() if v is not None}
    if overrides:
        data = ProblemData(**{**data.__dict__, "config": data.config.with_overrides(**overrides)})
    cfg, dtype = data.config, data.config.dtype
    limit = iteration_limit if iteration_limit is not None else max_iterations
    limit = cfg.max_iterations if limit is None else min(int(limit), cfg.max_iterations)
    seconds = time_limit if time_limit is not None else cfg.time_limit_seconds
    deadline = None if seconds is None else started + max(0.0, float(seconds))
    n = data.n_variables
    lo, hi = _finite_bounds(data)
    if np.any(hi < lo - cfg.bound_tol):
        return SolverResult(SolverStatus.INFEASIBLE, data.variable_names,
            data.constraint_names_eq + data.constraint_names_ub, message="Inconsistent variable bounds.")

    # Shift every variable by its finite lower bound; upper bounds become
    # ordinary <= rows. Original sparse rows are never converted to dense.
    shift = np.zeros(n, dtype=dtype)
    transform_rows, transform_cols, transform_values = [], [], []
    upper_rows, upper_cols = [], []
    transformed = 0
    for j in range(n):
        if np.isfinite(lo[j]):
            shift[j] = lo[j]
            transform_rows.append(j); transform_cols.append(transformed); transform_values.append(1.0)
            if np.isfinite(hi[j]):
                upper_rows.append(len(upper_rows)); upper_cols.append(transformed)
            transformed += 1
        elif np.isfinite(hi[j]):
            # x = upper - z represents a variable unbounded below.
            shift[j] = hi[j]
            transform_rows.append(j); transform_cols.append(transformed); transform_values.append(-1.0)
            transformed += 1
        else:
            # A free variable is represented as x = z_plus - z_minus.
            transform_rows.extend((j, j)); transform_cols.extend((transformed, transformed + 1))
            transform_values.extend((1.0, -1.0)); transformed += 2
    P = sp.csr_matrix((np.asarray(transform_values, dtype=dtype),
                       (transform_rows, transform_cols)), shape=(n, transformed))
    Aeq = (data.a_eq.data.astype(dtype, copy=True) @ P).tocsr()
    beq = data.b_eq.astype(dtype, copy=True) - np.asarray(data.a_eq.data @ shift, dtype=dtype)
    Aub = (data.a_ub.data.astype(dtype, copy=True) @ P).tocsr()
    bub = data.b_ub.astype(dtype, copy=True) - np.asarray(data.a_ub.data @ shift, dtype=dtype)
    if upper_cols:
        U = sp.csr_matrix((np.ones(len(upper_cols), dtype=dtype),
            (upper_rows, upper_cols)), shape=(len(upper_cols), transformed))
        Aub = sp.vstack([Aub, U], format="csr")
        finite_indices = [j for j in range(n) if np.isfinite(lo[j]) and np.isfinite(hi[j])]
        bub = np.concatenate([bub, hi[finite_indices] - lo[finite_indices]])
    # Convert all inequalities to equalities with nonnegative slack. Negative
    # RHS rows are multiplied by -1 so their artificial start is feasible.
    signs = np.where(bub < 0, -1.0, 1.0).astype(dtype)
    Aub = sp.diags(signs, format="csr") @ Aub
    bub *= signs
    m_eq, m_ub = Aeq.shape[0], Aub.shape[0]
    A0 = sp.vstack([Aeq, Aub], format="csr")
    b0 = np.concatenate([beq, bub]).astype(dtype, copy=False)
    # Equality artificials and inequality slacks form a feasible phase-I basis.
    ineq_sign = signs
    # Negative-RHS inequalities become >= after row reversal and therefore
    # receive a surplus column (-1) plus an artificial basic variable.
    slack = sp.vstack([sp.csr_matrix((m_eq, m_ub), dtype=dtype),
                       sp.diags(ineq_sign, format="csr")], format="csr")
    art_rows = list(range(m_eq)) + [m_eq + i for i in range(m_ub) if ineq_sign[i] < 0]
    art = sp.csr_matrix((np.ones(len(art_rows), dtype=dtype),
                         (art_rows, np.arange(len(art_rows)))),
                        shape=(m_eq + m_ub, len(art_rows)), dtype=dtype)
    A = sp.hstack([A0, slack, art], format="csc")
    art_start = transformed + m_ub
    art_col = {row: art_start + k for k, row in enumerate(art_rows)}
    basis = [art_col[r] if r in art_col else transformed + r - m_eq for r in range(m_eq + m_ub)]
    # With negative equality RHS, flip row so the artificial basic value is >=0.
    if m_eq:
        eq_sign = np.where(beq < 0, -1.0, 1.0).astype(dtype)
        A[:m_eq, :] = sp.diags(eq_sign) @ A[:m_eq, :]
        b0[:m_eq] *= eq_sign
    phase1 = np.zeros(A.shape[1], dtype=dtype); phase1[art_start:] = 1.0
    metrics = {"factorization_seconds": 0.0, "factorization_count": 0,
        "factor_solve_seconds": 0.0, "basis_update_seconds": 0.0,
        "basis_update_count": 0, "basis_residual_check_seconds": 0.0,
        "pricing_seconds": 0.0, "ratio_test_seconds": 0.0}
    phase1_started = time.perf_counter()
    status, x1, basis, it1 = _solve_standard(A, b0, phase1, cfg, deadline, limit, basis, metrics=metrics)
    metrics["phase1_iterations"] = it1
    metrics["phase1_seconds"] = time.perf_counter() - phase1_started
    iterations = it1
    metrics["phase2_seconds"] = 0.0
    if status is SolverStatus.OPTIMAL:
        if x1 is None or float(np.sum(x1[art_start:])) > cfg.feasibility_tol:
            status = SolverStatus.INFEASIBLE
        else:
            cleanup_started = time.perf_counter()
            status, A, b0, basis, cleanup_iterations = _remove_artificial_basics(
                A, b0, basis, art_start, cfg, deadline, max(0, limit - iterations))
            iterations += cleanup_iterations
            metrics["artificial_cleanup_seconds"] = time.perf_counter() - cleanup_started
        if status is SolverStatus.OPTIMAL:
            c = np.zeros(A.shape[1], dtype=dtype)
            sign = 1.0 if data.sense is ObjectiveSense.MINIMIZE else -1.0
            c[:transformed] = sign * np.asarray(P.T @ data.c, dtype=dtype).ravel()
            phase2_started = time.perf_counter()
            status, x2, basis, it2 = _solve_standard(A, b0, c, cfg, deadline,
                                                      max(0, limit-iterations), basis,
                                                      eligible=range(art_start), metrics=metrics)
            metrics["phase2_iterations"] = it2
            metrics["phase2_seconds"] = time.perf_counter() - phase2_started
            iterations += it2
            x1 = x2
        else:
            x1 = None
    else:
        x1 = None
    names = data.variable_names
    cons = data.constraint_names_eq + data.constraint_names_ub
    if status is not SolverStatus.OPTIMAL or x1 is None:
        return SolverResult(status, names, cons, statistics=OptimizationStatistics(
            iterations=iterations, solve_time_seconds=time.monotonic()-started, extra=metrics,
            solver_name="nirnaya-revised-simplex-cpu"),
            message=("Numerical issue while solving the initial feasibility basis." if status is SolverStatus.NUMERICAL_ERROR
                     else None))
    if deadline is not None and time.monotonic() >= deadline:
        return SolverResult(SolverStatus.TIME_LIMIT, data.variable_names,
            data.constraint_names_eq + data.constraint_names_ub,
            statistics=OptimizationStatistics(iterations=iterations,
                solve_time_seconds=time.monotonic()-started, extra=metrics,
                solver_name="nirnaya-revised-simplex-cpu"))
    # Recompute dual multipliers and reduced costs from the final basis.
    # Artificial columns are excluded from Phase II pricing and verification.
    try:
        final_factor = spla.splu(A[:, basis].tocsc())
        final_dual = final_factor.solve(c[basis], trans="T")
        final_reduced = c - np.asarray(A.T @ final_dual).ravel()
        nonbasic = np.ones(A.shape[1], dtype=bool)
        nonbasic[basis] = False
        eligible_mask = np.zeros(A.shape[1], dtype=bool)
        eligible_mask[:art_start] = True
        dual_error = float(np.max(np.maximum(-final_reduced[nonbasic & eligible_mask], 0.0), initial=0.0))
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        return SolverResult(SolverStatus.NUMERICAL_ERROR, data.variable_names,
            data.constraint_names_eq + data.constraint_names_ub,
            statistics=OptimizationStatistics(iterations=iterations,
                solve_time_seconds=time.monotonic()-started, extra=metrics,
                solver_name="nirnaya-revised-simplex-cpu"),
            message="Final basis optimality verification failed.")
    if not np.isfinite(dual_error) or dual_error > cfg.optimality_tol:
        return SolverResult(SolverStatus.NUMERICAL_ERROR, data.variable_names,
            data.constraint_names_eq + data.constraint_names_ub,
            statistics=OptimizationStatistics(iterations=iterations,
                solve_time_seconds=time.monotonic()-started, extra=metrics,
                max_dual_infeasibility=dual_error,
                solver_name="nirnaya-revised-simplex-cpu"),
            message="Final reduced-cost verification failed.")
    primal = shift + np.asarray(P @ x1[:transformed], dtype=dtype).ravel()
    eq_res = np.asarray(data.a_eq.matvec(primal) - data.b_eq)
    ub_res = np.asarray(data.a_ub.matvec(primal) - data.b_ub)
    violations = [np.max(np.abs(eq_res), initial=0.0), np.max(np.maximum(ub_res, 0), initial=0.0),
        np.max(np.maximum(lo-primal, 0), initial=0.0), np.max(np.maximum(primal-hi, 0), initial=0.0)]
    worst = float(max(violations))
    bound_error = float(max(violations[2:]))
    constraint_error = float(max(violations[:2]))
    objective = float(data.c @ primal + data.objective_constant)
    if (not (np.isfinite(objective) and np.all(np.isfinite(primal)))
            or constraint_error > cfg.feasibility_tol or bound_error > cfg.bound_tol):
        metrics["original_space_violations"] = {"equality": float(violations[0]),
            "inequality": float(violations[1]), "lower_bound": float(violations[2]),
            "upper_bound": float(violations[3])}
        return SolverResult(SolverStatus.NUMERICAL_ERROR, names, cons,
            statistics=OptimizationStatistics(iterations=iterations, solve_time_seconds=time.monotonic()-started, extra=metrics,
                max_primal_infeasibility=worst, max_dual_infeasibility=dual_error,
                solver_name="nirnaya-revised-simplex-cpu"),
            message="Independent original-space feasibility verification failed.")
    residuals = tuple(eq_res.tolist() + ub_res.tolist())
    return SolverResult(SolverStatus.OPTIMAL, names, cons, primal=primal.tolist(),
        constraint_residuals=residuals,
        statistics=OptimizationStatistics(iterations=iterations, solve_time_seconds=time.monotonic()-started, extra=metrics,
            primal_objective=objective, max_primal_infeasibility=worst,
            max_dual_infeasibility=dual_error,
            solver_name="nirnaya-revised-simplex-cpu"))
