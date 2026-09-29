"""Unit tests for SolverStatus, OptimizationStatistics, and SolverResult."""

import math

import pytest

from nirnaya_core.solution.result import OptimizationStatistics, SolverResult
from nirnaya_core.status.status import SolverStatus


class TestSolverStatus:
    def test_is_terminal_success_only_for_optimal(self):
        assert SolverStatus.OPTIMAL.is_terminal_success
        assert not SolverStatus.INFEASIBLE.is_terminal_success
        assert not SolverStatus.UNBOUNDED.is_terminal_success

    def test_is_conclusive(self):
        assert SolverStatus.OPTIMAL.is_conclusive
        assert SolverStatus.INFEASIBLE.is_conclusive
        assert SolverStatus.UNBOUNDED.is_conclusive
        assert not SolverStatus.ITERATION_LIMIT.is_conclusive
        assert not SolverStatus.NOT_SOLVED.is_conclusive


class TestOptimizationStatistics:
    def test_defaults_all_none(self):
        stats = OptimizationStatistics()
        assert stats.iterations is None
        assert stats.solve_time_seconds is None

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            OptimizationStatistics(primal_objective=math.nan)

    def test_negative_iterations_rejected(self):
        with pytest.raises(ValueError):
            OptimizationStatistics(iterations=-1)

    def test_round_trip(self):
        stats = OptimizationStatistics(iterations=10, primal_objective=42.0, solver_name="test")
        d = stats.to_dict()
        stats2 = OptimizationStatistics.from_dict(d)
        assert stats2 == stats


class TestSolverResult:
    def test_not_solved_default(self):
        result = SolverResult(status=SolverStatus.NOT_SOLVED)
        assert result.primal is None
        assert not result.is_optimal()

    def test_optimal_with_primal(self):
        result = SolverResult(
            status=SolverStatus.OPTIMAL,
            variable_names=["x", "y"],
            primal=[1.0, 2.0],
        )
        assert result.is_optimal()
        assert result.primal_value("x") == 1.0
        assert result.primal_value("y") == 2.0

    def test_primal_value_without_solution_raises(self):
        result = SolverResult(status=SolverStatus.NOT_SOLVED)
        with pytest.raises(ValueError):
            result.primal_value("x")

    def test_primal_value_unknown_name_raises(self):
        result = SolverResult(status=SolverStatus.OPTIMAL, variable_names=["x"], primal=[1.0])
        with pytest.raises(ValueError):
            result.primal_value("z")

    def test_primal_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            SolverResult(status=SolverStatus.OPTIMAL, variable_names=["x", "y"], primal=[1.0])

    def test_dual_values_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            SolverResult(
                status=SolverStatus.OPTIMAL,
                constraint_names=["c1", "c2"],
                dual_values=[1.0],
            )

    def test_reduced_costs_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            SolverResult(
                status=SolverStatus.OPTIMAL,
                variable_names=["x"],
                reduced_costs=[1.0, 2.0],
            )

    def test_constraint_residuals_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            SolverResult(
                status=SolverStatus.OPTIMAL,
                constraint_names=["c1"],
                constraint_residuals=[1.0, 2.0],
            )

    def test_infeasible_result_with_no_primal(self):
        result = SolverResult(status=SolverStatus.INFEASIBLE, message="no feasible point")
        assert result.primal is None
        assert result.status == SolverStatus.INFEASIBLE

    def test_round_trip(self):
        result = SolverResult(
            status=SolverStatus.OPTIMAL,
            variable_names=["x", "y"],
            constraint_names=["c1"],
            primal=[1.0, 2.0],
            dual_values=[0.5],
            reduced_costs=[0.0, 0.1],
            constraint_residuals=[0.0],
            statistics=OptimizationStatistics(iterations=3),
            message="ok",
        )
        d = result.to_dict()
        result2 = SolverResult.from_dict(d)
        assert result2.status == result.status
        assert result2.primal == result.primal
        assert result2.dual_values == result.dual_values
        assert result2.statistics == result.statistics
