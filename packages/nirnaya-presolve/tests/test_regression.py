"""Regression tests covering the required scenario categories."""
import pytest
from nirnaya_core import Sense, ObjectiveSense, Solution, SolveStatus
from nirnaya_presolve import Presolver, PresolveTolerances
from nirnaya_presolve.exceptions import RecoveryError
from helpers import var, con, model

INF = float("inf")


def test_simple_feasible_lp_reduces_and_recovers():
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 10)],
        constraints=[
            con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 15.0),
            con("c2", {"x": 1.0, "y": -1.0}, Sense.GE, -3.0),
        ],
        objective={"x": 1.0, "y": 1.0},
        sense=ObjectiveSense.MAXIMIZE,
    )
    result = Presolver().run(m)
    assert not result.infeasible
    assert not result.unbounded
    reduced_values = {n: 5.0 for n in result.reduced_model.variable_names()}
    sol = Solution(values=reduced_values, objective_value=10.0, status=SolveStatus.OPTIMAL)
    orig = result.recover_solution(sol)
    assert set(orig.values.keys()) == {"x", "y"}


def test_infeasible_lp_detected():
    m = model(
        variables=[var("x", 0, 5)],
        constraints=[con("c1", {"x": 1.0}, Sense.GE, 10.0)],
        objective={"x": 1.0},
    )
    result = Presolver().run(m)
    assert result.infeasible
    assert result.reduced_model is None
    assert result.statistics.infeasible


def test_infeasible_via_contradictory_equalities():
    m = model(
        variables=[var("x", 0, 100), var("y", 0, 100)],
        constraints=[
            con("c1", {"x": 1.0, "y": 1.0}, Sense.EQ, 10.0),
            con("c2", {"x": 1.0, "y": 1.0}, Sense.EQ, 20.0),
        ],
        objective={"x": 1.0},
    )
    result = Presolver().run(m)
    assert result.infeasible


def test_redundant_constraints_removed_in_statistics():
    m = model(
        variables=[var("x", 0, 5), var("y", 0, 5)],
        constraints=[
            con("c_binding", {"x": 1.0, "y": 1.0}, Sense.LE, 6.0),
            con("c_redundant", {"x": 1.0, "y": 1.0}, Sense.LE, 100.0),
        ],
        objective={"x": 1.0},
    )
    result = Presolver().run(m)
    assert not result.infeasible
    assert "c_redundant" not in result.reduced_model.constraint_names()
    assert result.statistics.rows_removed_redundant >= 1


def test_fixed_variables_reported_in_result():
    m = model(
        variables=[var("x", 4.0, 4.0), var("y", 0, 10), var("z", 0, 10)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0, "z": 1.0}, Sense.LE, 20.0)],
        objective={"x": 1.0, "y": 1.0, "z": 1.0},
    )
    result = Presolver().run(m)
    assert result.fixed_variables.get("x") == 4.0


def test_zero_row_all_zero_coefficients():
    m = model(
        variables=[var("x", 0, 10)],
        constraints=[
            con("c1", {"x": 1.0}, Sense.LE, 5.0),
            con("c_zero", {"x": 0.0}, Sense.LE, 1.0),
        ],
        objective={"x": 1.0},
    )
    # the shim drops exact-zero coefficients at construction, so c_zero
    # arrives at presolve as a genuinely empty row
    result = Presolver().run(m)
    assert not result.infeasible
    assert "c_zero" not in result.reduced_model.constraint_names()


def test_zero_column_unused_variable():
    m = model(
        variables=[var("x", 0, 10), var("unused", -5, 5)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0},
    )
    result = Presolver().run(m)
    assert not result.infeasible
    assert "unused" not in result.reduced_model.variable_names()
    assert result.fixed_variables["unused"] == 0.0


def test_scaled_problem_solves_and_recovers_consistently():
    m = model(
        variables=[var("x", 0, 100), var("y", 0, 100)],
        constraints=[con("c1", {"x": 1000.0, "y": 2000.0}, Sense.LE, 100000.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    result = Presolver(enable_scaling=True).run(m)
    assert not result.infeasible
    reduced_values = {n: 0.0 for n in result.reduced_model.variable_names()}
    sol = Solution(values=reduced_values, objective_value=0.0, status=SolveStatus.OPTIMAL)
    orig = result.recover_solution(sol)
    assert abs(orig.values["x"]) < 1e-6
    assert abs(orig.values["y"]) < 1e-6


def test_badly_scaled_problem_does_not_crash_or_misfire_infeasible():
    m = model(
        variables=[var("x", 0, 1e-3), var("y", 0, 1e3)],
        constraints=[con("c1", {"x": 1e8, "y": 1e-8}, Sense.LE, 50.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    result = Presolver(enable_scaling=True).run(m)
    assert not result.infeasible
    assert result.reduced_model is not None


def test_degenerate_equality_chain():
    # x == y, y == z, plus a binding constraint -- degenerate but feasible
    m = model(
        variables=[var("x", 0, 50), var("y", 0, 50), var("z", 0, 50)],
        constraints=[
            con("c1", {"x": 1.0, "y": -1.0}, Sense.EQ, 0.0),
            con("c2", {"y": 1.0, "z": -1.0}, Sense.EQ, 0.0),
            con("c3", {"z": 1.0}, Sense.LE, 7.0),
        ],
        objective={"x": 1.0, "y": 1.0, "z": 1.0},
    )
    result = Presolver().run(m)
    assert not result.infeasible
    reduced_values = {n: 0.0 for n in result.reduced_model.variable_names()}
    sol = Solution(values=reduced_values, objective_value=0.0, status=SolveStatus.OPTIMAL)
    orig = result.recover_solution(sol)
    assert set(orig.values.keys()) == {"x", "y", "z"}
    assert abs(orig.values["x"] - orig.values["y"]) < 1e-9
    assert abs(orig.values["y"] - orig.values["z"]) < 1e-9


def test_numerical_tolerance_boundary_just_inside_is_not_infeasible():
    tol = PresolveTolerances(infeasibility_tol=1e-6)
    m = model(
        variables=[var("x", 0, 5)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 5.0 + 1e-9)],  # noise-level violation
        objective={"x": 1.0},
    )
    # x <= 5 already, extra row is redundant-with-noise, not a real conflict
    result = Presolver(tolerances=tol).run(m)
    assert not result.infeasible


def test_numerical_tolerance_boundary_clearly_outside_is_infeasible():
    tol = PresolveTolerances(infeasibility_tol=1e-6)
    m = model(
        variables=[var("x", 0, 5)],
        constraints=[con("c1", {"x": 1.0}, Sense.GE, 5.1)],  # real violation, x<=5
        objective={"x": 1.0},
    )
    result = Presolver(tolerances=tol).run(m)
    assert result.infeasible


def test_solution_recovery_matches_manual_computation_end_to_end():
    # a is fixed; c is a free singleton column eliminated via row_fix_c
    # (c = 5 - b); d keeps b genuinely free afterward via a non-redundant
    # joint inequality, so b isn't *also* pinned by an empty-column fix.
    m = model(
        variables=[var("a", 2.0, 2.0), var("b", 0, 10), var("c", -INF, INF), var("d", 0, 10)],
        constraints=[
            con("row_fix_c", {"c": 1.0, "b": 1.0}, Sense.EQ, 5.0),   # c singleton, free
            con("row_bound_b", {"b": 1.0, "a": 1.0, "d": 1.0}, Sense.LE, 9.0),
        ],
        objective={"a": 1.0, "b": 2.0, "c": 0.5, "d": 0.0},
    )
    result = Presolver(enable_scaling=False).run(m)
    assert not result.infeasible
    assert set(result.reduced_model.variable_names()) == {"b", "d"}
    reduced_values = {"b": 1.0, "d": 2.0}  # satisfies b + d <= 7 (9 - a=2)
    sol = Solution(values=reduced_values, objective_value=0.0, status=SolveStatus.OPTIMAL)
    orig = result.recover_solution(sol)
    assert orig.values["a"] == 2.0
    assert orig.values["b"] == 1.0
    assert orig.values["d"] == 2.0
    assert abs(orig.values["c"] - (5.0 - 1.0)) < 1e-9  # c = 5 - b
    expected_obj = 1 * 2.0 + 2 * 1.0 + 0.5 * (5.0 - 1.0) + 0.0 * 2.0
    assert abs(orig.objective_value - expected_obj) < 1e-9
