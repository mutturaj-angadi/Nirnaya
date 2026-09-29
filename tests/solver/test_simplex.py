import numpy as np

from nirnaya_core import ConstraintSense, Model, NumericalConfig, ObjectiveSense, SolverStatus
from nirnaya_solver import solve


def make_model(sense=ObjectiveSense.MAXIMIZE, config=None):
    m = Model("smoke", config=config)
    m.add_variable("x", 0, 10)
    m.add_variable("y", 0, 10)
    m.add_constraint("capacity", {"x": 1, "y": 1}, ConstraintSense.LE, 4)
    m.set_objective(sense, {"x": 3, "y": 2}, constant=5)
    return m


def test_bounded_max_and_offset():
    result = solve(make_model())
    assert result.status is SolverStatus.OPTIMAL
    assert np.allclose(result.primal, [4, 0])
    assert result.statistics.primal_objective == 17


def test_equality_phase_one_and_minimization():
    m = Model("eq")
    m.add_variable("x", 0)
    m.add_variable("y", 0)
    m.add_constraint("balance", {"x": 1, "y": 1}, ConstraintSense.EQ, 3)
    m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1, "y": 2})
    result = solve(m)
    assert result.status is SolverStatus.OPTIMAL
    assert np.allclose(result.primal, [3, 0])


def test_infeasible_and_unbounded_are_distinguished():
    m = Model("infeasible")
    m.add_variable("x")
    m.add_constraint("lo", {"x": 1}, ConstraintSense.GE, 3)
    m.add_constraint("hi", {"x": 1}, ConstraintSense.LE, 1)
    m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1})
    assert solve(m).status is SolverStatus.INFEASIBLE
    u = Model("unbounded")
    u.add_variable("x")
    u.set_objective(ObjectiveSense.MINIMIZE, {"x": -1})
    assert solve(u).status is SolverStatus.UNBOUNDED


def test_iteration_limit_is_reported():
    m = Model("limit")
    m.add_variable("x")
    m.add_variable("y")
    m.add_constraint("sum", {"x": 1, "y": 1}, ConstraintSense.LE, 4)
    m.add_constraint("xcap", {"x": 1}, ConstraintSense.LE, 1)
    m.set_objective(ObjectiveSense.MAXIMIZE, {"x": 3, "y": 2})
    result = solve(m, iteration_limit=1)
    assert result.status is SolverStatus.ITERATION_LIMIT


def test_free_variable_is_split_into_nonnegative_directions():
    m = Model("free")
    m.add_variable("x", float("-inf"), float("inf"))
    m.add_constraint("lower", {"x": 1}, ConstraintSense.GE, 2)
    m.set_objective(ObjectiveSense.MINIMIZE, {"x": 1})
    result = solve(m)
    assert result.status is SolverStatus.OPTIMAL
    assert np.allclose(result.primal, [2])


def test_constant_model_and_time_limit():
    constant = Model("constant")
    constant.set_objective(ObjectiveSense.MINIMIZE, {}, constant=7)
    result = solve(constant)
    assert result.status is SolverStatus.OPTIMAL
    assert result.primal == ()
    assert result.statistics.primal_objective == 7
    assert solve(make_model(), time_limit=0).status is SolverStatus.TIME_LIMIT


def test_repeated_solves_are_deterministic():
    first, second = solve(make_model()), solve(make_model())
    assert first.status is second.status is SolverStatus.OPTIMAL
    assert first.primal == second.primal


def test_phase_two_cannot_reactivate_zero_artificial_variables():
    # Assignment equalities commonly leave zero artificials basic at the
    # Phase-I optimum. They must be pivoted out before Phase II, or a zero
    # Phase-II artificial cost could hide an unsatisfied assignment row.
    m = Model("assignment_basis_cleanup")
    costs = ((1, 9, 9), (9, 1, 9), (9, 9, 1))
    for i in range(3):
        for j in range(3):
            m.add_variable(f"x{i}_{j}", 0, 1)
    for i in range(3):
        m.add_constraint(f"worker_{i}", {f"x{i}_{j}": 1 for j in range(3)},
                         ConstraintSense.LE, 1)
    for j in range(3):
        m.add_constraint(f"job_{j}", {f"x{i}_{j}": 1 for i in range(3)},
                         ConstraintSense.EQ, 1)
    objective = {f"x{i}_{j}": costs[i][j] for i in range(3) for j in range(3)}
    m.set_objective(ObjectiveSense.MINIMIZE, objective)
    result = solve(m)
    assert result.status is SolverStatus.OPTIMAL
    assert np.isclose(result.statistics.primal_objective, 3.0)
    assert np.allclose(np.asarray(result.primal).reshape(3, 3).sum(axis=0), 1.0)
    assert np.all(np.asarray(result.primal).reshape(3, 3).sum(axis=1) <= 1.0 + 1e-8)


def test_unknown_solver_options_are_rejected():
    import pytest
    with pytest.raises(TypeError, match="Unknown solver option"):
        solve(make_model(), magic_fallback=True)
