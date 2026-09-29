import pytest
from nirnaya_core import Sense, ObjectiveSense, Solution, SolveStatus
from nirnaya_presolve import Presolver
from nirnaya_presolve.exceptions import RecoveryError
from helpers import var, con, model

INF = float("inf")


def test_recover_solution_after_fixed_variable():
    # x fixed; y, z kept genuinely free by a non-degenerate joint inequality
    # (x + y + z <= 8 becomes, after eliminating x=3, y+z <= 5 -- a real
    # two-variable row that presolve cannot reduce further on its own).
    m = model(
        variables=[var("x", 3.0, 3.0), var("y", 0, 10), var("z", 0, 10)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0, "z": 1.0}, Sense.LE, 8.0)],
        objective={"x": 2.0, "y": 1.0, "z": 1.0},
    )
    result = Presolver(enable_scaling=False).run(m)
    assert not result.infeasible and not result.unbounded
    assert "x" not in result.reduced_model.variable_names()
    assert set(result.reduced_model.variable_names()) == {"y", "z"}

    reduced_sol = Solution(values={"y": 4.0, "z": 1.0}, objective_value=5.0, status=SolveStatus.OPTIMAL)
    original = result.recover_solution(reduced_sol)
    assert original.values["x"] == 3.0
    assert original.values["y"] == 4.0
    assert original.values["z"] == 1.0
    assert abs(original.objective_value - (2 * 3.0 + 1 * 4.0 + 1 * 1.0)) < 1e-9


def test_recover_solution_after_doubleton_aggregation():
    # x + y == 10 aggregates x away in favor of y; a second, unrelated
    # variable z with its own non-redundant constraint keeps the reduced
    # model non-trivial so y's value isn't *also* pinned by presolve.
    m = model(
        variables=[var("x", 0, 20), var("y", 0, 20), var("z", 0, 20)],
        constraints=[
            con("c1", {"x": 1.0, "y": 1.0}, Sense.EQ, 10.0),
            con("c2", {"y": 1.0, "z": 1.0}, Sense.LE, 25.0),
        ],
        objective={"x": 1.0, "y": 2.0, "z": 0.0},
    )
    result = Presolver(enable_scaling=False).run(m)
    assert not result.infeasible
    assert "x" not in result.reduced_model.variable_names()
    assert "y" in result.reduced_model.variable_names()

    reduced_values = {n: (10.0 if n == "y" else 5.0) for n in result.reduced_model.variable_names()}
    reduced_sol = Solution(values=reduced_values, objective_value=20.0, status=SolveStatus.OPTIMAL)
    original = result.recover_solution(reduced_sol)
    assert abs(original.values["y"] - 10.0) < 1e-9
    assert abs(original.values["x"] - 0.0) < 1e-9  # x = 10 - y = 0
    assert abs(original.objective_value - (1 * 0.0 + 2 * 10.0 + 0.0 * 5.0)) < 1e-9


def test_recover_solution_after_singleton_column():
    # x is free and appears only in c1 (eliminated via singleton column).
    # c1 has three terms so it is NOT also a doubleton-equality candidate
    # (that pass only fires on exactly-two-term rows and runs first, so a
    # two-term row here would be aggregated away before singleton-column
    # gets a chance -- both are valid reductions, this just pins down
    # which one applies for this test). y, w stay genuinely free via c2.
    m = model(
        variables=[var("x", -INF, INF), var("y", 0, 100), var("w", 0, 50)],
        constraints=[
            con("c1", {"x": 1.0, "y": 2.0, "w": 1.0}, Sense.EQ, 10.0),
            con("c2", {"y": 1.0, "w": 1.0}, Sense.LE, 100.0),
        ],
        objective={"x": 3.0, "y": 0.0, "w": 0.0},
    )
    result = Presolver(enable_scaling=False).run(m)
    assert not result.infeasible
    assert "x" not in result.reduced_model.variable_names()
    assert set(result.reduced_model.variable_names()) == {"y", "w"}

    reduced_values = {"y": 5.0, "w": 2.0}
    reduced_sol = Solution(values=reduced_values, objective_value=-25.0, status=SolveStatus.OPTIMAL)
    original = result.recover_solution(reduced_sol)
    # x = 10 - 2*5 - 2 = -2
    assert abs(original.values["x"] - (-2.0)) < 1e-9
    assert abs(original.values["y"] - 5.0) < 1e-9
    assert abs(original.values["w"] - 2.0) < 1e-9
    assert abs(original.objective_value - (3 * -2.0)) < 1e-9


def test_recover_solution_round_trip_with_scaling():
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 10)],
        constraints=[con("c1", {"x": 1e5, "y": 1e-5}, Sense.LE, 500.0)],
        objective={"x": 1.0, "y": 2.0},
    )
    result = Presolver(enable_scaling=True).run(m)
    assert not result.infeasible
    reduced_values = {n: 0.0 for n in result.reduced_model.variable_names()}
    reduced_sol = Solution(values=reduced_values, objective_value=0.0, status=SolveStatus.OPTIMAL)
    original = result.recover_solution(reduced_sol)
    assert abs(original.values["x"] - 0.0) < 1e-6
    assert abs(original.values["y"] - 0.0) < 1e-6


def test_recover_solution_non_optimal_status_passthrough():
    m = model(
        variables=[var("x", 0, 10)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0},
    )
    result = Presolver().run(m)
    reduced_sol = Solution(values={}, objective_value=0.0, status=SolveStatus.INFEASIBLE)
    original = result.recover_solution(reduced_sol)
    assert original.status == SolveStatus.INFEASIBLE
    assert original.values == {}


def test_recover_solution_missing_value_raises():
    # Zero objective cost on both variables and a genuinely non-redundant
    # joint inequality means presolve cannot resolve either variable on
    # its own -- both remain active in the reduced model.
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 10)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 5.0)],
        objective={},
    )
    result = Presolver().run(m)
    assert not result.infeasible
    assert set(result.reduced_model.variable_names()) == {"x", "y"}
    reduced_sol = Solution(values={"x": 1.0}, objective_value=0.0, status=SolveStatus.OPTIMAL)  # missing y
    with pytest.raises(RecoveryError):
        result.recover_solution(reduced_sol)


def test_recover_solution_raises_if_presolve_found_infeasible():
    m = model(
        variables=[var("x", 0, 3)],
        constraints=[con("c1", {"x": 1.0}, Sense.GE, 10.0)],
        objective={"x": 1.0},
    )
    result = Presolver().run(m)
    assert result.infeasible
    with pytest.raises(RecoveryError):
        result.recover_solution(Solution(values={}, objective_value=0.0, status=SolveStatus.OPTIMAL))
