"""Explicit CPU-backend tests (always run, no hardware dependency)."""

from nirnaya_api import Model, SolveStatus


def test_cpu_solve_maximize(fake_backend):
    m = Model("cpu-max")
    m.add_variable("x", lb=0, ub=4)
    m.set_objective({"x": 1}, sense="max")
    result = m.solve(backend="cpu")
    assert result.status == SolveStatus.OPTIMAL
    assert result.variable_values["x"] == 4
    assert result.objective_value == 4
    assert result.backend == "cpu"


def test_cpu_solve_minimize(fake_backend):
    m = Model("cpu-min")
    m.add_variable("x", lb=1, ub=4)
    m.set_objective({"x": 1}, sense="min")
    result = m.solve(backend="cpu")
    assert result.variable_values["x"] == 1
    assert result.objective_value == 1


def test_cpu_solve_with_verbose_diagnostics(fake_backend):
    m = Model("cpu-verbose")
    m.add_variable("x", lb=0, ub=4)
    m.set_objective({"x": 1}, sense="max")
    result = m.solve(backend="cpu", verbose=True)
    assert result.diagnostics.max_infeasibility is not None


def test_cpu_solve_respects_no_presolve(fake_backend):
    m = Model("cpu-no-presolve")
    m.add_variable("x", lb=0, ub=4)
    m.set_objective({"x": 1}, sense="max")
    result = m.solve(backend="cpu", presolve=False)
    assert result.presolve_time_s == 0.0
