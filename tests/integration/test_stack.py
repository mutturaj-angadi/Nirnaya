from nirnaya_core import ConstraintSense, Model, ObjectiveSense, SolverStatus
from nirnaya_presolve import Presolver
from nirnaya_solver import solve


def test_core_presolve_solver_postsolve_recovery():
    model = Model("fixed-variable")
    model.add_variable("fixed", 2, 2)
    model.add_variable("x", 0, 10)
    model.add_constraint("cap", {"fixed": 1, "x": 1}, ConstraintSense.LE, 6)
    model.set_objective(ObjectiveSense.MAXIMIZE, {"fixed": 3, "x": 2}, constant=4)
    result = Presolver().run(model)
    assert result.reduced_model is not None
    reduced = solve(result.reduced_model)
    assert reduced.status is SolverStatus.OPTIMAL
    recovered = result.recover_solution(reduced)
    assert recovered.status is SolverStatus.OPTIMAL
    values = dict(zip(recovered.variable_names, recovered.primal))
    assert values == {"fixed": 2.0, "x": 4.0}
    assert recovered.statistics.primal_objective == 18.0
