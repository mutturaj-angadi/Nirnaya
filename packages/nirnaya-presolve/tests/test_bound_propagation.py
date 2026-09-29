from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model

INF = float("inf")


def test_bound_propagation_tightens_from_others():
    # x + y <= 10, 0<=x<=100, 3<=y<=3 (y pinned at 3)  =>  x <= 10 - 3 = 7
    m = model(
        variables=[var("x", 0, 100), var("y", 3, 3)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 10.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_bound_propagation(st)
    assert changed
    cx = st.name_to_var["x"]
    assert st.var_upper[cx] <= 7.0 + 1e-9


def test_bound_propagation_ge_row_tightens_lower_bound():
    # x + y >= 5, 0<=x<=100, 0<=y<=2  =>  x >= 3
    m = model(
        variables=[var("x", 0, 100), var("y", 0, 2)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.GE, 5.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_bound_propagation(st)
    cx = st.name_to_var["x"]
    assert st.var_lower[cx] >= 3.0 - 1e-9


def test_bound_propagation_detects_infeasibility():
    # x + y <= 5 but x,y both >= 10 => min activity 20 > 5
    m = model(
        variables=[var("x", 10, 100), var("y", 10, 100)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_bound_propagation(st)
    assert st.infeasible


def test_bound_propagation_no_change_when_already_tight():
    m = model(
        variables=[var("x", 0, 5), var("y", 0, 5)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 100.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_bound_propagation(st)
    assert not changed
