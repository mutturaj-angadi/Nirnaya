from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model


def test_doubleton_equality_eliminates_one_variable():
    # x + y == 10, x in [0,20], y in [0,20]
    m = model(
        variables=[var("x", 0, 20), var("y", 0, 20)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.EQ, 10.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_doubleton_aggregation(st)
    assert changed
    assert st.n_active_rows() == 0
    assert st.n_active_vars() == 1
    rec = st.eliminated[0]
    assert rec.mode == "aggregation"
    assert rec.name == "x"
    assert rec.other_name == "y"
    assert abs(rec.alpha - (-1.0)) < 1e-9
    assert abs(rec.beta - 10.0) < 1e-9


def test_doubleton_aggregation_tightens_survivor_bound():
    # x + y == 10, x in [0,3] (tight), y in [0,20]
    # x = 10 - y in [0,3]  =>  10-y in [0,3]  =>  y in [7,10]
    m = model(
        variables=[var("x", 0, 3), var("y", 0, 20)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.EQ, 10.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_doubleton_aggregation(st)
    cy = st.name_to_var["y"]
    assert abs(st.var_lower[cy] - 7.0) < 1e-9
    assert abs(st.var_upper[cy] - 10.0) < 1e-9


def test_doubleton_aggregation_detects_infeasible_implied_bound():
    # x + y == 100, x in [0,3], y in [0,5]  => x = 100-y in [95,100], impossible vs x<=3
    m = model(
        variables=[var("x", 0, 3), var("y", 0, 5)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.EQ, 100.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_doubleton_aggregation(st)
    assert st.infeasible


def test_inequality_doubleton_not_touched():
    m = model(
        variables=[var("x", 0, 20), var("y", 0, 20)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 10.0)],
        objective={"x": 1.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_doubleton_aggregation(st)
    assert not changed
