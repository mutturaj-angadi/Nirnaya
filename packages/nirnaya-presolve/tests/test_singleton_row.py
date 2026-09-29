from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model


def test_singleton_row_tightens_upper_bound():
    m = model(
        variables=[var("x", 0, 100)],
        constraints=[con("c1", {"x": 2.0}, Sense.LE, 10.0)],  # 2x <= 10 -> x <= 5
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_singleton_rows(st)
    assert changed
    c = st.name_to_var["x"]
    assert abs(st.var_upper[c] - 5.0) < 1e-9
    assert not st.row_active[st.name_to_row["c1"]]


def test_singleton_row_negative_coefficient_flips_sense():
    m = model(
        variables=[var("x", -100, 100)],
        constraints=[con("c1", {"x": -2.0}, Sense.LE, 10.0)],  # -2x <= 10 -> x >= -5
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_singleton_rows(st)
    c = st.name_to_var["x"]
    assert abs(st.var_lower[c] - (-5.0)) < 1e-9


def test_singleton_row_equality_fixes_both_bounds():
    m = model(
        variables=[var("x", -100, 100)],
        constraints=[con("c1", {"x": 4.0}, Sense.EQ, 8.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_singleton_rows(st)
    c = st.name_to_var["x"]
    assert abs(st.var_lower[c] - 2.0) < 1e-9
    assert abs(st.var_upper[c] - 2.0) < 1e-9


def test_singleton_row_infeasible_when_conflicting():
    m = model(
        variables=[var("x", 0, 3)],
        constraints=[con("c1", {"x": 1.0}, Sense.GE, 10.0)],  # x >= 10, but ub=3
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_singleton_rows(st)
    assert st.infeasible
