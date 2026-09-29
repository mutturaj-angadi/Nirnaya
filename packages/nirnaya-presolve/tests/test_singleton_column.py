from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model

INF = float("inf")


def test_free_singleton_column_in_equality_row_eliminated():
    # x is free and appears only in c1: x + 2y == 10
    m = model(
        variables=[var("x", -INF, INF), var("y", 0, 100)],
        constraints=[con("c1", {"x": 1.0, "y": 2.0}, Sense.EQ, 10.0)],
        objective={"x": 3.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_singleton_column(st)
    assert changed
    assert not st.var_active[st.name_to_var["x"]]
    assert not st.row_active[st.name_to_row["c1"]]
    rec = [e for e in st.eliminated if e.name == "x"][0]
    assert rec.mode == "multi_affine"
    # x = 10/1 - (2/1)*y = 10 - 2y ; objective absorbs 3*x = 3*(10-2y) = 30 - 6y
    assert abs(rec.beta - 10.0) < 1e-9
    assert abs(rec.terms["y"] - (-2.0)) < 1e-9
    cy = st.name_to_var["y"]
    assert abs(st.var_obj[cy] - (1.0 + 3.0 * (-2.0))) < 1e-9  # 1 - 6 = -5
    assert abs(st.obj_offset - 30.0) < 1e-9


def test_bounded_variable_not_eliminated_as_singleton_column():
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 100)],  # x NOT free
        constraints=[con("c1", {"x": 1.0, "y": 2.0}, Sense.EQ, 10.0)],
        objective={"x": 3.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_singleton_column(st)
    assert not changed


def test_inequality_row_not_eliminated_as_singleton_column():
    m = model(
        variables=[var("x", -INF, INF), var("y", 0, 100)],
        constraints=[con("c1", {"x": 1.0, "y": 2.0}, Sense.LE, 10.0)],
        objective={"x": 3.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_singleton_column(st)
    assert not changed
