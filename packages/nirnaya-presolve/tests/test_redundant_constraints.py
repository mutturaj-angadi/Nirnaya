from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model


def test_redundant_le_constraint_removed():
    # x<=5, y<=5  =>  x+y <= 20 is always true (max activity 10 <= 20)
    m = model(
        variables=[var("x", 0, 5), var("y", 0, 5)],
        constraints=[con("c_redundant", {"x": 1.0, "y": 1.0}, Sense.LE, 20.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_redundant_constraints(st)
    assert changed
    assert not st.row_active[st.name_to_row["c_redundant"]]


def test_non_redundant_constraint_kept():
    m = model(
        variables=[var("x", 0, 5), var("y", 0, 5)],
        constraints=[con("c_binding", {"x": 1.0, "y": 1.0}, Sense.LE, 6.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_redundant_constraints(st)
    assert not changed
    assert st.row_active[st.name_to_row["c_binding"]]


def test_redundant_ge_constraint_removed():
    m = model(
        variables=[var("x", 3, 5), var("y", 3, 5)],
        constraints=[con("c_redundant", {"x": 1.0, "y": 1.0}, Sense.GE, 1.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_redundant_constraints(st)
    assert changed
