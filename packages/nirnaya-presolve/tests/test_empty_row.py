from nirnaya_core import Sense, ObjectiveSense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model


def test_empty_row_trivially_satisfied_is_removed():
    m = model(
        variables=[var("x", 0, 10)],
        constraints=[
            con("c1", {"x": 1.0}, Sense.LE, 5.0),
            con("c_empty", {}, Sense.LE, 3.0),  # 0 <= 3, trivially true
        ],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_empty_rows(st)
    assert changed
    assert not st.row_active[st.name_to_row["c_empty"]]
    assert st.row_active[st.name_to_row["c1"]]
    assert not st.infeasible


def test_empty_row_violated_is_infeasible():
    m = model(
        variables=[var("x", 0, 10)],
        constraints=[con("c_bad", {}, Sense.LE, -3.0)],  # 0 <= -3, false
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_empty_rows(st)
    assert st.infeasible
    assert "c_bad" in st.infeasible_reason


def test_empty_row_equality_violated():
    m = model(
        variables=[var("x", 0, 10)],
        constraints=[con("c_eq", {}, Sense.EQ, 1.0)],  # 0 == 1, false
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_empty_rows(st)
    assert st.infeasible
