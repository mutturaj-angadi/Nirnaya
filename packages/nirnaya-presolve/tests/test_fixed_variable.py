from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model


def test_fixed_variable_eliminated_and_substituted():
    m = model(
        variables=[var("x", 3.0, 3.0), var("y", 0, 10)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 8.0)],
        objective={"x": 2.0, "y": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_fixed_variables(st)
    assert changed
    assert not st.var_active[st.name_to_var["x"]]
    r = st.name_to_row["c1"]
    # rhs should have absorbed x's contribution: 8 - 1*3 = 5
    assert abs(st.row_rhs[r] - 5.0) < 1e-9
    # objective offset should have absorbed 2*3 = 6
    assert abs(st.obj_offset - 6.0) < 1e-9


def test_fixed_variable_within_tolerance():
    tol = PresolveTolerances(bound_tol=1e-6)
    m = model(
        variables=[var("x", 1.0, 1.0 + 1e-9)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 8.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, tol)
    changed = P.pass_fixed_variables(st)
    assert changed
    assert not st.var_active[st.name_to_var["x"]]
