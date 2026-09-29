from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model

INF = float("inf")


def test_negligible_coefficient_dropped_when_bounded():
    tol = PresolveTolerances(coeff_drop_tol=1e-6)
    # y in [0, 1e-9], coefficient 1e-4  => worst case impact 1e-4*1e-9 = 1e-13 < 1e-6
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 1e-9)],
        constraints=[con("c1", {"x": 1.0, "y": 1e-4}, Sense.LE, 5.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, tol)
    changed = P.pass_coefficient_cleanup(st)
    assert changed
    r = st.name_to_row["c1"]
    cy = st.name_to_var["y"]
    assert cy not in st.rows[r]


def test_coefficient_not_dropped_when_variable_unbounded():
    tol = PresolveTolerances(coeff_drop_tol=1e-6)
    # y unbounded above -> worst case impact is infinite, must never drop
    m = model(
        variables=[var("x", 0, 10), var("y", 0, INF)],
        constraints=[con("c1", {"x": 1.0, "y": 1e-9}, Sense.LE, 5.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, tol)
    changed = P.pass_coefficient_cleanup(st)
    assert not changed
    r = st.name_to_row["c1"]
    cy = st.name_to_var["y"]
    assert cy in st.rows[r]


def test_significant_coefficient_kept():
    tol = PresolveTolerances(coeff_drop_tol=1e-6)
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 10)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, tol)
    changed = P.pass_coefficient_cleanup(st)
    assert not changed
