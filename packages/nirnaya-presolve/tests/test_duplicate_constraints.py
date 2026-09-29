from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model


def test_duplicate_le_constraints_merge_to_tighter():
    m = model(
        variables=[var("x", 0, 100), var("y", 0, 100)],
        constraints=[
            con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 10.0),
            con("c2", {"x": 2.0, "y": 2.0}, Sense.LE, 30.0),  # equiv to x+y<=15, looser
        ],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_duplicate_constraints(st)
    assert changed
    active = [r for r in st.active_row_indices()]
    assert len(active) == 1
    r = active[0]
    assert abs(st.row_rhs[r] - 10.0) < 1e-9  # tighter of the two kept


def test_proportional_equalities_consistent_merge():
    m = model(
        variables=[var("x", 0, 100), var("y", 0, 100)],
        constraints=[
            con("c1", {"x": 1.0, "y": 1.0}, Sense.EQ, 10.0),
            con("c2", {"x": 2.0, "y": 2.0}, Sense.EQ, 20.0),
        ],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_duplicate_constraints(st)
    assert changed
    assert st.n_active_rows() == 1
    assert not st.infeasible


def test_proportional_equalities_contradictory_infeasible():
    m = model(
        variables=[var("x", 0, 100), var("y", 0, 100)],
        constraints=[
            con("c1", {"x": 1.0, "y": 1.0}, Sense.EQ, 10.0),
            con("c2", {"x": 2.0, "y": 2.0}, Sense.EQ, 21.0),  # equiv x+y=10.5, contradicts
        ],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_duplicate_constraints(st)
    assert st.infeasible


def test_unrelated_constraints_not_merged():
    m = model(
        variables=[var("x", 0, 100), var("y", 0, 100)],
        constraints=[
            con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 10.0),
            con("c2", {"x": 1.0, "y": 2.0}, Sense.LE, 10.0),  # not proportional
        ],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_duplicate_constraints(st)
    assert not changed
    assert st.n_active_rows() == 2
