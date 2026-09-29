from nirnaya_core import Sense, ObjectiveSense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model

INF = float("inf")


def test_empty_column_zero_obj_fixed_at_zero_within_bounds():
    m = model(
        variables=[var("x", 0, 10), var("y", -5, 5)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0},  # y absent from objective and constraints
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_empty_columns(st)
    assert changed
    assert not st.var_active[st.name_to_var["y"]]
    rec = [e for e in st.eliminated if e.name == "y"][0]
    assert rec.mode == "fixed"
    assert rec.fixed_value == 0.0


def test_empty_column_positive_obj_fixed_at_lower_bound():
    m = model(
        variables=[var("x", 0, 10), var("y", 2, 8)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0, "y": 3.0},  # minimize: y wants to be small
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_empty_columns(st)
    rec = [e for e in st.eliminated if e.name == "y"][0]
    assert rec.fixed_value == 2.0


def test_empty_column_negative_obj_fixed_at_upper_bound():
    m = model(
        variables=[var("x", 0, 10), var("y", 2, 8)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0, "y": -3.0},  # minimize: y wants to be large
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_empty_columns(st)
    rec = [e for e in st.eliminated if e.name == "y"][0]
    assert rec.fixed_value == 8.0


def test_empty_column_unbounded_direction_flags_unbounded():
    m = model(
        variables=[var("x", 0, 10), var("y", 0, INF)],
        constraints=[con("c1", {"x": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0, "y": -3.0},  # minimize wants y -> +inf, no upper bound
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_empty_columns(st)
    assert st.unbounded
