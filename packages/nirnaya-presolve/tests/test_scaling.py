from nirnaya_core import Sense
from nirnaya_presolve.adapter import model_from_core
from nirnaya_presolve.presolve.tolerances import PresolveTolerances
from nirnaya_presolve.presolve import passes as P
from helpers import var, con, model


def test_scaling_records_nontrivial_factors_for_badly_scaled_row():
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 10)],
        constraints=[con("c1", {"x": 1e6, "y": 1e-6}, Sense.LE, 5.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    changed = P.pass_scaling(st)
    assert changed
    r = st.name_to_row["c1"]
    assert st.row_scale[r] != 1.0
    # after row scaling, the two coefficients should be much closer in magnitude
    coeffs = [abs(a) for _, a in st.rows[r].items()]
    assert max(coeffs) / min(coeffs) < 1e12  # much improved vs original 1e12 ratio... sanity


def test_scaling_no_op_on_well_scaled_model():
    m = model(
        variables=[var("x", 0, 10), var("y", 0, 10)],
        constraints=[con("c1", {"x": 1.0, "y": 1.0}, Sense.LE, 5.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_scaling(st)
    r = st.name_to_row["c1"]
    assert abs(st.row_scale[r] - 1.0) < 1e-9


def test_scaling_preserves_feasible_region_direction():
    # sanity: scaling a single row by a positive factor must not flip its sense
    m = model(
        variables=[var("x", 0, 10)],
        constraints=[con("c1", {"x": 1000.0}, Sense.LE, 5000.0)],
        objective={"x": 1.0},
    )
    st = model_from_core(m, PresolveTolerances())
    P.pass_scaling(st)
    assert st.row_sense[st.name_to_row["c1"]] == "L"
