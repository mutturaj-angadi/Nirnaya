"""
The single translation boundary between `nirnaya_core` public objects and
presolve's internal sparse representation (`presolve.state._PresolveState`).

If the real nirnaya-core's API differs from docs/ASSUMED_CORE_API.md, this
is the only file that should need to change.
"""

from __future__ import annotations

from nirnaya_core import Model, VariableType, ConstraintSense, ObjectiveSense

from .presolve.state import _PresolveState
from .exceptions import ModelBuildError

_SENSE_TO_INTERNAL = {ConstraintSense.LE: "L", ConstraintSense.GE: "G", ConstraintSense.EQ: "E"}
_INTERNAL_TO_SENSE = {v: k for k, v in _SENSE_TO_INTERNAL.items()}


def model_from_core(model: Model, tolerances) -> _PresolveState:
    st = _PresolveState(tolerances)
    st.name = getattr(model, "name", "") or ""
    data = model.build()
    st.original_sense = data.sense
    st.obj_offset = float(data.objective_constant)
    minimize = data.sense == ObjectiveSense.MINIMIZE
    sign = 1.0 if minimize else -1.0
    if not minimize:
        st.obj_offset = -st.obj_offset

    for i, name in enumerate(data.variable_names):
        obj_coeff = sign * float(data.c[i])
        st.add_variable(name, float(data.lower[i]), float(data.upper[i]), obj_coeff,
                        data.var_types[i] != VariableType.CONTINUOUS)

    st._original_objective_for_report = {name: float(data.c[i]) for i, name in enumerate(data.variable_names)}
    st._original_objective_offset_for_report = float(data.objective_constant)

    for matrix, rhs, row_names, sense in (
        (data.a_eq, data.b_eq, data.constraint_names_eq, "E"),
        (data.a_ub, data.b_ub, data.constraint_names_ub, "L"),
    ):
        csr = matrix.to_csr()
        for row, name in enumerate(row_names):
            start, end = csr.indptr[row], csr.indptr[row + 1]
            coeffs = {int(c): float(v) for c, v in zip(csr.indices[start:end], csr.data[start:end])}
            st.add_row(name, coeffs, sense, float(rhs[row]))

    st.stats.original_num_variables = len(st.var_names)
    st.stats.original_num_constraints = len(st.row_names)
    return st


def model_to_core(st: _PresolveState, config=None) -> Model:
    """Build the reduced-space core LPModel from the active rows/columns
    of `st`. Only nirnaya_core's public constructors are used."""
    active_cols = [c for c in st.active_col_indices()]
    active_rows = [r for r in st.active_row_indices()]

    reduced = Model(name=st.name or "presolved_model", config=config)
    for c in active_cols:
        reduced.add_variable(st.var_names[c], lower=st.var_lower[c], upper=st.var_upper[c],
                             vtype=VariableType.INTEGER if st.var_is_integer[c] else VariableType.CONTINUOUS)
    for r in active_rows:
        coeffs = {st.var_names[c]: a for c, a in st.rows[r].items() if st.var_active[c]}
        sense = {"L": ConstraintSense.LE, "G": ConstraintSense.GE, "E": ConstraintSense.EQ}[st.row_sense[r]]
        reduced.add_constraint(st.row_names[r], coeffs, sense, st.row_rhs[r])

    minimize = st.original_sense == ObjectiveSense.MINIMIZE
    sign = 1.0 if minimize else -1.0
    objective = {
        st.var_names[c]: sign * st.var_obj[c]
        for c in active_cols
        if st.var_obj[c] != 0.0
    }
    offset = sign * st.obj_offset

    reduced.set_objective(st.original_sense, objective, constant=offset)
    return reduced
