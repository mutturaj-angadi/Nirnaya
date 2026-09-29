"""
examples/03_sparse_model.py

Builds a larger, genuinely sparse LP (a small transportation problem: a
set of supply nodes shipping to a set of demand nodes, where each
constraint only touches the handful of variables for that node) and shows
that the compiled ProblemData stays sparse -- no dense n x m matrix is
ever materialized unless you explicitly ask for one.

Run:
    python examples/03_sparse_model.py
"""

from __future__ import annotations

from nirnaya_core import ConstraintSense, Model, ObjectiveSense


def build_transportation_model(n_supply: int, n_demand: int) -> Model:
    model = Model(name=f"transport_{n_supply}x{n_demand}")

    # One variable per (supply, demand) route.
    costs = {}
    for i in range(n_supply):
        for j in range(n_demand):
            name = f"ship_{i}_{j}"
            model.add_variable(name, lower=0.0)
            # Arbitrary but deterministic per-route cost.
            costs[name] = 1.0 + (i * n_demand + j) % 7

    model.set_objective(ObjectiveSense.MINIMIZE, costs)

    # Supply constraints: each supply node ships at most `capacity` units,
    # touching only the n_demand variables for that node.
    capacity = 20.0
    for i in range(n_supply):
        row = {f"ship_{i}_{j}": 1.0 for j in range(n_demand)}
        model.add_constraint(f"supply_{i}", row, ConstraintSense.LE, capacity)

    # Demand constraints: each demand node must receive at least
    # `requirement` units, touching only the n_supply variables for that
    # node.
    requirement = 5.0
    for j in range(n_demand):
        row = {f"ship_{i}_{j}": 1.0 for i in range(n_supply)}
        model.add_constraint(f"demand_{j}", row, ConstraintSense.GE, requirement)

    return model


def main() -> None:
    n_supply, n_demand = 10, 12
    model = build_transportation_model(n_supply, n_demand)
    data = model.build()

    n_variables = data.n_variables
    total_ub_entries = data.n_ub * n_variables

    print(f"variables: {n_variables}")
    print(f"a_ub shape: {data.a_ub.shape}, nnz: {data.a_ub.nnz}")
    print(f"a_ub density: {data.a_ub.density:.4%}")
    print(
        f"(a dense a_ub would store {total_ub_entries} floats; "
        f"the sparse form stores {data.a_ub.nnz})"
    )


if __name__ == "__main__":
    main()
