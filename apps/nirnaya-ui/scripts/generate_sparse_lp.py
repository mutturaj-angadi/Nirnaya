#!/usr/bin/env python3
"""
generate_sparse_lp.py

Deterministically generates a sparse large LP resembling a multi-node
network-flow / production-transport problem (a legitimate LP family, not
random noise dressed up as data). The same seed always yields the same
model, so results are reproducible across runs. GPU numerical primitives are
separate and do not solve this LP.

Model shape:
  - `nodes` supply/demand nodes arranged in `layers` stages (like a
    multi-echelon distribution network: plants -> warehouses -> customers).
  - Arcs only connect adjacent layers (sparse structure), each arc is one
    variable with a unit shipping cost and a capacity upper bound.
  - Flow-balance equality constraints at every node.
  - The network is built with total supply >= total demand so it is
    feasible by construction; balance rhs values are derived, not guessed.

Usage:
    python3 generate_sparse_lp.py --layers 4 --width 25 --seed 7 \
        --out demo-data/sparse_large_lp.json
"""
import argparse
import json
import random


def build_network_lp(layers: int, width: int, seed: int):
    rng = random.Random(seed)

    # layer_sizes: first layer = supply nodes, last = demand nodes,
    # middle layers = transshipment nodes.
    layer_sizes = [width] * layers
    node_ids = []
    for l, size in enumerate(layer_sizes):
        node_ids.append([f"n{l}_{i}" for i in range(size)])

    variables = []
    obj_coeffs = {}
    arcs = []  # (var_name, tail_node, head_node)

    # sparse bipartite connections between consecutive layers: each node
    # connects to a small fixed number of nodes in the next layer.
    fanout = min(6, width)
    for l in range(layers - 1):
        for i, tail in enumerate(node_ids[l]):
            targets = rng.sample(range(width), fanout)
            for j in targets:
                head = node_ids[l + 1][j]
                vname = f"f_{tail}_{head}"
                cost = round(rng.uniform(1.0, 9.0), 2)
                cap = round(rng.uniform(40.0, 90.0), 1)
                variables.append({"name": vname, "lower": 0, "upper": cap, "type": "continuous"})
                obj_coeffs[vname] = cost
                arcs.append((vname, tail, head))

    # supply at layer 0, demand at last layer, derived so that
    # total_supply >= total_demand (guarantees feasibility).
    n_supply = layer_sizes[0]
    n_demand = layer_sizes[-1]
    base_supply = 30.0
    supply = {node_ids[0][i]: round(base_supply + rng.uniform(0, 10), 1) for i in range(n_supply)}
    total_supply = sum(supply.values())

    # demand fractions sum to <= 1, scaled to total_supply * 0.85 to leave slack
    raw = [rng.uniform(0.5, 1.5) for _ in range(n_demand)]
    s = sum(raw)
    demand = {node_ids[-1][i]: round(total_supply * 0.55 * raw[i] / s, 1) for i in range(n_demand)}

    constraints = []

    # supply nodes: outflow <= supply
    for i, node in enumerate(node_ids[0]):
        coeffs = {v: 1 for (v, t, h) in arcs if t == node}
        if coeffs:
            constraints.append({
                "name": f"supply_{node}", "sense": "<=", "rhs": supply[node],
                "coefficients": coeffs
            })

    # transshipment nodes (middle layers): inflow == outflow
    for l in range(1, layers - 1):
        for node in node_ids[l]:
            coeffs = {}
            for (v, t, h) in arcs:
                if h == node:
                    coeffs[v] = coeffs.get(v, 0) + 1
                if t == node:
                    coeffs[v] = coeffs.get(v, 0) - 1
            if coeffs:
                constraints.append({
                    "name": f"balance_{node}", "sense": "=", "rhs": 0,
                    "coefficients": coeffs
                })

    # demand nodes: inflow >= demand
    for node in node_ids[-1]:
        coeffs = {v: 1 for (v, t, h) in arcs if h == node}
        if coeffs:
            constraints.append({
                "name": f"demand_{node}", "sense": ">=", "rhs": demand[node],
                "coefficients": coeffs
            })

    nnz = sum(len(c["coefficients"]) for c in constraints)

    model = {
        "name": "sparse_large_lp",
        "title": f"Sparse Multi-Echelon Network Flow ({layers} layers x {width} nodes)",
        "description": (
            "Reproducibly generated sparse network-flow LP (plants -> "
            "intermediate warehouses -> customers) used for scale/sparsity "
            "validation with Nirnaya?s CPU LP solver. GPU numerical primitives are
            separate and do not solve this LP. Deterministic given the seed."
        ),
        "domain": "logistics",
        "sense": "minimize",
        "variables": variables,
        "objective": {"offset": 0, "coefficients": obj_coeffs},
        "constraints": constraints,
        "metadata": {
            "generator": "generate_sparse_lp.py",
            "seed": seed,
            "layers": layers,
            "width": width,
            "num_variables": len(variables),
            "num_constraints": len(constraints),
            "nonzeros": nnz,
            "density_pct": round(100.0 * nnz / (len(variables) * max(1, len(constraints))), 4)
        }
    }
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--width", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    model = build_network_lp(args.layers, args.width, args.seed)
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Wrote {args.out}: {model['metadata']['num_variables']} vars, "
          f"{model['metadata']['num_constraints']} constraints, "
          f"{model['metadata']['nonzeros']} nonzeros")


if __name__ == "__main__":
    main()
