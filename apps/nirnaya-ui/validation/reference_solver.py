#!/usr/bin/env python3
"""
reference_solver.py

An *independent* LP solver used only to check Nirnaya's answers. It is
deliberately implemented against a different numerical library (SciPy's
HiGHS-backed `linprog`) than Nirnaya's repository-authored CPU revised-simplex solver, so
agreement between the two is real evidence of correctness rather than a
tautology.

Converts a Nirnaya-schema LP model (see docs/API_CONTRACT.md) into standard
form and solves it with `scipy.optimize.linprog(method="highs")`.

Usage as a library:
    from reference_solver import solve_reference
    result = solve_reference(model_dict)

Usage as a script:
    python3 reference_solver.py path/to/model.json
"""
import json
import sys
from typing import Any, Dict

import numpy as np
from scipy.optimize import linprog


def solve_reference(model: Dict[str, Any], time_limit: float | None = None) -> Dict[str, Any]:
    var_names = [v["name"] for v in model["variables"]]
    idx = {name: i for i, name in enumerate(var_names)}
    n = len(var_names)

    sense = model.get("sense", "minimize").lower()
    obj = model["objective"]
    c = np.zeros(n)
    for name, coef in obj.get("coefficients", {}).items():
        c[idx[name]] = coef
    offset = obj.get("offset", 0.0)
    # linprog always minimizes; flip sign for maximize.
    c_solve = -c if sense == "maximize" else c.copy()

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    for con in model["constraints"]:
        row = np.zeros(n)
        for name, coef in con["coefficients"].items():
            row[idx[name]] = coef
        s = con["sense"]
        rhs = con["rhs"]
        if s == "<=":
            A_ub.append(row); b_ub.append(rhs)
        elif s == ">=":
            A_ub.append(-row); b_ub.append(-rhs)
        elif s == "=":
            A_eq.append(row); b_eq.append(rhs)
        else:
            raise ValueError(f"Unknown constraint sense: {s}")

    bounds = []
    for v in model["variables"]:
        lo = v.get("lower", 0)
        hi = v.get("upper", None)
        lo = None if lo is None else lo
        bounds.append((lo, hi))

    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None

    options = {"time_limit": float(time_limit)} if time_limit is not None else None
    res = linprog(c_solve, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs", options=options)

    status_map = {
        0: "optimal",
        1: "iteration_limit",
        2: "infeasible",
        3: "unbounded",
        4: "numerical_difficulty",
    }
    status = status_map.get(res.status, f"unknown({res.status})")

    out = {
        "engine": "scipy.optimize.linprog(method=highs)",
        "status": status,
        "objective": None,
        "variables": None,
        "iterations": int(getattr(res, "nit", 0)) if res.nit is not None else None,
        "raw_message": res.message,
    }

    if status == "optimal":
        raw_obj = res.fun
        true_obj = (-raw_obj if sense == "maximize" else raw_obj) + offset
        out["objective"] = float(true_obj)
        out["variables"] = {name: float(res.x[idx[name]]) for name in var_names}

        # compute constraint residuals for diagnostics
        residuals = {}
        for con in model["constraints"]:
            lhs = sum(coef * out["variables"][name] for name, coef in con["coefficients"].items())
            rhs = con["rhs"]
            if con["sense"] == "<=":
                residuals[con["name"]] = max(0.0, lhs - rhs)
            elif con["sense"] == ">=":
                residuals[con["name"]] = max(0.0, rhs - lhs)
            else:
                residuals[con["name"]] = abs(lhs - rhs)
        out["max_residual"] = max(residuals.values()) if residuals else 0.0

    return out


def main():
    if len(sys.argv) != 2:
        print("usage: reference_solver.py <model.json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        model = json.load(f)
    result = solve_reference(model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
