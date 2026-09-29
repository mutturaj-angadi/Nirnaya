"""Deterministic NOC-10K instance generation and shared metadata."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from functools import lru_cache
from typing import Any

GENERATOR_VERSION = "noc10k-generator-1.0.0"
SCHEMA_VERSION = "noc10k-instance-1.0"
DEFAULT_SEED = 20260928
FAMILY_COUNTS = {
    "small_dense": 1000,
    "small_sparse": 1000,
    "medium_sparse": 2000,
    "large_sparse": 1500,
    "degenerate": 750,
    "ill_conditioned": 500,
    "equality_heavy": 500,
    "bound_heavy": 500,
    "presolve_heavy": 500,
    "infeasible": 500,
    "unbounded": 250,
    # The brief's listed allocations sum to 9,750 despite stating 10,000.
    # Add the arithmetic difference to the generic industrial-style family.
    "industrial_style": 1000,
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


@lru_cache(maxsize=None)
def _split_map(family: str, count: int, seed: int) -> dict[int, str]:
    # Rank by a stable digest so each family receives its exact 80/10/10
    # allocation while membership is independent of solver outcomes.
    order = sorted(range(count), key=lambda i: hashlib.sha256(
        f"{seed}:{family}:{i}:noc10k-split-v1".encode()).digest())
    dev_end = math.floor(count * .8)
    val_end = math.floor(count * .9)
    return {index: "development" if rank < dev_end else
            "validation" if rank < val_end else "holdout"
            for rank, index in enumerate(order)}


def _bounded_random_lp(family: str, seed: int, rows: int, cols: int, density: float,
                       *, sense: str = "maximize", duplicate: bool = False,
                       scaled: bool = False, equality_share: float = 0.0,
                       fixed_share: float = 0.0, singleton_share: float = 0.0) -> dict:
    rng = random.Random(seed)
    names = [f"x{j:04d}" for j in range(cols)]
    x0 = [rng.uniform(1.0, 7.0) for _ in range(cols)]
    variables = []
    for j, name in enumerate(names):
        lo, hi = 0.0, 10.0
        if fixed_share and j < int(cols * fixed_share):
            lo = hi = round(x0[j], 5)
        elif family == "bound_heavy":
            hi = round(rng.uniform(1.0, 4.0), 4)
            x0[j] = min(x0[j], hi * .6)
        variables.append({"name": name, "lower": lo, "upper": hi, "type": "continuous"})
    obj = {name: round(rng.uniform(.2, 12.0), 6) for name in names}
    constraints = []
    k = max(1, min(cols, round(cols * density)))
    for i in range(rows):
        if singleton_share and i < int(rows * singleton_share):
            selected = [i % cols]
        elif k == cols:
            selected = list(range(cols))
        else:
            selected = sorted(rng.sample(range(cols), k))
        scale = (10.0 ** rng.choice((-5, 0, 5))) if scaled else 1.0
        coeff = {names[j]: round(rng.uniform(.2, 2.0) * scale, 12) for j in selected}
        rhs = sum(coeff[names[j]] * x0[j] for j in selected) + abs(scale) * rng.uniform(.2, 2.0)
        is_eq = equality_share and i < int(rows * equality_share)
        constraints.append({"name": f"row_{i:04d}", "sense": "=" if is_eq else "<=",
                            "rhs": rhs if not is_eq else sum(coeff[names[j]] * x0[j] for j in selected),
                            "coefficients": coeff})
    if duplicate and rows:
        first = constraints[0]
        constraints.append({**first, "name": f"row_duplicate_{rows:04d}",
                            "coefficients": dict(first["coefficients"])})
    if family == "infeasible":
        constraints.extend([
            {"name": "contradiction_upper", "sense": "<=", "rhs": 1.0,
             "coefficients": {names[0]: 1.0}},
            {"name": "contradiction_lower", "sense": ">=", "rhs": 2.0,
             "coefficients": {names[0]: 1.0}},
        ])
    if family == "unbounded":
        variables[0]["upper"] = None
        obj = {names[0]: 1.0}
        constraints = []
        sense = "maximize"
    return {"name": f"noc10k_{family}_{seed}", "title": f"NOC-10K {family}",
            "description": "Synthetic industrial-style optimization instance generated deterministically.",
            "domain": "synthetic-industrial-style-optimization", "sense": sense,
            "variables": variables, "objective": {"offset": 0.0, "coefficients": obj},
            "constraints": constraints}


def _industrial(seed: int) -> dict:
    rng = random.Random(seed)
    products = rng.randint(4, 9)
    resources = rng.randint(3, 7)
    names = [f"product_{j:02d}" for j in range(products)]
    variables = [{"name": n, "lower": 0.0, "upper": round(rng.uniform(20, 80), 3),
                  "type": "continuous"} for n in names]
    objective = {n: round(rng.uniform(2, 20), 4) for n in names}
    nominal = [rng.uniform(4, 12) for _ in names]
    constraints = []
    for r in range(resources):
        coeff = {n: round(rng.uniform(.1, 4.0), 5) for n in names}
        rhs = sum(coeff[n] * nominal[j] for j, n in enumerate(names)) * rng.uniform(1.1, 1.6)
        constraints.append({"name": f"resource_{r:02d}", "sense": "<=", "rhs": rhs,
                            "coefficients": coeff})
    mix = {n: round(rng.uniform(.1, 1.0), 5) for n in names}
    constraints.append({"name": "minimum_demand", "sense": ">=",
                        "rhs": sum(mix[n] * nominal[j] for j, n in enumerate(names)) * .45,
                        "coefficients": mix})
    return {"name": f"noc10k_industrial_style_{seed}", "title": "Synthetic production and resource allocation",
            "description": "Synthetic industrial-style optimization instance; not operational plant data.",
            "domain": "synthetic-industrial-style-optimization", "sense": "maximize",
            "variables": variables, "objective": {"offset": 0.0, "coefficients": objective},
            "constraints": constraints}


def family_shape(family: str, seed: int) -> tuple[int, int, float]:
    rng = random.Random(seed ^ 0x5A17)
    if family == "small_dense": return rng.randint(5, 12), rng.randint(6, 14), 1.0
    if family == "small_sparse": return rng.randint(6, 16), rng.randint(10, 28), .18
    if family == "medium_sparse": return rng.randint(18, 45), rng.randint(40, 90), .08
    if family == "large_sparse": return rng.randint(35, 80), rng.randint(90, 180), .035
    if family == "degenerate": return rng.randint(8, 24), rng.randint(12, 40), .25
    if family == "ill_conditioned": return rng.randint(8, 22), rng.randint(12, 35), .25
    if family == "equality_heavy": return rng.randint(8, 24), rng.randint(12, 45), .2
    if family == "bound_heavy": return rng.randint(12, 35), rng.randint(20, 60), .12
    if family == "presolve_heavy": return rng.randint(10, 30), rng.randint(20, 55), .14
    if family == "infeasible": return rng.randint(4, 15), rng.randint(8, 25), .2
    if family == "unbounded": return 0, 1, 0.0
    return 0, 0, 0.0


def build_instance(family: str, seed: int) -> dict:
    m, n, density = family_shape(family, seed)
    if family == "industrial_style":
        return _industrial(seed)
    return _bounded_random_lp(family, seed, m, n, density,
        duplicate=family == "degenerate", scaled=family == "ill_conditioned",
        equality_share=.7 if family == "equality_heavy" else 0.0,
        fixed_share=.3 if family == "presolve_heavy" else 0.0,
        singleton_share=.5 if family == "presolve_heavy" else 0.0,
        sense="minimize" if family == "infeasible" else "maximize")


def describe_instance(model: dict, family: str, seed: int, family_index: int,
                      family_count: int, corpus_seed: int) -> dict:
    variables, constraints = model["variables"], model["constraints"]
    nnz = sum(len(row["coefficients"]) for row in constraints)
    total = len(variables) * len(constraints)
    all_coeff = [abs(float(v)) for row in constraints for v in row["coefficients"].values()]
    finite_ub = sum(v.get("upper") is not None for v in variables)
    fixed = sum(v.get("upper") is not None and v.get("lower", 0) == v.get("upper")
                for v in variables)
    digest = hashlib.sha256(canonical_json(model)).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "instance_id": f"NOC10K-{family}-{family_index:05d}",
        "seed": seed,
        "corpus_seed": corpus_seed,
        "family": family,
        "family_index": family_index,
        "split": _split_map(family, family_count, corpus_seed)[family_index],
        "variables": len(variables), "constraints": len(constraints), "nonzeros": nnz,
        "density": nnz / total if total else 0.0,
        "objective_direction": model["sense"],
        "constraint_senses": dict(sorted(Counter(r["sense"] for r in constraints).items())),
        "bound_characteristics": {"finite_upper_bounds": finite_ub,
                                  "unbounded_upper_bounds": len(variables) - finite_ub,
                                  "fixed_variables": fixed},
        "coefficient_scale": {"min_abs": min(all_coeff, default=None),
                              "max_abs": max(all_coeff, default=None)},
        "expected_structural_class": "infeasible" if family == "infeasible" else
            "unbounded" if family == "unbounded" else "feasible_bounded",
        "generation_parameters": {"rows": len(constraints), "columns": len(variables),
                                  "observed_density": nnz / total if total else 0.0},
        "model_sha256": digest,
    }


def materialize_instance(family: str, seed: int, family_index: int,
                         family_count: int, corpus_seed: int) -> tuple[dict, dict]:
    model = build_instance(family, seed)
    record = describe_instance(model, family, seed, family_index, family_count, corpus_seed)
    model["metadata"] = record
    return model, record


def allocation(count: int, family: str | None = None) -> dict[str, int]:
    if family:
        if family not in FAMILY_COUNTS:
            raise ValueError(f"Unknown family: {family}")
        return {family: count}
    weights = sum(FAMILY_COUNTS.values())
    raw = {key: count * value / weights for key, value in FAMILY_COUNTS.items()}
    result = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda k: (-(raw[k] - result[k]), k))[:count - sum(result.values())]:
        result[key] += 1
    return result


def iter_specs(seed: int = DEFAULT_SEED, family: str | None = None,
               count: int | None = None):
    default_count = FAMILY_COUNTS[family] if family else sum(FAMILY_COUNTS.values())
    counts = allocation(count if count is not None else default_count, family)
    for family_name, size in counts.items():
        for index in range(size):
            yield family_name, index, seed + index + list(FAMILY_COUNTS).index(family_name) * 1_000_000, size
