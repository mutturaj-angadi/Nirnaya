"""Generate the deterministic MRPL-inspired synthetic refinery scenario."""
from __future__ import annotations

import json
from pathlib import Path


def build_model() -> dict:
    crude = ("light", "medium", "heavy")
    streams = ("reformate", "fcc", "alkylate", "lsr")
    grades = ("regular", "premium")
    market_grade = (("domestic", "regular"), ("domestic", "premium"),
                    ("export", "regular"), ("export", "premium"))
    yields = {"light": {"reformate": .25, "fcc": .30, "alkylate": .08, "lsr": .25},
              "medium": {"reformate": .18, "fcc": .38, "alkylate": .12, "lsr": .20},
              "heavy": {"reformate": .10, "fcc": .42, "alkylate": .05, "lsr": .18}}
    available = {"light": 38.0, "medium": 42.0, "heavy": 30.0}
    crude_cost = {"light": 48.0, "medium": 39.0, "heavy": 32.0}
    market_revenue = {("domestic", "regular"): 98.0, ("domestic", "premium"): 114.0,
                      ("export", "regular"): 93.0, ("export", "premium"): 108.0}
    shipment_caps = {("domestic", "regular"): 42.0, ("domestic", "premium"): 28.0,
                     ("export", "regular"): 30.0, ("export", "premium"): 20.0}
    shipment_min = {("domestic", "regular"): 18.0, ("domestic", "premium"): 10.0,
                    ("export", "regular"): 8.0, ("export", "premium"): 5.0}

    variables = [{"name": f"crude_{c}", "lower": 0, "upper": available[c],
                  "unit": "kbbl/day", "type": "continuous"} for c in crude]
    variables += [{"name": f"blend_{s}_{g}", "lower": 0, "upper": None,
                   "unit": "kbbl/day", "type": "continuous"} for s in streams for g in grades]
    variables += [{"name": f"ship_{m}_{g}", "lower": 0, "upper": shipment_caps[(m, g)],
                   "unit": "kbbl/day", "type": "continuous"} for m, g in market_grade]

    objective = {f"crude_{c}": -crude_cost[c] for c in crude}
    objective.update({f"ship_{m}_{g}": market_revenue[(m, g)] for m, g in market_grade})
    constraints = [{"name": "refinery_throughput_capacity", "sense": "<=", "rhs": 88.0,
                    "coefficients": {f"crude_{c}": 1.0 for c in crude}}]

    for stream in streams:
        coeff = {f"blend_{stream}_{g}": 1.0 for g in grades}
        coeff.update({f"crude_{c}": -yields[c][stream] for c in crude})
        constraints.append({"name": f"{stream}_production_balance", "sense": "<=", "rhs": 0.0,
                            "coefficients": coeff})

    for grade in grades:
        coeff = {f"blend_{s}_{grade}": 1.0 for s in streams}
        coeff.update({f"ship_{m}_{grade}": -1.0 for m, g in market_grade if g == grade})
        constraints.append({"name": f"{grade}_product_balance", "sense": "=", "rhs": 0.0,
                            "coefficients": coeff})

    quality = {
        "regular_octane_RON_87": {"reformate": 12.5, "fcc": 2.5, "alkylate": 11.0, "lsr": -8.0},
        "premium_octane_RON_92": {"reformate": 7.5, "fcc": -2.5, "alkylate": 6.0, "lsr": -13.0},
        "regular_RVP_9_psi": {"reformate": -3.9, "fcc": -2.9, "alkylate": -8.9, "lsr": 6.1},
        "premium_RVP_10_psi": {"reformate": -4.9, "fcc": -3.9, "alkylate": -9.9, "lsr": 5.1},
    }
    quality_grade = {"regular_octane_RON_87": "regular", "premium_octane_RON_92": "premium",
                     "regular_RVP_9_psi": "regular", "premium_RVP_10_psi": "premium"}
    for name, stream_coeffs in quality.items():
        grade = quality_grade[name]
        constraints.append({"name": name, "sense": ">=" if "octane" in name else "<=",
            "rhs": 0.0, "coefficients": {f"blend_{s}_{grade}": a for s, a in stream_coeffs.items()}})

    for market, grade in market_grade:
        constraints.append({"name": f"{market}_{grade}_minimum_demand", "sense": ">=",
            "rhs": shipment_min[(market, grade)], "coefficients": {f"ship_{market}_{grade}": 1.0}})
    constraints += [
        {"name": "domestic_loading_capacity", "sense": "<=", "rhs": 62.0,
         "coefficients": {f"ship_domestic_{g}": 1.0 for g in grades}},
        {"name": "export_loading_capacity", "sense": "<=", "rhs": 44.0,
         "coefficients": {f"ship_export_{g}": 1.0 for g in grades}},
    ]

    return {
        "name": "refinery_decision_demo",
        "title": "Refinery Production & Blending — Synthetic Decision Scenario",
        "description": "Choose a three-crude slate, allocate four intermediate streams into two gasoline grades, and ship to domestic/export markets while meeting illustrative quality, demand, and capacity constraints.",
        "domain": "petroleum-refining",
        "sense": "maximize",
        "variables": variables,
        "objective": {"offset": 0.0, "coefficients": objective},
        "constraints": constraints,
        "metadata": {
            "dataset_id": "nirnaya_refinery_decision_synthetic_v1",
            "category": "industrial-style synthetic",
            "source": "Nirnaya-authored illustrative assumptions; no operational MRPL or third-party plant data",
            "license": "Nirnaya-authored synthetic data; no separate license file",
            "units_note": "Throughput in thousand barrels/day; objective contribution in illustrative currency units/day.",
            "assumptions": {
                "crude_availability_kbbl_day": available,
                "yield_fraction_by_crude_and_stream": yields,
                "crude_cost_per_kbbl": crude_cost,
                "shipment_caps_kbbl_day": {f"{m}_{g}": v for (m, g), v in shipment_caps.items()},
                "shipment_minimums_kbbl_day": {f"{m}_{g}": v for (m, g), v in shipment_min.items()},
                "quality_specs": "Linear blend-quality constraints use weighted deviation from RON and RVP thresholds; values are illustrative only.",
                "objective": "Net illustrative market contribution less crude acquisition cost; no tax, emissions, or operating complexity modeled."
            },
            "preprocessing": [],
            "random_seed": None,
        }
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("refinery_decision_demo.json")
    model = build_model()
    out.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    web_data = root / "apps/nirnaya-ui/src/data"
    web_data.mkdir(parents=True, exist_ok=True)
    (web_data / out.name).write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    scenario_path = Path(__file__).with_name("scenarios.json")
    (web_data / scenario_path.name).write_bytes(scenario_path.read_bytes())
    nnz = sum(len(row["coefficients"]) for row in model["constraints"])
    print(f"Wrote {out}: {len(model['variables'])} vars, {len(model['constraints'])} rows, {nnz} nonzeros")
