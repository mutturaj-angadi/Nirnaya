"""
examples/01_basic_model.py

Builds a small refinery-blending-style LP, validates it, compiles it to
ProblemData, inspects the numerical arrays, and round-trips it through
deterministic JSON.

Run:
    python examples/01_basic_model.py
"""

from __future__ import annotations

from nirnaya_core import ConstraintSense, Model, ObjectiveSense
from nirnaya_core.io import model_from_json, model_to_json


def build_model() -> Model:
    model = Model(name="toy_blend")

    # Two crude blends, each bounded by tank capacity.
    model.add_variable("crude_a", lower=0.0, upper=100.0)
    model.add_variable("crude_b", lower=0.0, upper=100.0)

    # Maximize profit.
    model.set_objective(
        ObjectiveSense.MAXIMIZE,
        {"crude_a": 12.5, "crude_b": 9.0},
    )

    # Total throughput cannot exceed refinery capacity.
    model.add_constraint(
        "refinery_capacity",
        {"crude_a": 1.0, "crude_b": 1.0},
        ConstraintSense.LE,
        150.0,
    )

    # Contractual minimum throughput.
    model.add_constraint(
        "min_throughput",
        {"crude_a": 1.0, "crude_b": 1.0},
        ConstraintSense.GE,
        40.0,
    )

    return model


def main() -> None:
    model = build_model()

    # Exhaustive validation -- raises ModelValidationError listing every
    # problem found, or returns None if the model is structurally valid.
    model.validate()
    print(f"Model {model.name!r} is valid: {model.is_valid()}")

    # Compile to an immutable, purely numerical standard-form snapshot.
    data = model.build()
    print("variable_names:", data.variable_names)
    print("objective sense:", data.sense)
    print("c (objective vector):", data.c)
    print("a_ub (dense view, for display only):")
    print(data.a_ub.to_dense())
    print("b_ub:", data.b_ub)
    print("lower / upper bounds:", data.lower, data.upper)

    # Deterministic JSON round trip.
    text = model_to_json(model)
    restored = model_from_json(text)
    assert restored.variable_names() == model.variable_names()
    print("\nRound-tripped through JSON successfully.")
    print(text)


if __name__ == "__main__":
    main()
