"""
examples/02_validation_errors.py

Demonstrates that Model.validate() aggregates every structural problem it
finds into a single ModelValidationError, rather than stopping at the
first one -- useful for surfacing all issues to a user at once.

Run:
    python examples/02_validation_errors.py
"""

from __future__ import annotations

from nirnaya_core import ConstraintSense, Model, ObjectiveSense
from nirnaya_core.validation import ModelValidationError


def main() -> None:
    model = Model(name="broken_model")

    # Only one variable is declared...
    model.add_variable("x", lower=0.0)

    # ...but the objective and a constraint both reference an unknown
    # variable "y". Two distinct problems, in two different places.
    model.set_objective(ObjectiveSense.MINIMIZE, {"x": 1.0, "y": 2.0})
    model.add_constraint("c1", {"y": 1.0}, ConstraintSense.LE, 10.0)

    try:
        model.build()
    except ModelValidationError as exc:
        print(f"Model failed validation with {len(exc.errors)} error(s):\n")
        for err in exc.errors:
            print(f"  [{err.code}] {err.message}")
        print("\nMachine-readable form (exc.to_dict()):")
        print(exc.to_dict())


if __name__ == "__main__":
    main()
