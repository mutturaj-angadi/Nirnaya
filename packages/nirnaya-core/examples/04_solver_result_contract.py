"""
examples/04_solver_result_contract.py

Nirnaya Core ships no solver. This example shows the *shape* a Part 2+
solver backend is expected to produce: given a ProblemData, return a
SolverResult whose arrays are index-aligned to the ProblemData's
variable/constraint name ordering, and whose status accurately reflects
what happened.

This is illustrative wiring, not a real solver: `fake_solve` below simply
demonstrates the expected call signature and result shape using a known
feasible point, and self-checks that point against the model's own
constraint/bound definitions using Constraint.residual and
LinearObjective.evaluate -- exactly the sanity-check a real backend should
perform before returning SolverStatus.OPTIMAL.

Run:
    python examples/04_solver_result_contract.py
"""

from __future__ import annotations

from nirnaya_core import ConstraintSense, Model, ObjectiveSense
from nirnaya_core.model import ProblemData
from nirnaya_core.solution import OptimizationStatistics, SolverResult
from nirnaya_core.status import SolverStatus


def build_model() -> Model:
    model = Model(name="two_variable_lp")
    model.add_variable("x", lower=0.0)
    model.add_variable("y", lower=0.0)
    model.set_objective(ObjectiveSense.MAXIMIZE, {"x": 3.0, "y": 2.0})
    model.add_constraint("cap", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 4.0)
    return model


def fake_solve(data: ProblemData, model: Model) -> SolverResult:
    """Stand-in for a Part 2+ solver backend's entry point.

    A real backend would run simplex/interior-point iterations here. This
    stub instead demonstrates the contract: it uses a point that is known
    (by hand) to be optimal for this specific toy problem, verifies it
    against the *model's* own constraint/objective definitions (not
    hardcoded arithmetic), and packages the result in the standard shape.
    """
    candidate = {"x": 4.0, "y": 0.0}  # known optimal for this toy LP

    # A real backend should perform exactly this kind of self-check
    # before claiming SolverStatus.OPTIMAL.
    tol = data.config.feasibility_tol
    for constraint in model.constraints:
        residual = constraint.residual(candidate)
        if constraint.sense == ConstraintSense.LE and residual > tol:
            raise AssertionError(f"candidate violates {constraint.name}")
        if constraint.sense == ConstraintSense.GE and residual < -tol:
            raise AssertionError(f"candidate violates {constraint.name}")
        if constraint.sense == ConstraintSense.EQ and abs(residual) > tol:
            raise AssertionError(f"candidate violates {constraint.name}")

    objective_value = model.objective.evaluate(candidate)  # type: ignore[union-attr]

    primal = tuple(candidate[name] for name in data.variable_names)
    all_constraint_names = data.constraint_names_eq + data.constraint_names_ub
    residuals = tuple(
        model.get_constraint(name).residual(candidate) for name in all_constraint_names
    )

    return SolverResult(
        status=SolverStatus.OPTIMAL,
        variable_names=data.variable_names,
        constraint_names=all_constraint_names,
        primal=primal,
        constraint_residuals=residuals,
        statistics=OptimizationStatistics(
            iterations=1,
            primal_objective=objective_value,
            solver_name="example-fake-solver",
        ),
        message="Illustrative result only -- not a real solve.",
    )


def main() -> None:
    model = build_model()
    data = model.build()
    result = fake_solve(data, model)

    print("status:", result.status)
    print("is_optimal:", result.is_optimal())
    print("x =", result.primal_value("x"))
    print("y =", result.primal_value("y"))
    print("objective:", result.statistics.primal_objective)
    print("constraint residuals:", dict(zip(result.constraint_names, result.constraint_residuals)))


if __name__ == "__main__":
    main()
