"""Solve options shared by the Python API, CLI, and REST API.

This is the single definition of "what can be tuned about a solve" so the
three interfaces stay in lockstep. It intentionally only configures how
Parts 1-4 are invoked (solver choice, backend/device choice, tolerances,
limits, diagnostics) - it never changes solving *logic*.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Literal

SolverChoice = Literal["auto", "simplex", "interior-point", "mip"]
BackendChoice = Literal["auto", "cpu", "gpu"]


@dataclass
class SolveOptions:
    #: Which algorithm family in nirnaya-solver to request. "auto" lets
    #: nirnaya-solver pick based on problem characteristics.
    solver: SolverChoice = "auto"

    #: Compute backend. "auto" resolves to the CPU in this release; GPU
    #: simplex is not implemented. Explicit GPU requests are rejected.
    backend: BackendChoice = "auto"

    #: Specific device id to target (e.g. "gpu:0"). None = let backend
    #: selection decide.
    device: Optional[str] = None

    #: Numerical tolerances.
    feasibility_tol: float = 1e-7
    optimality_tol: float = 1e-7

    #: Wall-clock time limit for the solve, in seconds. Enforced by this
    #: layer regardless of what the solver itself does internally.
    time_limit: Optional[float] = 30.0

    #: Iteration limit passed through to nirnaya-solver.
    iteration_limit: Optional[int] = 10_000

    #: Whether to run nirnaya-presolve before handing the model to the
    #: solver.
    presolve: bool = True

    #: Emit detailed, structured diagnostics (per-iteration / numerical
    #: conditioning info) in the returned Solution, at some performance
    #: cost.
    verbose: bool = False

    #: Random seed forwarded to the solver, for reproducibility.
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SolveOptions":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            from .errors import ValidationError

            raise ValidationError(
                f"Unknown option(s): {sorted(unknown)}",
                details={"unknown_options": sorted(unknown), "allowed": sorted(known)},
            )
        return cls(**data)

    def validate(self) -> None:
        from .errors import ValidationError

        if self.time_limit is not None and self.time_limit <= 0:
            raise ValidationError("options.time_limit must be > 0 if set")
        if self.iteration_limit is not None and self.iteration_limit <= 0:
            raise ValidationError("options.iteration_limit must be > 0 if set")
        if self.feasibility_tol <= 0 or self.optimality_tol <= 0:
            raise ValidationError("tolerances must be > 0")
        if self.solver not in ("auto", "simplex", "interior-point", "mip"):
            raise ValidationError(f"Unknown solver choice: {self.solver!r}")
        if self.backend not in ("auto", "cpu", "gpu"):
            raise ValidationError(f"Unknown backend choice: {self.backend!r}")
