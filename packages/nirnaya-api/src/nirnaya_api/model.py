"""
The Python API: create model -> add variables -> add constraints ->
set objective -> solve -> inspect result.

This is a thin, ergonomic builder over the JSON model format
(docs/JSON_FORMAT.md). Calling :meth:`Model.solve` hands the resulting
dict to :func:`nirnaya_api.orchestrator.solve_model`, the same function
the CLI and REST API use - so behavior is identical across all three
interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .errors import ValidationError
from .options import SolveOptions
from .orchestrator import solve_model, validate_model
from .security import DEFAULT_LIMITS, SecurityLimits
from .solution import Solution


@dataclass
class Variable:
    name: str
    lb: float = 0.0
    ub: float = float("inf")
    kind: str = "continuous"


@dataclass
class Constraint:
    name: str
    coefficients: Dict[str, float]
    sense: str
    rhs: float


@dataclass
class Objective:
    coefficients: Dict[str, float] = field(default_factory=dict)
    sense: str = "min"
    offset: float = 0.0


class Model:
    """A mutable, in-progress optimization model.

    Example
    -------
    >>> m = Model("diet")
    >>> m.add_variable("bread", lb=0, ub=5)
    >>> m.add_variable("milk", lb=0, ub=5)
    >>> m.add_constraint("budget", {"bread": 2, "milk": 3}, "<=", 10)
    >>> m.set_objective({"bread": 4, "milk": 3}, sense="max")
    >>> result = m.solve()
    >>> result.status
    <SolveStatus.OPTIMAL: 'optimal'>
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._variables: Dict[str, Variable] = {}
        self._constraints: Dict[str, Constraint] = {}
        self._objective: Optional[Objective] = None

    # -- building --------------------------------------------------------
    def add_variable(
        self,
        name: str,
        lb: float = 0.0,
        ub: float = float("inf"),
        kind: str = "continuous",
    ) -> Variable:
        if name in self._variables:
            raise ValidationError(f"Variable {name!r} already exists")
        v = Variable(name=name, lb=lb, ub=ub, kind=kind)
        self._variables[name] = v
        return v

    def add_constraint(
        self,
        name: str,
        coefficients: Dict[str, float],
        sense: str,
        rhs: float,
    ) -> Constraint:
        if name in self._constraints:
            raise ValidationError(f"Constraint {name!r} already exists")
        c = Constraint(name=name, coefficients=dict(coefficients), sense=sense, rhs=rhs)
        self._constraints[name] = c
        return c

    def set_objective(
        self,
        coefficients: Dict[str, float],
        sense: str = "min",
        offset: float = 0.0,
    ) -> Objective:
        self._objective = Objective(coefficients=dict(coefficients), sense=sense, offset=offset)
        return self._objective

    def remove_variable(self, name: str) -> None:
        self._variables.pop(name, None)

    def remove_constraint(self, name: str) -> None:
        self._constraints.pop(name, None)

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to the documented JSON model format
        (docs/JSON_FORMAT.md)."""
        if self._objective is None:
            raise ValidationError("Model has no objective set; call set_objective() first")
        return {
            "name": self.name,
            "variables": {
                v.name: {"lb": v.lb, "ub": v.ub, "kind": v.kind}
                for v in self._variables.values()
            },
            "constraints": {
                c.name: {
                    "coefficients": c.coefficients,
                    "sense": c.sense,
                    "rhs": c.rhs,
                }
                for c in self._constraints.values()
            },
            "objective": {
                "coefficients": self._objective.coefficients,
                "sense": self._objective.sense,
                "offset": self._objective.offset,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        m = cls(name=data.get("name"))
        for vname, vspec in (data.get("variables") or {}).items():
            m.add_variable(
                vname,
                lb=vspec.get("lb", 0.0),
                ub=vspec.get("ub", float("inf")),
                kind=vspec.get("kind", "continuous"),
            )
        for cname, cspec in (data.get("constraints") or {}).items():
            m.add_constraint(
                cname, cspec["coefficients"], cspec["sense"], cspec["rhs"]
            )
        obj = data.get("objective")
        if obj is not None:
            m.set_objective(
                obj.get("coefficients", {}), obj.get("sense", "min"), obj.get("offset", 0.0)
            )
        return m

    @classmethod
    def from_json(cls, text: str) -> "Model":
        import json

        return cls.from_dict(json.loads(text))

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)

    # -- validation / solving ----------------------------------------------
    def validate(self, limits: SecurityLimits = DEFAULT_LIMITS) -> dict:
        """Format + size validation without invoking any solver. Returns a
        dict: ``{"valid": bool, "issues": [...], ...}``."""
        return validate_model(self.to_dict(), limits)

    def solve(
        self,
        solver: str = "auto",
        backend: str = "auto",
        device: Optional[str] = None,
        feasibility_tol: float = 1e-7,
        optimality_tol: float = 1e-7,
        time_limit: Optional[float] = 30.0,
        iteration_limit: Optional[int] = 10_000,
        presolve: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        limits: SecurityLimits = DEFAULT_LIMITS,
    ) -> Solution:
        options = SolveOptions(
            solver=solver,
            backend=backend,
            device=device,
            feasibility_tol=feasibility_tol,
            optimality_tol=optimality_tol,
            time_limit=time_limit,
            iteration_limit=iteration_limit,
            presolve=presolve,
            verbose=verbose,
            seed=seed,
        )
        return solve_model(self.to_dict(), options, limits)

    def __repr__(self) -> str:
        return (
            f"Model(name={self.name!r}, variables={len(self._variables)}, "
            f"constraints={len(self._constraints)})"
        )
