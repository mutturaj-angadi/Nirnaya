"""
Pydantic schemas for:

  * the documented JSON model format (docs/JSON_FORMAT.md)
  * REST request/response bodies (docs/API.md)

Kept separate from :mod:`nirnaya_api.model` (the ergonomic Python API) so
the wire format can be validated strictly and independently of how a user
builds a model programmatically.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .version import JSON_MODEL_FORMAT_VERSION, API_SCHEMA_VERSION


# --------------------------------------------------------------------------
# JSON model format
# --------------------------------------------------------------------------
class VariableSchema(BaseModel):
    lb: float = 0.0
    ub: float = float("inf")
    #: "continuous" (default) or "integer" / "binary" for MIP problems.
    kind: Literal["continuous", "integer", "binary"] = "continuous"

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.lb > self.ub:
            raise ValueError(f"lb ({self.lb}) must be <= ub ({self.ub})")
        return self


class ConstraintSchema(BaseModel):
    coefficients: Dict[str, float] = Field(default_factory=dict)
    sense: Literal["<=", ">=", "="]
    rhs: float

    @field_validator("coefficients")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("constraint.coefficients must be non-empty")
        return v


class ObjectiveSchema(BaseModel):
    coefficients: Dict[str, float] = Field(default_factory=dict)
    sense: Literal["min", "max"] = "min"
    #: Optional constant offset added to the objective value.
    offset: float = 0.0


class ModelSchema(BaseModel):
    """The documented on-the-wire JSON model format. See
    docs/JSON_FORMAT.md for the authoritative description and examples."""

    format_version: str = JSON_MODEL_FORMAT_VERSION
    name: Optional[str] = None
    variables: Dict[str, VariableSchema]
    constraints: Dict[str, ConstraintSchema] = Field(default_factory=dict)
    objective: ObjectiveSchema
    #: Free-form, solver-specific hints (never executed; passed through as
    #: plain data only).
    options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("variables")
    @classmethod
    def _non_empty_vars(cls, v):
        if not v:
            raise ValueError("model.variables must be non-empty")
        return v

    @model_validator(mode="after")
    def _check_references(self):
        names = set(self.variables)
        for cname, c in self.constraints.items():
            unknown = set(c.coefficients) - names
            if unknown:
                raise ValueError(
                    f"constraint {cname!r} references unknown variable(s): {sorted(unknown)}"
                )
        unknown_obj = set(self.objective.coefficients) - names
        if unknown_obj:
            raise ValueError(
                f"objective references unknown variable(s): {sorted(unknown_obj)}"
            )
        return self

    def to_plain_dict(self) -> dict:
        """Convert to the plain-dict shape the integration/orchestrator
        layer works with internally."""
        return {
            "name": self.name,
            "variables": {k: v.model_dump() for k, v in self.variables.items()},
            "constraints": {k: c.model_dump() for k, c in self.constraints.items()},
            "objective": self.objective.model_dump(),
        }


# --------------------------------------------------------------------------
# REST: solve options / request / response
# --------------------------------------------------------------------------
class SolveOptionsSchema(BaseModel):
    solver: Literal["auto", "simplex", "interior-point", "mip"] = "auto"
    backend: Literal["auto", "cpu", "gpu"] = "auto"
    device: Optional[str] = None
    feasibility_tol: float = 1e-7
    optimality_tol: float = 1e-7
    time_limit: Optional[float] = 30.0
    iteration_limit: Optional[int] = 10_000
    presolve: bool = True
    verbose: bool = False
    seed: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _dashboard_option_aliases(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "max_iterations" in data and "iteration_limit" not in data:
            data["iteration_limit"] = data["max_iterations"]
        if "time_limit_s" in data and "time_limit" not in data:
            data["time_limit"] = data["time_limit_s"]
        if "tolerance" in data:
            data.setdefault("feasibility_tol", data["tolerance"])
            data.setdefault("optimality_tol", data["tolerance"])
        return data


class SolveRequestSchema(BaseModel):
    # Kept raw here because the original dashboard sends the array-based
    # industrial-model schema while the Python API also accepts mappings.
    # The orchestrator performs strict normalization and validation.
    model: Any
    backend: Optional[Literal["auto", "cpu", "gpu"]] = None
    options: SolveOptionsSchema = Field(default_factory=SolveOptionsSchema)


class ObservabilitySchema(BaseModel):
    solve_time_s: float
    presolve_time_s: float
    iterations: int
    backend: str
    device: str
    solver: str


class DiagnosticsSchema(BaseModel):
    max_infeasibility: Optional[float] = None
    duality_gap: Optional[float] = None
    conditioning: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SolveResponseSchema(BaseModel):
    status: str
    objective_value: Optional[float]
    variable_values: Dict[str, float]
    observability: ObservabilitySchema
    diagnostics: DiagnosticsSchema
    warnings: list[str] = Field(default_factory=list)
    # Aliases retained for the supplied Part 6 dashboard contract.
    objective: Optional[float] = None
    variables: Dict[str, float] = Field(default_factory=dict)
    iterations: int = 0
    solve_time_ms: Optional[float] = None
    presolve_time_ms: Optional[float] = None
    numerical: Dict[str, Any] = Field(default_factory=dict)
    presolve: Dict[str, Any] = Field(default_factory=dict)
    constraint_residuals: Dict[str, float] = Field(default_factory=dict)


class ValidateResponseSchema(BaseModel):
    valid: bool
    format_version: str = JSON_MODEL_FORMAT_VERSION
    num_variables: int = 0
    num_constraints: int = 0
    issues: list[str] = Field(default_factory=list)


class DeviceSchema(BaseModel):
    id: str
    kind: str
    name: str
    available: bool
    detail: Dict[str, Any] = Field(default_factory=dict)


class HealthResponseSchema(BaseModel):
    status: Literal["ok", "degraded"]
    checks: Dict[str, bool]


class VersionResponseSchema(BaseModel):
    api_version: str
    json_model_format_version: str = JSON_MODEL_FORMAT_VERSION
    api_schema_version: str = API_SCHEMA_VERSION
    dependencies: Dict[str, Optional[str]]


class ErrorResponseSchema(BaseModel):
    error: Dict[str, Any]


class JobSubmitResponseSchema(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]


class JobStatusResponseSchema(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    result: Optional[SolveResponseSchema] = None
    error: Optional[Dict[str, Any]] = None
