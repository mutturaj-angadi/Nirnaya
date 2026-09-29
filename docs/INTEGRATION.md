# Integration contracts

Core's public `Model`, `ConstraintSense`, `ObjectiveSense`, `VariableType`, `ProblemData`, `NumericalConfig`, `SolverResult`, and `SolverStatus` are authoritative. The supplied presolve adapter targeted an older assumed `LPModel`/`Sense` contract; it now builds presolve state from core `ProblemData` and reconstructs reduced core `Model` instances. Postsolve consumes and returns core `SolverResult`.

The API uses one adapter for JSON construction, presolve, solve, and recovery. GPU device discovery uses `nirnaya_gpu.discover_all()`. GPU primitives currently do not participate in simplex execution; the result reports CPU. Validation-only reference solving is isolated under the UI tree.
