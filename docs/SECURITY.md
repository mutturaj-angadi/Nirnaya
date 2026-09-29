# Security and input validation

## API limits in the current code

`packages/nirnaya-api/src/nirnaya_api/security.py` defines the defaults:

- JSON request body: 5 MiB.
- Variables: 200,000.
- Constraints: 200,000.
- Coefficient entries: 5,000,000.
- Requested solve time: at most 600 seconds.
- Outer solve wall-clock wrapper: 900 seconds.
- Solver also receives a cooperative time and iteration limit.

These are resource bounds, not a promise that a model of those maximum dimensions will solve within them. Clients should request lower workload budgets appropriate to their deployment.

## Validation and serialization

Request parsing uses typed API schemas and a single JSON-to-core model adapter. Core validates finite coefficients, bounds, variable types, and model structure. Malformed/oversized inputs return structured errors. Unexpected exceptions are wrapped before response serialization; traceback text is not returned. API code does not evaluate user input, launch shell commands, deserialize Python objects, or accept caller-selected filesystem paths. Benchmark and validation endpoints read only fixed, repository-local report paths and return compact JSON projections.

The solver uses deterministic pivots. Before success it checks reduced-cost feasibility and original-model primal feasibility; presolve recovery is checked again by the API. These checks detect invalid solver results but do not make the service a sandbox for hostile workloads.

## Operational boundaries

- The optional local static UI server binds to loopback by default. `scripts/launch_demo.py` is for local judge/demo use, not hardened production hosting.
- The outer Python wall-clock worker wrapper cannot forcibly stop a blocked thread; solver deadlines provide cooperative cancellation.
- No authentication, tenant isolation, TLS termination, request-rate limiting, or process/container isolation is implemented by this repository. Deploy behind an appropriately configured service boundary before exposing beyond a trusted local network.
- `apps/nirnaya-ui/validation/reference_solver.py` is independent validation code using SciPy/HiGHS. It is not a production fallback and must not be exposed as a solve backend.
- GPU discovery/primitives do not make the LP solver a GPU solver. GPU solve requests fail explicitly in this release.
- The canonical demo launcher configures browser CORS for its exact UI origin. Standalone or deployed API instances must set `NIRNAYA_CORS_ORIGINS` to an explicit comma-separated allowlist; the API does not enable wildcard origins.

Security/input tests are run with the API and integration test commands in the root README.
