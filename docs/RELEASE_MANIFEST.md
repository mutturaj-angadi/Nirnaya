# Release manifest

**Artifact:** Nirnaya SIH 2026 continuous LP decision-support demo  
**Freeze date:** 2026-09-28  
**Scope:** Reproducible local demo and engineering handoff. This is not an operational refinery system.

## Architecture and algorithm

Validated continuous LP input flows through `nirnaya-core`, reversible presolve, the repository-authored two-phase revised-simplex CPU solver, postsolve recovery, and original-model verification. The solver uses sparse model matrices, a sparse LU base factorization and bounded product-form eta basis updates with residual checks and refactorization. SciPy/HiGHS is used in isolated validation and benchmark tooling only; it is not a production fallback. The `nirnaya-gpu` package provides device discovery and numerical primitives, but the LP simplex path is CPU-only.

The refinery scenario is **MRPL-inspired synthetic data**. It contains no MRPL data, live plant data, or operational recommendations. Provenance and dataset audit are documented in `DATASETS.md`.

## Current verification evidence

- Full root suite against the live current source API: **283 passed, 8 skipped**, no failures. All skips require unavailable CUDA/CuPy hardware.
- Both live UI/API E2E trees: **12 passed**. Independent API reference suite: **5/5 passed**.
- Fresh release validation: **12/12 passed**; the sparse solve completed after 8,436 iterations in 2.70 s solver time under the unchanged 30-second deadline. Its maximum original-model violation was `5.68e-14`, and objective difference from independent HiGHS was `1.82e-12`. See [NOC10K.md](NOC10K.md) for further details.
- In-process benchmark matrix: **18 records**; 16 optimal, one infeasible, one unbounded, zero time limits, numerical errors, or execution errors.
- NOC-10K: 10,000/10,000 PASS; status agreement 10,000/10,000; objective and original-space feasibility agreement 9,250/9,250 optimal cases. The isolated holdout is 1,000/1,000 PASS, including 925/925 optimal reference objective/feasibility checks. Full metrics and limitations are in [NOC10K.md](NOC10K.md).
- Five packages installed editably in an isolated temporary venv with system dependencies. A fully isolated network build could not fetch build requirements in this restricted-network environment.
- Workspace source imports were verified. Security and provenance review found no production reference optimizer call or GPU LP claim. The sparse dataset and generator now state that its solve is CPU-only; GPU primitives do not solve it.

## Known limitations

- CPU-only LP simplex; no GPU simplex, integer or nonlinear solving, dual certificates, or sensitivity analysis.
- Eta basis updates have bounded depth and periodic residual checks. The exact sparse case completed repeatedly on the current source within 30 seconds, but no service-level timing guarantee is claimed because host load varies.
- No admission control for concurrent heavy API solves.
- The standalone historical presolve package test suite depends on a removed API and is excluded from current root `testpaths`; it was not counted as passing.
- Browser evidence is a local Edge smoke test, not cross-browser, accessibility, or deployment verification.
- Eight GPU tests skip because CUDA/CuPy device execution is unavailable. No GPU execution is claimed.

## Evidence and commands

See [VALIDATION.md](VALIDATION.md), [BENCHMARKS.md](BENCHMARKS.md), [DEMO.md](DEMO.md), [SECURITY.md](SECURITY.md), and [SOLVER_ALGORITHM.md](SOLVER_ALGORITHM.md). Reproduction commands are in the Validation and Demo guides. Current machine-readable results and failed runs are preserved beneath `../benchmarks/results/`; do not replace archived outcomes with fabricated values.
