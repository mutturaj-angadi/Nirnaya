# Nirnaya: technical judge guide

## What problem does it solve?

Nirnaya models continuous linear decisions: select production quantities, allocate scarce inputs, route flows, or blend products under linear capacity, demand, balance, and bound constraints. It returns a candidate primal solution plus a status, objective, iteration count, timing, residuals, and presolve record.

## Why optimization?

When decisions compete for limited capacity, optimization gives an explicit objective and constraints that can be inspected and changed. An LP is suitable only when the decision relationships and objective can be represented linearly and decisions are continuous. Nirnaya does not currently solve integer, nonlinear, stochastic, or mixed-integer models.

## What is actually implemented?

- Core model and immutable sparse problem-data contracts.
- Structural presolve with recorded reductions and postsolve recovery.
- Two-phase revised simplex with deterministic Bland-style pivots, sparse basis LU, iteration/deadline checks, primal feasibility and reduced-cost checks.
- API schema/size/resource validation, orchestration, and UI.
- Independent SciPy/HiGHS validation tools kept out of the production dependency path.
- GPU device discovery and numerical primitives; **no GPU LP simplex implementation**.

The solver architecture is intentionally one real algorithm behind the current solver entry point. There are no placeholder interior-point/decomposition classes or hidden reference-solver fallback.

## How do we know an answer is correct?

At `OPTIMAL`, Nirnaya checks reduced-cost optimality within configured tolerance, recomputes the objective in original variable coordinates, checks original equalities/inequalities/bounds, and postsolve recovery is checked again by the API. The external validation harness submits the same input model to SciPy/HiGHS, compares status and objective, records primal coordinate differences, and requires the API's original-model feasibility residual. The reference is not used to produce the production answer.

For infeasible/unbounded decisions, current evidence is status agreement on explicit test LPs; there is no separately published Farkas ray or unbounded direction certificate. Judges should treat these as status tests, not formal proof certificates.

## What data is used?

The refinery showcase and other business examples are industrial-style **synthetic** models written for this project. The flagship's source, license, parameter assumptions, yields, costs, market limits, quality equations, omitted factors, and generation method are in its model metadata and generator. No MRPL operational data is included or implied. Seeded synthetic tests and all dataset dimensions/provenance are in `benchmarks/manifest.json`; a read-only audit records that no data cleanup was applied.

External standard LP collections are not bundled. The Netlib LP catalog is documented as a candidate, but file-specific rights/parser review is required before redistribution. MIPLIB is primarily mixed-integer, outside this solver's supported problem class.

## How is security handled?

The API bounds JSON bytes, dimensions, nonzero entries, requested solve time, and wall-clock duration; validates schema and finite bounds/data before model construction; serializes structured errors without traceback text; and executes no user-provided commands or object deserialization. The solver enforces its own cooperative deadline and original-space numerical checks. The hard wall wrapper cannot forcibly kill a blocked Python worker thread. See `SECURITY.md` for actual limits and caveats.

## What is technically novel/useful here?

The contribution is an integrated, inspectable open engineering stack rather than a claim of a new simplex theorem: sparse model contracts, reversible presolve, a from-repository CPU revised-simplex implementation, explicit verification, reproducible comparison tooling, and scenario-focused decision UI. The current flagship shows crude/feed availability, yield/material balance, quality, product demand, throughput, shipping, and modeled contribution in one solvable continuous LP.

## Limitations and next extensions

Repeated sparse LU refactorization and Bland pivot counts are bottlenecks. No dual certificate/sensitivity output, robust numerical scaling suite beyond current presolve behavior, warm-start/basis reuse API, GPU simplex, integer solver, or real licensed plant dataset is claimed. Next work: larger public LP suite after license/parser review, basis updates with refactor safeguards, explicit dual certificates, dimensional-unit metadata, and authorized operational data validation.

## Reproduce

See the commands in `README.md`, the datasets in `benchmarks/manifest.json`, generated results in `benchmarks/results/`, and the exact methodology in `BENCHMARKS.md`. The single local demo command is `python scripts/launch_demo.py`.
