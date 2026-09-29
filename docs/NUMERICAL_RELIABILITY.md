# Numerical reliability and limits

## Checks before `optimal`

- Core validates model shapes, finite coefficients/bounds, variable type support, and bounds.
- Phase I identifies infeasibility when its artificial objective exceeds the configured feasibility tolerance.
- Phase II uses reduced costs and pivot tolerances for optimality and deterministic unbounded checks.
- Sparse LU singularity/non-finite results, significant basic infeasibility, reduced-cost violations, and original-space violations are reported as numerical errors, never relabeled as an optimum.
- The solver recomputes the objective from the original model. API postsolve recovery is checked again against original equalities, inequalities, and bounds.

## Determinism and instrumentation

Pivot choices use smallest-index Bland rules with deterministic tie-breaking. Solver statistics include Phase I/II, sparse LU factorization, factor solves, product-form eta updates, periodic residual checks, pricing, ratio-test, and artificial-basis cleanup times. A time/iteration limit can return no primal objective; diagnostics do not invent one. Phase timings overlap: factorization/pricing/ratio work occurs within a simplex phase.

## Known numerical limits

- The basis remains sparse. Product-form eta updates use a bounded chain and periodic checks against the current basis with configured primal and dual tolerances; small pivots, residual failures, or the update window trigger sparse LU refactorization. There is no cross-solve basis warm start.
- Bland pricing is deterministic and anti-cycling but can require many iterations on degenerate LPs.
- No validated dual vector, Farkas infeasibility certificate, unbounded ray, or sensitivity/shadow-price report is published.
- The implementation uses configured feasibility, optimality, bound, pivot, zero, iteration, and time policies, but does not provide a general condition estimator or comprehensive float32 reliability evidence.
- Large coefficient ranges, very ill-conditioned bases, and models requiring aggressive scaling can return numerical failure. Current presolve cleanup/scaling behavior is recorded; do not interpret this as a full numerical-equilibration implementation.
- The API's outer wall-clock wrapper is cooperative and cannot forcibly terminate a blocked Python worker thread; the solver has its own deadline check.

The seeded ill-scaled synthetic model is included in current validation. It records this machine/run's actual status; passing one constructed scaling case does not establish robust performance on arbitrary ill-conditioned matrices.
