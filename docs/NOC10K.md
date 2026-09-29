# NOC-10K synthetic LP evaluation

## Purpose and scope

NOC-10K is a deterministic corpus of **synthetic industrial-style optimization instances** for evaluating solver correctness, robustness, performance, and presolve. It is not ML training. It contains no real industrial or MRPL operational data. Only the separate flagship may be called “MRPL-inspired synthetic.” The Nirnaya production solver remains the repository-authored CPU two-phase revised simplex; SciPy/HiGHS is a separate reference labeler, never a production fallback.

The corpus manifest is `datasets/noc10k/manifest.jsonl.gz`. Full LPs are regenerated from the recorded family, seed, family index, and generator version; the compact manifest does not store opaque random matrices. SHA-256 is over canonical sorted-key JSON of the LP payload before its metadata block. Regeneration validates every model against the production API schema. Deterministic family-rank assignment isolates 8,000 development, 1,000 validation, and 1,000 holdout cases before solver outcomes exist.

## Family allocation

The requested family counts in the program brief sum to 9,750, although the brief also requires exactly 10,000. The corpus adds that 250-case arithmetic difference to generic industrial-style synthetic production/resource-allocation cases. This is a documented count correction, not a change based on observed solver outcomes.

| Family | Instances |
|---|---:|
| Small dense | 1,000 |
| Small sparse | 1,000 |
| Medium sparse | 2,000 |
| Large sparse | 1,500 |
| Degenerate | 750 |
| Ill-conditioned | 500 |
| Equality-heavy | 500 |
| Bound-heavy | 500 |
| Presolve-heavy | 500 |
| Infeasible | 500 |
| Unbounded | 250 |
| Synthetic industrial-style | 1,000 |
| **Total** | **10,000** |

Manifest audit: 10,000 unique IDs, 10,000 unique model hashes; split counts are exactly 8,000/1,000/1,000. Structural classes are generated intentionally: feasible bounded random packing-style LPs, contradictory bound constraints for infeasible cases, no constraining rows for unbounded cases, duplicated rows for degeneracy, scaled rows for conditioning stress, equalities, finite/fixed bounds, singleton/fixed-variable presolve cases, and generic production/resource planning.

## Method and data recorded

`scripts/generate_noc10k.py` produces the compact deterministic corpus. `scripts/run_noc10k.py` regenerates each LP, checks its manifest hash, calls the production API orchestrator with CPU, presolve, original-model verification, the existing 30-second solver limit and existing tolerances, then calls the independent HiGHS reference. Results are flushed in manifest order; worker count is bounded and recorded. Set `--workers 1` for comparable sequential timings. Failures/timeouts remain in each JSONL run; analysis never merges away an earlier attempt.

Records include status, objective, total and Phase I/II iterations, solver and presolve time, LU factorization, factor solve, eta update, residual-check, pricing and ratio-test timings, original-model constraint and bound violations, presolve dimensions/reductions, reference status/objective/feasibility/runtime/version, classification, model hash, timestamps, machine and software data. Postsolve sub-time and per-instance memory remain null because the production API does not expose them and no portable RSS sampler is enabled; the report does not estimate them.

The objective agreement threshold is the existing `1e-7` optimality tolerance with a scale factor `max(1, |reference objective|)`. Primal constraint and bound feasibility are independently recomputed in original model space against the existing absolute `1e-7` feasibility/bound criterion. Different optimal vectors are accepted when both are feasible and objectives agree. SciPy's HiGHS time limit is 30 seconds; time limit/numerical difficulty is recorded as reference unavailable with the raw reason.

## Reproduction commands

Run from the repository root after installing the local package dependencies:

```powershell
python scripts/generate_noc10k.py
python scripts/run_noc10k.py --workers 1
python scripts/analyze_noc10k.py
```

Each command accepts `--seed`, `--family`, `--count`, `--start`, and `--end`; the runner accepts `--workers` (default at most four). Generator `--workers` is accepted for CLI consistency, while deterministic generation itself is serial. `--start` is inclusive and `--end` exclusive within the selected manifest. Run outputs are unique timestamped JSONL files in `benchmarks/results/noc10k/runs/`; specify `--output` to choose an explicit path. Completed run files may be gzip-compressed with their line records preserved as `.jsonl.gz`; the analyzer reads both forms. Analysis defaults to the newest timestamped run but should be given `--input` for an auditable selection. A partial run is summarized as partial, not extrapolated to 10,000.

## Run status

The canonical single-worker run `benchmarks/results/noc10k/run_eta_final.jsonl` completed all 10,000 records. Analyzer artifacts are `summary.json`, `family_summary.csv`, `failures.csv`, and `holdout_summary.json` in the same directory. All 10,000 records classified PASS: 9,250 optimal, 500 infeasible, and 250 unbounded. Status agreement was 10,000/10,000; all 9,250 optimal cases matched the independent objective and original-space feasibility checks. There were no timeouts, numerical errors, execution errors, or unavailable references. The isolated 1,000-case holdout classified PASS 1,000/1,000, with 925/925 optimal objective and feasibility comparisons and no timeouts or unavailable references. Holdout metrics are reported for evaluation only and were not used to choose the solver update.

## Performance findings and limitations

The existing large sparse release LP is separate from this corpus. Profiling showed that refactorizing sparse LU at every pivot dominated the previous run. The selected optimization is a bounded product-form eta basis update with periodic residual checks and refactorization; it leaves pricing, ratio testing, tolerances, pivot selection and original-model verification unchanged. On a fixed 100-case large-sparse development sample, median solve time changed from 70.5 ms to 49.5 ms (same per-case iteration counts; host timings are variable). The original 1,440-variable / 300-row / 2,880-nonzero sparse model completed in 2.773 s solver time after 8,436 iterations; API wall time was 3.045 s, presolve 0.083 s, Phase I 0.122 s, Phase II 2.635 s, with 265 LU factorizations, 8,173 stored basis updates, 0.196 s pricing and 0.152 s ratio testing. It returned optimal with maximum original-model violation `5.68e-14`; independent HiGHS returned optimal with objective difference `1.82e-12`. All 1,440 primal values were compared: 1,430 match within `1e-7`; the other 10 are zero-objective flow variables on an alternate optimum. The two primal vectors have maximum original constraint/bound violation below `5e-14` and equal objective to within `2e-12`. These timings are one host run, not a guarantee.

Across NOC-10K, solve-time p50/p90/p95/p99 was 10.6/55.6/76.7/138.7 ms. The large-sparse family was the slowest (50.1/117.6/151.0/210.7 ms). On that family, factor solve and eta updates together remain the largest instrumented numerical costs; pricing and ratio tests are smaller. Its median presolve removed 18 variables and no rows, so presolve output quality is limited for these generated large sparse instances. The sparse release case's presolve removed one row and one nonzero. Matrix operations and basis storage remain sparse; no dense basis conversion was introduced.

This corpus contains generated LPs, not a representative sample of all real-world LPs. The SciPy reference implementation densifies some input structures; reference failures are retained as unavailable and are not converted to answers. Shared-host timing variation, threading, reference overhead, lack of memory measurements, and no solver warm-start support limit performance conclusions. Holdout results must be reported separately and must not guide tuning. The successful exact sparse run satisfies the 30-second gate in this environment; earlier archived runs timed out under a 30-second deadline, so reliable completion under load is not guaranteed.
