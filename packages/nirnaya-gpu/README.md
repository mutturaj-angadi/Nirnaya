# nirnaya-gpu

Device-independent hardware acceleration layer for **Nirnaya**. Provides
CPU and GPU (CUDA) backends behind a single vendor-neutral interface, so
`nirnaya-core`'s mathematical model and `nirnaya-solver`'s algorithms
never depend on a specific GPU vendor.

```
nirnaya-core   -- mathematical model (no GPU dependency)
nirnaya-solver -- solver algorithms, calls nirnaya-gpu for numerics
nirnaya-gpu    -- THIS PACKAGE: Device / DeviceBuffer / Kernel /
                  ExecutionContext abstraction, CPU + CUDA backends
```

## Install

```bash
pip install -e .                 # CPU backend only (numpy)
pip install -e ".[scipy]"        # + scipy-accelerated CPU SpMV
pip install -e ".[cuda]"         # + CUDA backend (requires a working
                                  #   CUDA driver; installs cupy)
pip install -e ".[dev]"          # scipy + pytest, for running tests
```

The GPU backend is a genuinely optional dependency: importing
`nirnaya_gpu` never requires `cupy` to be installed, and
`GPUBackend.available()` performs a real capability probe rather than
assuming presence.

## Quick start

```python
import numpy as np
from nirnaya_gpu import Session, SparseCSR, discover_all

print(discover_all().summary())

A = SparseCSR.random(n_rows=1000, n_cols=1000, density=0.01, seed=0)
x = np.random.default_rng(1).uniform(-1, 1, size=1000)

session = Session()               # GPU if genuinely available, else CPU
print(session.describe())

y = session.spmv(A, x)            # sparse matrix-vector multiply
r = session.residual(A, x, b=np.zeros(1000))
ok, violation = session.feasibility(A, x, b=np.zeros(1000), tol=1e-6, sense="le")
```

See `docs/INTEGRATION_CONTRACT.md` for how `nirnaya-solver` is expected
to call this package, and `docs/ARCHITECTURE.md` for kernel design,
memory layout, transfer-cost, synchronization, and precision details.

## Run the demo

```bash
python demo/demo.py
```

Proves, using real computation on the machine it runs on: device
discovery, automatic GPU selection when available, correct sparse
matrix-vector multiplication, CPU fallback, and (when a CUDA device is
present) CPU/GPU numerical agreement within tolerance.

## Run the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

CPU correctness and edge-case tests (`test_cpu_backend.py`,
`test_edge_cases.py`) always run. `test_correctness_cpu_vs_gpu.py` is
honestly **skipped** (reported as `skipped`, not `passed`) on machines
without a real CUDA device.

## Run the benchmarks

```bash
python benchmarks/bench_spmv.py          # SpMV, small/medium/large
python benchmarks/bench_vector_ops.py    # dot/axpy/scale/reduce
python benchmarks/run_benchmarks.py      # everything + end-to-end proxy
```

Every number printed is measured on the run that produced it -- these
scripts contain no pre-recorded or estimated results. GPU rows are
reported as unavailable, not omitted or guessed, on non-CUDA machines.

## What "no fake GPU" means, concretely

* `Device.is_gpu` and `Device.model_name` only ever come from an
  actual `cupy.cuda.runtime` query.
* `GPUBackend()` raises `DeviceUnavailableError` (not a silent CPU
  substitution) when CUDA isn't real; `nirnaya_gpu.registry.get_backend()`
  is the one place that *chooses* to fall back to CPU, and it does so
  visibly (`Session.is_gpu` / `backend.name` reflect the true choice).
* Benchmarks and the demo compute and print real numbers on every run.
