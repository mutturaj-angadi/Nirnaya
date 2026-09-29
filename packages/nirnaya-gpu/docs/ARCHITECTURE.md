# Architecture

## Layers

```
nirnaya_gpu/
  device.py         Device, DeviceBuffer, ExecutionContext, Kernel -- the
                     vendor-neutral contract everything else is built on.
  sparse.py          SparseCSR -- the host-side wire format for sparse
                     matrices (see "Memory layout" below).
  backends/
    base.py           Backend abstract interface.
    cpu.py            CPUBackend: numpy (+ optional scipy.sparse for SpMV).
    gpu.py            GPUBackend: cupy + cupyx.scipy.sparse (CUDA only).
  registry.py         Real device discovery + backend selection policy.
  ops.py              Session: numpy-in/numpy-out convenience wrapper
                     over Backend/DeviceBuffer for less perf-critical call
                     sites (see docs/INTEGRATION_CONTRACT.md).
```

A future HIP or SYCL backend is added as `backends/hip.py` implementing
`Backend`, plus a branch in `registry.py`'s discovery/selection. No
change to `device.py`, `sparse.py`, `ops.py`, or solver code is needed.

## Kernel design

### Sparse matrix-vector multiplication (SpMV)

This is the dominant operation for the solver's use case (repeated
`A @ x` against a fixed or slowly-changing constraint matrix), so it
gets the most attention.

* **CPU**: delegates to `scipy.sparse.csr_matrix.dot`, which is a
  well-tested, cache-blocked C implementation, when scipy is
  installed. When it is not, `_csr_matvec_numpy` computes it via
  `numpy.bincount` over a `(row_id, value)` pair per nonzero -- one
  thread-parallel-friendly vectorized pass, no Python-level loop over
  nonzeros or rows. This keeps `nirnaya-gpu` usable with only `numpy`
  as a hard dependency.
* **GPU**: delegates to `cupyx.scipy.sparse.csr_matrix.dot`, which
  dispatches to NVIDIA's cuSPARSE. This is the standard
  "one-thread-block-per-row" (or warp-per-row, chosen internally by
  cuSPARSE based on row density) strategy for CSR SpMV: each block
  reads a contiguous run of `(indices[k], data[k])` for its row via
  `indptr[row]:indptr[row+1]`, which is why CSR (not COO) is the
  wire format -- COO would require a scatter-add and explicit
  synchronization across blocks to accumulate per-row sums; CSR needs
  none.

### Vector kernels (dot, axpy, scale, elementwise, reduce)

These map directly onto BLAS-1-style operations:
* CPU: numpy ufuncs (`np.dot`, in-place `+=`/`*=`, `np.sum`/`np.max`/...).
* GPU: cupy ufuncs, which dispatch to cuBLAS (`dot`) or fused
  elementwise CUDA kernels (`+=`, `*=`, `maximum`, ...) and a
  tree-reduction kernel for `sum`/`max`/`norm2`.

### Feasibility check (`feasibility_check`)

Computes `A @ x` once, then an elementwise violation amount
(`max(0, Ax - b - tol)` for `<=`, etc.) with a single reduction
(`all(violation <= tol)`) to decide overall feasibility. This is
"parallel" in the sense that every row's violation is computed
independently and simultaneously (a single elementwise kernel + one
reduction), rather than a row-by-row Python loop with early exit --
which matters on GPU (early exit defeats parallelism) and is still
faster on CPU (vectorized) than a Python loop.

### `argmin` / `ratio_test` (basis-related kernels)

Two primitives an LP-style solver's basis-update step is built from:

* `argmin(x)`: index and value of the minimum entry -- used for
  entering-variable selection (most negative reduced cost).
* `ratio_test(x, direction, tol)`: over all `i` with `direction[i] >
  tol`, `argmin_i x[i] / direction[i]` -- the classic minimum-ratio
  test for leaving-variable selection, returning `(None, inf)` when no
  index qualifies (unbounded direction).

Both are single-pass reductions (`argmin`/masked-argmin), so they get
the same GPU parallel-reduction treatment as `reduce()`. `nirnaya-gpu`
deliberately stops at these primitives; full basis factorization/LU
updates are a `nirnaya-solver`-level concern (see
`docs/INTEGRATION_CONTRACT.md`, section 7).

## Memory layout

`SparseCSR` (Compressed Sparse Row) is the one representation this
package uses at the host/wire boundary:

```
indptr:  int, length n_rows+1     -- indptr[r]:indptr[r+1] gives the
                                      index range into `indices`/`data`
                                      for row r.
indices: int, length nnz          -- column index of each nonzero.
data:    float, length nnz        -- value of each nonzero.
```

CSR was chosen over COO or CSC because:
* **Row-major access matches SpMV**, the dominant operation. A GPU
  kernel can assign one thread block per row and read
  `indices[indptr[r]:indptr[r+1]]` as a contiguous, coalesced memory
  access.
* **O(1) row-length lookup** (`indptr[r+1] - indptr[r]`) without a
  separate pass, which both `argmin`-style row iteration and load
  balancing (row splitting for very uneven nnz-per-row matrices, handled
  internally by cuSPARSE) depend on.
* It is the **native format of both scipy.sparse and
  cupyx.scipy.sparse**, so converting a `SparseCSR` to either backend's
  representation is a direct data copy (three arrays, no restructuring
  pass) rather than a re-parse -- see `CPUBackend.upload_sparse` /
  `GPUBackend.upload_sparse`.

CSC (better for matvec with the *transpose*, i.e. `A.T @ x`) and COO
(simpler to build incrementally, e.g. while assembling constraints) are
both trivially derivable from CSR (`scipy`/`cupy` provide `.tocsc()`)
and are not exposed as a *second* wire format, to keep the contract in
`docs/INTEGRATION_CONTRACT.md` narrow.

## Transfer costs

Host&harr;device transfer is the single largest fixed cost in any GPU
call and is deliberately made visible rather than hidden:

* `ExecutionContext.transfer_stats` accumulates real
  `(bytes_transferred, seconds)` for every `to_device`/`to_host` call
  issued through that context (see `GPUBackend.to_device`/`to_host`,
  which time the copy and call `ctx._record_transfer`).
* `Session.upload_matrix()` is a *separate* call from `Session.spmv()`
  specifically so a matrix that is reused across many iterations (the
  normal solver access pattern -- the constraint matrix doesn't change
  every iteration) is transferred **once**, not once per `spmv` call.
  `benchmarks/bench_spmv.py` times matrix upload and steady-state
  kernel execution separately for exactly this reason.
* For small problems, transfer cost dominates and a GPU can be
  *slower* end-to-end than the CPU backend despite faster raw compute
  -- this is a real, measured phenomenon (see "CPU/GPU tradeoffs"
  below and the `small` size class in `benchmarks/bench_spmv.py`), not
  a hypothetical caveat.

## Synchronization

* **CPU backend**: execution is synchronous by construction (numpy/
  scipy calls block until done), so `CPUContext.synchronize()` is a
  documented no-op. It still exists so solver code written against
  `ExecutionContext` doesn't need a backend-specific branch.
* **GPU backend**: kernel launches and `cupy` operations are
  asynchronous with respect to the host by default. `CUDAContext` owns
  a non-blocking `cupy.cuda.Stream`; `synchronize()` calls both
  `stream.synchronize()` and `cupy.cuda.runtime.deviceSynchronize()`.
  Any `Backend` method that must return a host-visible scalar (`dot`,
  `reduce`, `argmin`, `ratio_test`, `feasibility_check`'s boolean)
  synchronizes internally before reading the value back, so callers
  never get a stale/incomplete result. Methods that return a
  `DeviceBuffer` (`spmv`, `axpy`, `scale`, `elementwise`, `residual`)
  do **not** force a synchronize, so a chain of such calls can pipeline
  without unnecessary round-trips -- `Kernel.timed_call()` and
  `Session`'s methods explicitly synchronize when a host-visible/timed
  result is required.

## Numerical precision

* Default dtype throughout is `float64` (matching `nirnaya-core`'s
  presumed double-precision model). `float32` is supported (pass
  `dtype=np.float32` to `SparseCSR.random`/allocation calls) for
  problems where GPU memory bandwidth or older-GPU FP64 throughput is
  the bottleneck, but this is an explicit opt-in, not a default.
* **CPU and GPU reductions (`sum`, `dot`, `norm2`, `argmin`) are not
  bit-identical**, because they use different summation orders: numpy
  reductions on CPU are effectively sequential/pairwise, while
  cupy/cuBLAS reductions use a parallel tree reduction across thread
  blocks. Floating-point addition is not associative, so a different
  order produces a different (still correct, but not identical)
  rounding pattern. `tests/test_correctness_cpu_vs_gpu.py` therefore
  compares with `rtol=1e-6, atol=1e-8` (configurable), never `==`.
* SpMV itself has the same property: CSR row sums are accumulated in a
  different per-row order between scipy's C loop, the numpy
  `bincount` fallback, and cuSPARSE's block-parallel accumulation.
  `tests/test_cpu_backend.py` uses `np.allclose` throughout, and
  `demo/demo.py` asserts `< 1e-8` absolute error against a dense-numpy
  reference, not exact equality.
* Edge cases explicitly tested (`tests/test_edge_cases.py`): all-zero
  vectors, single-element vectors, `NaN` propagation through
  elementwise ops, sub-`tol` direction components in `ratio_test`
  (must be excluded, not divided-by), values spanning ~22 orders of
  magnitude in one SpMV, and a feasibility check where the violation
  sits at *exactly* the tolerance boundary.

## CPU/GPU tradeoffs

| Factor | CPU backend | GPU backend |
|---|---|---|
| Availability | Always (numpy is a hard dependency) | Only with a working CUDA driver + `cupy` installed; `GPUBackend.available()` performs a real probe every time, never cached as "true" past a failure |
| Best for | Small/medium problems, or any machine without a CUDA GPU, or as the correctness reference | Large, low-density matrices where a single `spmv` call's compute time dominates transfer time |
| Fixed overhead | Near-zero (no context/stream setup) | CUDA context init, stream creation, first-call JIT-like warmup inside cupy/cuSPARSE |
| Transfer cost | N/A (data is already "on device") | Real, measured (`ExecutionContext.transfer_stats`); can dominate for small problems -- see `benchmarks/bench_spmv.py`'s `small` size class |
| Reliability of "it's actually accelerated" | N/A | Enforced structurally: `GPUBackend.__init__` raises `DeviceUnavailableError` rather than degrading to CPU silently; nothing in this package labels a CPU-executed result as "GPU" |

## Honesty contract

This package will never:
* Report a `Device` with `is_gpu=True` that wasn't returned by an
  actual runtime device-count/property query (`gpu.py`'s
  `_probe_cupy` + `getDeviceProperties`).
* Execute an operation on CPU while a caller's code path believes it
  ran on GPU -- `GPUBackend` methods only exist on `GPUBackend`
  instances, which only construct successfully when CUDA is real.
* Print or persist a benchmark number that wasn't measured on the
  run producing the output (`benchmarks/*.py` always call
  `time.perf_counter()` around real kernel invocations; there is no
  lookup table of "expected" numbers anywhere in this package).
