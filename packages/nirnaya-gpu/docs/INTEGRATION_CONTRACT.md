# Integration Contract: `nirnaya-solver` &harr; `nirnaya-gpu`

This document specifies exactly how `nirnaya-solver` (Part 3) is expected
to call into `nirnaya-gpu` (Part 4), so that:

* `nirnaya-core`'s mathematical model never imports a vendor-specific
  module (`cupy`, `numba.cuda`, etc.).
* `nirnaya-solver` can swap CPU/GPU, or add a future HIP/SYCL backend,
  without changing solver logic.
* Nothing about "GPU acceleration" is ever asserted without a real,
  checkable device behind it.

## 1. Dependency direction

```
nirnaya-core        (pure math model: no dependency on nirnaya-gpu)
      ^
      |  consumed by
      |
nirnaya-solver  ---->  nirnaya-gpu   (this package)
```

`nirnaya-gpu` has **no** dependency on `nirnaya-core` or `nirnaya-solver`.
It only knows about plain `numpy` arrays and the `SparseCSR` wire format
defined in `nirnaya_gpu/sparse.py`. `nirnaya-solver` is responsible for
converting its internal constraint-matrix / basis-matrix representation
into a `SparseCSR` (a one-time, O(nnz) conversion) before calling into
this package.

## 2. What `nirnaya-solver` imports

Solver code should only ever import from the following, stable, public
surface:

```python
from nirnaya_gpu import Session, SparseCSR
from nirnaya_gpu.registry import discover_all, get_backend, get_backend_by_name
from nirnaya_gpu.device import Device, DeviceType
from nirnaya_gpu.exceptions import DeviceUnavailableError, ToleranceExceededError
```

It should **not** import `nirnaya_gpu.backends.cpu` or
`nirnaya_gpu.backends.gpu` directly. Those modules exist so that a
backend can be added or removed without touching solver code; reaching
past `Session`/`registry` reintroduces the coupling this package exists
to remove.

## 3. Two call patterns

### 3.1 `Session` (numpy-in / numpy-out) -- default choice

For most solver call sites (setup, occasional residual/feasibility
checks, anywhere a host round-trip per call is acceptable), use
`Session`:

```python
from nirnaya_gpu import Session, SparseCSR

# Solver constructs A once from its own internal matrix representation.
A_csr = SparseCSR(indptr=..., indices=..., data=..., shape=(m, n))

session = Session()               # picks GPU if genuinely available, else CPU
mat_handle = session.upload_matrix(A_csr)   # upload ONCE, reuse handle

for iteration in solver_loop():
    y = session.spmv(mat_handle, x)         # numpy in, numpy out
    r = session.residual(mat_handle, x, b)
    ok, violation = session.feasibility(mat_handle, x, b, tol=1e-6, sense="le")
    idx, reduced_cost = session.argmin(reduced_costs)
    leave_idx, ratio = session.ratio_test(basic_values, direction)
```

Each `Session` method:
1. Moves any plain-numpy arguments to the active device,
2. Runs the kernel,
3. Synchronizes,
4. Copies the result back to a numpy array.

This is simple and correct but pays a host&harr;device transfer on every
call. It is the right choice unless profiling shows it is a bottleneck.

### 3.2 Raw `Backend` + `DeviceBuffer` -- hot loops

For a tight iterative loop (e.g. a simplex or interior-point inner
loop) where `x` stays resident on the device across many iterations,
bypass `Session`'s per-call transfers and drive the backend directly:

```python
from nirnaya_gpu.registry import get_backend

backend = get_backend(prefer_gpu=True)
device = backend.discover_devices()[0]
ctx = backend.create_context(device)

mat_handle = backend.upload_sparse(ctx, A_csr)
x_buf = backend.to_device(ctx, x0)
b_buf = backend.to_device(ctx, b)

for iteration in solver_loop():
    y_buf = backend.spmv(ctx, mat_handle, x_buf)
    # ... more kernel calls, all operating on DeviceBuffers ...
    # only synchronize / copy to host when the solver actually needs
    # a host-visible value (e.g. a convergence check every K iterations)

ctx.synchronize()
x_final = backend.to_host(ctx, x_buf)
```

`ExecutionContext` is also a context manager (`with backend.create_context(device) as ctx:`)
that synchronizes on exit, for solver code that wants scoped cleanup.

## 4. Backend selection & honest fallback

* `get_backend(prefer_gpu=True)` (used internally by `Session()`)
  attempts to construct a CUDA backend; if CUDA is not genuinely
  available it falls back to `CPUBackend` **and does not hide this** --
  `backend.name` will read `"cpu"`, and `Session.is_gpu` will be
  `False`.
* If the solver needs to *require* a specific backend (e.g. a
  benchmark script that must not silently fall back), use
  `get_backend_by_name("cuda")`, which raises `DeviceUnavailableError`
  instead of substituting CPU.
* `discover_all()` returns every device nirnaya-gpu can currently see
  across all backends, plus the CUDA probe's error message if CUDA is
  unavailable -- useful for logging/diagnostics at solver startup.

Solver code that wants to log what it's actually running on should do:

```python
session = Session()
logger.info("nirnaya-gpu backend=%s device=%s", session.backend.name, session.device.describe())
```

## 5. Numerical tolerance contract

GPU reductions (`dot`, `reduce(op="sum"/"norm2"/...)`) use a different
summation order than the CPU path (see `docs/ARCHITECTURE.md`,
"Numerical precision"), so solver code that compares a GPU-computed
value against a CPU-computed reference (e.g. in tests, or in a
correctness-checking mode) must use a tolerance, not exact equality.
This package's own test suite uses `rtol=1e-6, atol=1e-8` for
double-precision comparisons (`tests/test_correctness_cpu_vs_gpu.py`);
solver code should use a comparable tolerance, exposed as a parameter
rather than hard-coded, and should raise/handle
`nirnaya_gpu.exceptions.ToleranceExceededError` (or its own equivalent)
rather than silently accepting an arbitrarily large discrepancy.

## 6. Kernels available to the solver

| Operation | `Session` method | `Backend` method | Typical solver use |
|---|---|---|---|
| Sparse matvec | `spmv(mat, x)` | `spmv(ctx, mat_handle, x)` | `A @ x`, forming `Ax` |
| Residual | `residual(mat, x, b)` | `residual(ctx, mat_handle, x, b)` | `b - Ax` |
| Feasibility check | `feasibility(mat, x, b, tol, sense)` | `feasibility_check(...)` | constraint satisfaction, parallel across rows |
| Dot product | `dot(x, y)` | `dot(ctx, x, y)` | convergence norms, line search |
| AXPY | `axpy(alpha, x, y)` | `axpy(ctx, alpha, x, y)` | update steps `y += alpha*x` |
| Scale | `scale(alpha, x)` | `scale(ctx, alpha, x)` | step scaling |
| Elementwise | `elementwise(op, x, y)` | `elementwise(ctx, op, x, y)` | bound clipping, componentwise max/min |
| Reduction | `reduce(x, op)` | `reduce(ctx, x, op)` | norms, sums |
| Argmin (pivot selection) | `argmin(x)` | `argmin(ctx, x)` | entering-variable selection (most negative reduced cost) |
| Ratio test (leaving variable) | `ratio_test(x, direction, tol)` | `ratio_test(ctx, x, direction, tol)` | minimum-ratio test for basis changes |

If the solver needs an operation not in this table, it should be added
to `Backend` (both `CPUBackend` and `GPUBackend`) rather than having
solver code fall back to raw numpy/cupy -- that is exactly the coupling
this package exists to prevent.

## 7. What this package explicitly does NOT do

* It does not decide *when* to run on GPU vs CPU for solver-level
  algorithmic reasons (e.g. "problem too small, not worth a GPU
  launch") -- that policy belongs in `nirnaya-solver`, which can
  inspect `session.is_gpu`, problem size, and `discover_all()` to make
  that call.
* It does not implement basis factorization/LU updates -- only the
  primitive kernels (`spmv`, `argmin`, `ratio_test`, reductions) that
  such an algorithm is built from on top, in `nirnaya-solver`.
* It never reports a GPU result when the GPU backend was unavailable;
  see `docs/ARCHITECTURE.md`, "Honesty contract".
