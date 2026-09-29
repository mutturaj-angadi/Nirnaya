"""
Benchmark: sparse matrix-vector multiplication, CPU vs GPU, across
small/medium/large problem sizes.

This script performs REAL timed executions on whatever machine it is
run on -- it does not print pre-recorded numbers. If CUDA is not
available, the GPU columns are reported as "n/a (cuda unavailable)"
rather than a fabricated or estimated value.

Run:
    python benchmarks/bench_spmv.py
    python benchmarks/bench_spmv.py --sizes small medium
    python benchmarks/bench_spmv.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

from nirnaya_gpu import Session, SparseCSR
from nirnaya_gpu.registry import discover_all, get_backend_by_name

SIZE_PRESETS = {
    # name: (n_rows, n_cols, density)
    "small": (500, 500, 0.05),
    "medium": (5_000, 5_000, 0.01),
    "large": (50_000, 50_000, 0.001),
}


@dataclass
class BenchResult:
    size_name: str
    n_rows: int
    n_cols: int
    density: float
    nnz: int
    backend: str
    device: str
    n_repeats: int
    transfer_seconds_mean: Optional[float]
    kernel_seconds_mean: float
    kernel_seconds_stdev: float
    gflops: float  # 2*nnz FLOPs per SpMV


def _time_repeats(fn, n_repeats: int) -> List[float]:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def bench_one(session: Session, size_name: str, n_repeats: int) -> BenchResult:
    n_rows, n_cols, density = SIZE_PRESETS[size_name]
    A = SparseCSR.random(n_rows, n_cols, density=density, seed=0)
    x = np.random.default_rng(1).uniform(-1, 1, size=n_cols)

    # Time the host->device transfer of the matrix separately from
    # steady-state kernel execution (see docs/ARCHITECTURE.md).
    t0 = time.perf_counter()
    mat_handle = session.upload_matrix(A)
    session.ctx.synchronize()
    transfer_s = time.perf_counter() - t0 if session.is_gpu else None

    xb = session.backend.to_device(session.ctx, x)

    def run_once():
        yb = session.backend.spmv(session.ctx, mat_handle, xb)
        session.ctx.synchronize()
        return yb

    # warmup (JIT / first-touch / cache effects should not count)
    run_once()
    times = _time_repeats(run_once, n_repeats)

    flops = 2 * A.nnz  # one multiply + one add per nonzero
    mean_t = statistics.mean(times)
    gflops = (flops / 1e9) / mean_t if mean_t > 0 else float("inf")

    return BenchResult(
        size_name=size_name,
        n_rows=n_rows,
        n_cols=n_cols,
        density=density,
        nnz=A.nnz,
        backend=session.backend.name,
        device=session.device.model_name,
        n_repeats=n_repeats,
        transfer_seconds_mean=transfer_s,
        kernel_seconds_mean=mean_t,
        kernel_seconds_stdev=statistics.stdev(times) if len(times) > 1 else 0.0,
        gflops=gflops,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", default=list(SIZE_PRESETS.keys()), choices=SIZE_PRESETS.keys()
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json", type=str, default=None, help="write results as JSON here")
    args = parser.parse_args()

    print(f"Platform: {platform.platform()}")
    report = discover_all()
    print(report.summary())
    print()

    results: List[BenchResult] = []

    cpu_session = Session(backend=get_backend_by_name("cpu"))
    for size in args.sizes:
        r = bench_one(cpu_session, size, args.repeats)
        results.append(r)
        print(
            f"[CPU/{r.device}] {size:7s} nnz={r.nnz:>9,} "
            f"mean={r.kernel_seconds_mean*1e3:8.3f} ms "
            f"(+/- {r.kernel_seconds_stdev*1e3:.3f} ms)  {r.gflops:.3f} GFLOP/s"
        )

    try:
        gpu_session = Session(backend=get_backend_by_name("cuda"))
    except Exception as e:
        print(f"\nGPU benchmarks skipped: {e}")
        gpu_session = None

    if gpu_session is not None:
        for size in args.sizes:
            r = bench_one(gpu_session, size, args.repeats)
            results.append(r)
            transfer_ms = (
                f"{r.transfer_seconds_mean*1e3:.3f} ms" if r.transfer_seconds_mean else "n/a"
            )
            print(
                f"[GPU/{r.device}] {size:7s} nnz={r.nnz:>9,} "
                f"mean={r.kernel_seconds_mean*1e3:8.3f} ms "
                f"(+/- {r.kernel_seconds_stdev*1e3:.3f} ms)  {r.gflops:.3f} GFLOP/s "
                f"  transfer={transfer_ms}"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nWrote {len(results)} results to {args.json}")


if __name__ == "__main__":
    main()
