"""
Benchmark: dense vector kernels (dot, axpy, scale, reduce) across vector
lengths. Real timed measurements only -- see bench_spmv.py's docstring
for the same honesty contract.

Run:
    python benchmarks/bench_vector_ops.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from nirnaya_gpu import Session
from nirnaya_gpu.registry import get_backend_by_name

LENGTHS = [1_000, 100_000, 5_000_000]


def _time(fn, n_repeats: int):
    fn()  # warmup
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


def bench_session(session: Session, lengths, n_repeats: int):
    label = f"{session.backend.name}/{session.device.model_name}"
    for n in lengths:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, size=n)
        y = rng.uniform(-1, 1, size=n)
        xb = session.backend.to_device(session.ctx, x)
        yb = session.backend.to_device(session.ctx, y)

        def do_dot():
            session.backend.dot(session.ctx, xb, yb)

        def do_axpy():
            session.backend.axpy(session.ctx, 1.0001, xb, yb)
            session.ctx.synchronize()

        def do_scale():
            session.backend.scale(session.ctx, 0.9999, xb)
            session.ctx.synchronize()

        def do_reduce():
            session.backend.reduce(session.ctx, xb, "norm2")

        for op_name, fn in [
            ("dot", do_dot),
            ("axpy", do_axpy),
            ("scale", do_scale),
            ("reduce(norm2)", do_reduce),
        ]:
            mean_s, std_s = _time(fn, n_repeats)
            gbps = (n * x.itemsize * 2 / 1e9) / mean_s if mean_s > 0 else float("inf")
            print(
                f"[{label}] n={n:>9,} {op_name:14s} "
                f"mean={mean_s*1e6:9.2f} us (+/- {std_s*1e6:.2f} us)  ~{gbps:.2f} GB/s"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    cpu = Session(backend=get_backend_by_name("cpu"))
    bench_session(cpu, LENGTHS, args.repeats)

    try:
        gpu = Session(backend=get_backend_by_name("cuda"))
    except Exception as e:
        print(f"\nGPU vector-op benchmarks skipped: {e}")
        return
    bench_session(gpu, LENGTHS, args.repeats)


if __name__ == "__main__":
    main()
