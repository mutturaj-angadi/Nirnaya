"""
Runs every benchmark in this directory and prints a combined report,
including an explicit host<->device transfer-cost breakdown so the
transfer-vs-compute tradeoff (see docs/ARCHITECTURE.md) is visible in
one place rather than only inside bench_spmv.py.

This produces real numbers for whatever machine it runs on. It writes
nothing pre-computed; if you have not run this script, there is no
benchmark result to read.

Run:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run_script(path: Path):
    print(f"\n{'='*70}\n{path.name}\n{'='*70}")
    subprocess.run([sys.executable, str(path)], check=True)


def end_to_end_solve_style_benchmark():
    """
    A rough proxy for solver-style usage: repeated SpMV + residual +
    feasibility-check calls against the *same* uploaded matrix, timing
    steady-state throughput the way an iterative solver would hit this
    package (upload once, call many times).
    """
    from nirnaya_gpu import Session, SparseCSR
    from nirnaya_gpu.registry import get_backend_by_name

    print(f"\n{'='*70}\nend-to-end iterative-solver-style benchmark\n{'='*70}")
    n = 20_000
    A = SparseCSR.random(n, n, density=0.0005, seed=0)
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=n)
    b = rng.uniform(-1, 1, size=n)

    for backend_name in ["cpu", "cuda"]:
        try:
            session = Session(backend=get_backend_by_name(backend_name))
        except Exception as e:
            print(f"[{backend_name}] skipped: {e}")
            continue

        t_upload0 = time.perf_counter()
        mat_handle = session.upload_matrix(A)
        session.ctx.synchronize()
        t_upload = time.perf_counter() - t_upload0

        xb = session.backend.to_device(session.ctx, x)
        bb = session.backend.to_device(session.ctx, b)

        n_iters = 50
        t0 = time.perf_counter()
        for _ in range(n_iters):
            r = session.backend.residual(session.ctx, mat_handle, xb, bb)
            ok, viol = session.backend.feasibility_check(
                session.ctx, mat_handle, xb, bb, tol=1e-6, sense="le"
            )
        session.ctx.synchronize()
        t_total = time.perf_counter() - t0

        print(
            f"[{backend_name}/{session.device.model_name}] "
            f"n={n:,} nnz={A.nnz:,} matrix_upload={t_upload*1e3:.3f} ms  "
            f"{n_iters} iters of (residual+feasibility) in {t_total*1e3:.3f} ms "
            f"= {t_total/n_iters*1e3:.4f} ms/iter"
        )


def main():
    run_script(HERE / "bench_spmv.py")
    run_script(HERE / "bench_vector_ops.py")
    end_to_end_solve_style_benchmark()


if __name__ == "__main__":
    main()
