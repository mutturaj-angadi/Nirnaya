#!/usr/bin/env python3
"""
generate_report.py

Turns one or more recorded benchmark JSON files (produced by
run_benchmarks.py, never hand-edited) into a Markdown report. Every number
in the report is copied verbatim from the recorded run; unavailable metrics
(e.g. GPU not present) are rendered as "unavailable", not zero or omitted
silently.

Usage:
    python3 generate_report.py results/*.json --out results/REPORT.md
"""
import argparse
import glob
import json
from pathlib import Path


def fmt(v, suffix=""):
    if v is None:
        return "unavailable"
    if isinstance(v, float):
        return f"{v:.4g}{suffix}"
    return f"{v}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="benchmarks/results/REPORT.md")
    args = ap.parse_args()

    paths = []
    for pattern in args.files:
        paths.extend(glob.glob(pattern))
    if not paths:
        print("No matching benchmark files found.")
        return

    lines = ["# Nirnaya Benchmark Report", ""]
    lines.append("_Generated from real recorded runs only — no synthetic numbers._\n")

    for p in sorted(paths):
        report = json.loads(Path(p).read_text())
        lines.append(f"## {Path(p).name}")
        lines.append(f"- Generated at: {report.get('generated_at')}")
        lines.append(f"- API base URL: {report.get('api_base_url')}")
        sysinfo = report.get("system_info", {})
        cpu = sysinfo.get("cpu", {})
        gpu = sysinfo.get("gpu", {})
        lines.append(f"- CPU: {fmt(cpu.get('model'))} ({fmt(cpu.get('physical_cores'))} physical / {fmt(cpu.get('logical_cores'))} logical cores)")
        if gpu.get("available"):
            lines.append(f"- GPU: {fmt(gpu.get('device_name'))}")
        else:
            lines.append(f"- GPU: unavailable ({fmt(gpu.get('reason'))})")
        lines.append("")
        lines.append("| Problem | Vars | Cons | NNZ | Backend | Status | Iterations | Solve (ms) | Objective | Max Residual |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in report.get("runs", []):
            lines.append("| " + " | ".join([
                str(r.get("problem")),
                fmt(r.get("num_variables")),
                fmt(r.get("num_constraints")),
                fmt(r.get("nonzeros")),
                str(r.get("backend_requested")),
                str(r.get("status")),
                fmt(r.get("iterations")),
                fmt(r.get("solve_time_ms")),
                fmt(r.get("objective")),
                fmt(r.get("max_residual")),
            ]) + " |")
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
