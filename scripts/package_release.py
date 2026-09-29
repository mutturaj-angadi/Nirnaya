#!/usr/bin/env python3
"""Create a curated portable Nirnaya ZIP without workspace/cache baggage."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "Nirnaya_SIH2026_Final_Release.zip"
OUTPUT = ROOT / "dist" / ARCHIVE_NAME
TOP_LEVEL_FILES = (
    ".gitignore", "README.md", "requirements.txt", "requirements-dev.txt",
    "pyproject.toml", "RELEASE_MANIFEST.txt",
)
TOP_LEVEL_DIRS = (
    "apps", "benchmarks", "datasets", "docs", "examples", "packages",
    "scripts", "tests",
)
EXCLUDED_DIRS = {
    ".git", ".codex", ".agents", ".idea", ".vscode", "__pycache__",
    ".pytest_cache", ".pytest-tmp", ".mypy_cache", ".ruff_cache", "node_modules",
    ".venv", "venv", "build", "dist", "target",
}
BENCHMARK_EVIDENCE = {
    "benchmarks/results/reproducible/latest.json",
    "benchmarks/results/synthetic/latest.json",
    "benchmarks/results/refinery-scenarios/latest.json",
    "benchmarks/results/release-gate/latest.json",
    "benchmarks/results/release-gate/runs/concurrent-contention-reproduction.json",
    "benchmarks/results/noc10k/run_eta_final.jsonl",
    "benchmarks/results/noc10k/run.jsonl",
    "benchmarks/results/noc10k/dev_eta_sample.jsonl",
    "benchmarks/results/noc10k/summary.json",
    "benchmarks/results/noc10k/summary.csv",
    "benchmarks/results/noc10k/family_summary.csv",
    "benchmarks/results/noc10k/failures.csv",
    "benchmarks/results/noc10k/holdout_summary.json",
    "benchmarks/results/noc10k/machine_info.json",
    "benchmarks/results/noc10k/runs/run_20260928T131538377320Z.jsonl",
}


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return True
    name = path.name.lower()
    if name.endswith((".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp")):
        return True
    if rel.parts[0] == "source-archives":
        return True
    if rel.parts[:2] == ("benchmarks", "results"):
        return rel.as_posix() not in BENCHMARK_EVIDENCE
    # Generated reports are recreated by validation; don't ship host-local run output.
    if rel.as_posix() in {
        "apps/nirnaya-ui/validation/validation_report.json",
        "apps/nirnaya-ui/validation/validation_run_metadata.json",
    }:
        return True
    return False


def collect_files() -> list[Path]:
    files = []
    for name in TOP_LEVEL_FILES:
        path = ROOT / name
        if path.is_file() and not is_excluded(path):
            files.append(path)
    for directory in TOP_LEVEL_DIRS:
        base = ROOT / directory
        for path in base.rglob("*"):
            if path.is_file() and not is_excluded(path):
                files.append(path)
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix().casefold())


def create_manifest(paths: list[Path]) -> bytes:
    rel_paths = [p.relative_to(ROOT).as_posix() for p in paths]
    rel_paths.extend(("RELEASE_MANIFEST.json", "RELEASE_MANIFEST.txt"))
    payload = {
        "project": "Nirnaya — SIH 2026",
        "release_version": "1.0.0-sih2026",
        "release_date": "2026-09-28",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_name": ARCHIVE_NAME,
        "archive_layout": "project files at ZIP root; no extra directory level",
        "canonical_install_command": "python -m pip install -r requirements.txt",
        "canonical_launch_command": "python scripts\\launch_demo.py",
        "canonical_test_command": "python -m pytest -q",
        "production_solver": "repository-authored deterministic two-phase revised simplex on CPU with sparse LU and bounded product-form eta basis updates",
        "reference_solver_role": "SciPy/HiGHS is validation-only; never a production fallback",
        "gpu": {
            "optional": True,
            "claim": "GPU-ready capability and primitive layer; production LP simplex executes on CPU",
        },
        "noc10k": {
            "instances": 10000,
            "split_counts": {"development": 8000, "validation": 1000, "holdout": 1000},
            "purpose": "optimization validation corpus; not ML training",
        },
        "included_file_count": len(set(rel_paths)),
        "included_files": sorted(set(rel_paths)),
        "excluded": ["Git/Codex/IDE state", "caches and bytecode", "source archive bundles", "duplicate historical benchmark runs", "temporary and host-generated report files"],
        "known_limits": [
            "continuous LPs only",
            "no API admission control for concurrent heavy solves",
            "standalone legacy presolve tests target removed core APIs",
            "no service-level runtime guarantee under CPU contention",
        ],
        "package_method": "scripts/package_release.py; curated allowlist; ZIP_DEFLATED; CRC tested before handoff",
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    paths = collect_files()
    required = {
        "README.md", "requirements.txt", "RELEASE_MANIFEST.txt",
        "scripts/launch_demo.py", "scripts/generate_noc10k.py",
        "scripts/run_noc10k.py", "scripts/analyze_noc10k.py",
        "datasets/noc10k/manifest.jsonl.gz",
        "benchmarks/results/noc10k/run_eta_final.jsonl",
        "benchmarks/results/noc10k/runs/run_20260928T131538377320Z.jsonl",
        "apps/nirnaya-ui/src/index.html",
        "packages/nirnaya-solver/src/nirnaya_solver/solver.py",
    }
    included = {p.relative_to(ROOT).as_posix() for p in paths}
    missing = sorted(required - included)
    if missing:
        raise SystemExit(f"release inputs missing: {missing}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing release archive: {OUTPUT}")

    manifest = create_manifest(paths)
    with zipfile.ZipFile(OUTPUT, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("RELEASE_MANIFEST.json", manifest)
    with zipfile.ZipFile(OUTPUT, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            OUTPUT.unlink(missing_ok=True)
            raise SystemExit(f"ZIP CRC validation failed: {bad_member}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            OUTPUT.unlink(missing_ok=True)
            raise SystemExit("ZIP contains duplicate paths")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    digest_path = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {OUTPUT.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
        "sha256": digest, "files": len(paths) + 1,
        "manifest": "RELEASE_MANIFEST.json", "crc_ok": True}, sort_keys=True))


if __name__ == "__main__":
    main()
