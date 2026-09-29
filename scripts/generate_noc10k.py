#!/usr/bin/env python3
"""Generate the deterministic NOC-10K compact metadata corpus."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in (
    "packages/nirnaya-core/src", "packages/nirnaya-presolve/src",
    "packages/nirnaya-gpu/src", "packages/nirnaya-solver/src",
    "packages/nirnaya-api/src")]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nirnaya_api.orchestrator import parse_and_validate_model  # noqa: E402
from noc10k_common import (  # noqa: E402
    DEFAULT_SEED, FAMILY_COUNTS, GENERATOR_VERSION, iter_specs, materialize_instance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--family", choices=sorted(FAMILY_COUNTS))
    parser.add_argument("--count", type=int)
    parser.add_argument("--start", type=int, default=0, help="zero-based inclusive corpus offset")
    parser.add_argument("--end", type=int, help="zero-based exclusive corpus offset")
    parser.add_argument("--workers", type=int, default=1,
                        help="accepted for command consistency; generation is serial and deterministic")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/noc10k/manifest.jsonl.gz")
    args = parser.parse_args()
    if args.workers < 1 or args.start < 0 or args.count is not None and args.count < 0:
        parser.error("workers/count must be positive/nonnegative and start must be nonnegative")
    specs = list(iter_specs(args.seed, args.family, args.count))
    end = len(specs) if args.end is None else args.end
    if end < args.start:
        parser.error("--end must be greater than or equal to --start")
    selected = specs[args.start:end]
    if len(selected) != max(0, min(end, len(specs)) - args.start):
        parser.error("invalid range")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as zipped:
            for family, family_index, instance_seed, family_count in selected:
                model, record = materialize_instance(family, instance_seed, family_index,
                                                      family_count, args.seed)
                parse_and_validate_model(model)
                zipped.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode("utf-8") + b"\n")
    print(json.dumps({"generator_version": GENERATOR_VERSION,
        "instances_written": len(selected), "corpus_count": len(specs),
        "output": str(args.output), "generation_mode": "serial",
        "workers_argument": args.workers},
        sort_keys=True))


if __name__ == "__main__":
    main()
