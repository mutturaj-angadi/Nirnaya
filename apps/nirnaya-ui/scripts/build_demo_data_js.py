#!/usr/bin/env python3
"""Regenerate the UI model bundle from local demo model files.

Archived reference outputs stay out of the browser bundle. They are for
offline validation and are never used to render a production solve result.
"""
import json
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    models = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "demo-data", "*.json"))):
        name = os.path.basename(f)[:-5]
        models[name] = json.load(open(f))

    bundle = {"models": models}
    out_path = os.path.join(ROOT, "src", "js", "demo-data.js")
    with open(out_path, "w") as out:
        out.write("// Auto-generated from demo-data/*.json; contains models only.\n")
        out.write("// Solver results are fetched from the integrated Nirnaya API.\n")
        out.write("// Regenerate with scripts/build_demo_data_js.py\n")
        out.write("window.NIRNAYA_DEMO_DATA = ")
        out.write(json.dumps(bundle, indent=2))
        out.write(";\n")
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
