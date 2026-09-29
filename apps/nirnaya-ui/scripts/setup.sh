#!/usr/bin/env bash
# Install the local Nirnaya packages into the active Python environment.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 -m pip install -e \
  "$ROOT_DIR/packages/nirnaya-core" \
  "$ROOT_DIR/packages/nirnaya-presolve" \
  "$ROOT_DIR/packages/nirnaya-solver" \
  "$ROOT_DIR/packages/nirnaya-gpu" \
  "$ROOT_DIR/packages/nirnaya-api"
echo "Installed local packages. Start the API and dashboard with scripts/launch_demo.py."
