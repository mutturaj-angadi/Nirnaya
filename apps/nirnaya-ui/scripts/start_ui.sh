#!/usr/bin/env bash
# Compatibility wrapper for the canonical production API + dashboard launcher.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec python3 "$ROOT_DIR/scripts/launch_demo.py" "$@"
