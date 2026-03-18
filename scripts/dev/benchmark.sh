#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
CONFIG_PATH="${1:-configs/config.yaml}"
source "$PROJECT_ROOT/.venv/bin/activate"
python "$PROJECT_ROOT/scripts/benchmark_live.py" --config "$CONFIG_PATH" --mode quick
