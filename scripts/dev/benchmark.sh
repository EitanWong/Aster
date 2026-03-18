#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:-configs/config.yaml}"
source .venv/bin/activate
python scripts/benchmark_live.py --config "$CONFIG_PATH" --mode quick
