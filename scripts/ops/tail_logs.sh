#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

mkdir -p "$LOG_DIR"
if [[ ! -f "$LOG_FILE" ]]; then
  echo "No log file exists yet."
  echo "Start the server first with: bash scripts/ops/start.sh"
  exit 1
fi

echo "Tailing $LOG_FILE"
exec tail -n 100 -f "$LOG_FILE"
