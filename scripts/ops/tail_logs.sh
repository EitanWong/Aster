#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

mkdir -p "$LOG_DIR"
if [[ ! -f "$LOG_FILE" ]]; then
  echo "Log file does not exist yet: $LOG_FILE"
  echo "Start the server first with: bash scripts/ops.sh start"
  exit 1
fi

echo "Tailing $LOG_FILE"
exec tail -n 100 -f "$LOG_FILE"
