#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

echo "Tailing $LOG_FILE"
exec tail -n 100 -f "$LOG_FILE"
