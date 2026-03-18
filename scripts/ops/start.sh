#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

require_venv
require_config

FOREGROUND="${ASTER_FOREGROUND:-0}"
if [[ "${2:-}" == "--foreground" || "${1:-}" == "--foreground" ]]; then
  FOREGROUND=1
fi

if is_running; then
  echo "Aster is already running (pid $(current_pid))."
  print_runtime_info
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  rm -f "$PID_FILE"
fi

echo "Starting Aster..."
print_runtime_info

if [[ "$FOREGROUND" == "1" ]]; then
  exec "$PYTHON_BIN" server.py --config "$CONFIG_PATH"
fi

nohup "$PYTHON_BIN" server.py --config "$CONFIG_PATH" >>"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 1

if is_pid_running "$pid"; then
  echo "Aster started in background (pid $pid)."
  echo "Use: bash scripts/status.sh"
else
  echo "Aster failed to stay up. Check logs: $LOG_FILE" >&2
  exit 1
fi
