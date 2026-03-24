#!/usr/bin/env bash
# Start Aster.
#
# When model.runtime is vllm_mlx, Aster spawns and manages the vllm-mlx
# sidecar internally — no separate process to start or track.
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

require_venv
require_config

FOREGROUND="${ASTER_FOREGROUND:-0}"
if [[ "${2:-}" == "--foreground" || "${1:-}" == "--foreground" ]]; then
  FOREGROUND=1
fi

sync_pid_file

if is_running; then
  pid="$(current_pid)"
  echo "aster is already running  pid=$pid  url=$(base_url)"
  exit 0
fi

if port_in_use_by_other_process; then
  echo "error: port $(api_port) is in use by another process" >&2
  lsof -nP -iTCP:"$(api_port)" -sTCP:LISTEN >&2 || true
  exit 1
fi

rm -f "$PID_FILE"

echo "starting aster@$(api_port)  config=$(basename "$CONFIG_PATH_ABS")  runtime=$(model_runtime)"

if [[ "$FOREGROUND" == "1" ]]; then
  exec "$PYTHON_BIN" server.py --config "$CONFIG_PATH_ABS"
fi

nohup "$PYTHON_BIN" server.py --config "$CONFIG_PATH_ABS" >>"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

# Wait up to 5 minutes — vllm-mlx model loading can be slow on first start.
echo -n "waiting for /health"
for _ in {1..300}; do
  sync_pid_file
  if ! is_pid_running "$pid"; then
    echo ""
    echo "error: aster exited unexpectedly — check $LOG_FILE" >&2
    exit 1
  fi
  if "$CURL_BIN" -fsS "$(base_url)/health" >/dev/null 2>&1; then
    echo ""
    echo "started  pid=$pid  url=$(base_url)"
    exit 0
  fi
  echo -n "."
  sleep 1
done

echo ""
echo "error: /health did not respond within 300 s — check $LOG_FILE" >&2
exit 1
