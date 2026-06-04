#!/usr/bin/env bash
# Stop Aster.
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

pid="$(current_pid || true)"

if [[ -z "${pid:-}" ]]; then
  echo "aster is not running"
  rm -f "$PID_FILE"
  exit 0
fi

echo "stopping aster  pid=$pid"
kill "$pid"

for _ in {1..30}; do
  if ! is_pid_running "$pid"; then
    rm -f "$PID_FILE"
    echo "aster stopped"
    exit 0
  fi
  sleep 0.5
done

echo "graceful stop timed out — sending SIGKILL to aster"
kill -9 "$pid" >/dev/null 2>&1 || true
rm -f "$PID_FILE"
echo "aster stopped (forced)"
