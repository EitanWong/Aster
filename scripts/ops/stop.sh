#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

if ! [[ -f "$PID_FILE" ]]; then
  echo "Aster is not running (no pid file)."
  exit 0
fi

pid="$(current_pid || true)"
if [[ -z "${pid:-}" ]]; then
  echo "Aster pid file is empty; cleaning up."
  rm -f "$PID_FILE"
  exit 0
fi

if ! is_pid_running "$pid"; then
  echo "Aster is not running (stale pid $pid). Cleaning up pid file."
  rm -f "$PID_FILE"
  exit 0
fi

echo "Stopping Aster (pid $pid)..."
kill "$pid"

for _ in {1..20}; do
  if ! is_pid_running "$pid"; then
    rm -f "$PID_FILE"
    echo "Aster stopped."
    exit 0
  fi
  sleep 0.5
done

echo "Graceful stop timed out; sending SIGKILL."
kill -9 "$pid" >/dev/null 2>&1 || true
rm -f "$PID_FILE"
echo "Aster stopped forcefully."
