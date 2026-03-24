#!/usr/bin/env bash
# Stop Aster and ensure its vllm-mlx sidecar is also gone.
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

# ---------------------------------------------------------------------------
# Helper: kill any process still listening on the vllm port
# ---------------------------------------------------------------------------
stop_vllm_if_running() {
  [[ "$(model_runtime)" == "vllm_mlx" ]] || return 0

  local port pid pgid
  port="$(vllm_port)"
  pid="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  [[ -n "${pid:-}" ]] || return 0

  echo "stopping vllm-mlx  pid=$pid  port=$port"
  # Kill the whole process group (set by start_new_session=True in Python).
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "${pgid:-}" && "$pgid" != "0" ]]; then
    kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  else
    kill "$pid" 2>/dev/null || true
  fi

  # Wait up to 10 s for the port to be released.
  for _ in {1..20}; do
    sleep 0.5
    lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || { echo "vllm-mlx stopped"; return 0; }
  done

  # Escalate to SIGKILL.
  pid="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if [[ -n "${pid:-}" ]]; then
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${pgid:-}" && "$pgid" != "0" ]]; then
      kill -9 -- "-$pgid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    else
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "vllm-mlx stopped (forced)"
  fi
}

# ---------------------------------------------------------------------------
# Stop Aster
# ---------------------------------------------------------------------------
pid="$(current_pid || true)"

if [[ -z "${pid:-}" ]]; then
  echo "aster is not running"
  rm -f "$PID_FILE"
  # Aster may have crashed or been killed externally — sidecar could be orphaned.
  stop_vllm_if_running
  exit 0
fi

echo "stopping aster  pid=$pid"
kill "$pid"

for _ in {1..30}; do
  if ! is_pid_running "$pid"; then
    rm -f "$PID_FILE"
    echo "aster stopped"
    # Give Aster's lifespan stop() a moment to send SIGTERM to the sidecar,
    # then verify the sidecar is gone (kill if still up).
    sleep 1
    stop_vllm_if_running
    exit 0
  fi
  sleep 0.5
done

echo "graceful stop timed out — sending SIGKILL to aster"
kill -9 "$pid" >/dev/null 2>&1 || true
rm -f "$PID_FILE"
echo "aster stopped (forced)"
stop_vllm_if_running
