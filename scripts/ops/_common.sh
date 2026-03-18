#!/usr/bin/env bash
set -euo pipefail

ASTER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ASTER_ROOT"

CONFIG_PATH="${ASTER_CONFIG_PATH:-${1:-configs/config.yaml}}"
RUN_DIR="$ASTER_ROOT/run"
LOG_DIR="$ASTER_ROOT/logs"
PID_FILE="$RUN_DIR/aster.pid"
LOG_FILE="$LOG_DIR/aster.log"
PYTHON_BIN="$ASTER_ROOT/.venv/bin/python"
CURL_BIN="${CURL_BIN:-curl}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

require_venv() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing virtualenv python: $PYTHON_BIN" >&2
    echo "Create it first: /opt/homebrew/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi
}

require_config() {
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Missing config: $CONFIG_PATH" >&2
    exit 1
  fi
}

get_api_field() {
  local field="$1"
  "$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "$field"
import sys, yaml
config_path, field = sys.argv[1], sys.argv[2]
with open(config_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
api = data.get('api', {})
print(api.get(field, ''))
PY
}

api_host() { get_api_field host; }
api_port() { get_api_field port; }
base_url() { echo "http://$(api_host):$(api_port)"; }

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

current_pid() {
  if [[ -f "$PID_FILE" ]]; then
    cat "$PID_FILE"
  fi
}

is_running() {
  local pid
  pid="$(current_pid || true)"
  [[ -n "${pid:-}" ]] && is_pid_running "$pid"
}

print_runtime_info() {
  echo "root:      $ASTER_ROOT"
  echo "config:    $CONFIG_PATH"
  echo "base_url:  $(base_url)"
  echo "pid_file:  $PID_FILE"
  echo "log_file:  $LOG_FILE"
}
