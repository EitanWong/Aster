#!/usr/bin/env bash
# Common utilities sourced by ops scripts.
# Usage: source _common.sh [config_path]
set -euo pipefail

ASTER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ASTER_ROOT"

CONFIG_PATH="${ASTER_CONFIG_PATH:-${1:-configs/config.yaml}}"
CONFIG_PATH_ABS="$(cd "$(dirname "$CONFIG_PATH")" && pwd -P)/$(basename "$CONFIG_PATH")"
RUN_DIR="$ASTER_ROOT/run"
LOG_DIR="$ASTER_ROOT/logs"
PID_FILE="$RUN_DIR/aster.pid"
LOG_FILE="$LOG_DIR/aster.log"
VLLM_PID_FILE="$RUN_DIR/vllm-mlx.pid"   # informational only (written by Aster)
PYTHON_BIN="$ASTER_ROOT/.venv/bin/python"
CURL_BIN="${CURL_BIN:-curl}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Config reader
# ---------------------------------------------------------------------------

get_config_value() {
  local path="$1"
  local default_value="${2:-}"
  "$PYTHON_BIN" - <<'PY' "$CONFIG_PATH" "$path" "$default_value"
import sys, yaml

config_path, dotted_path, default_value = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}

value = data
for part in dotted_path.split('.'):
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break

if value is None:
    value = default_value

if isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)
PY
}

# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------

api_host()    { get_config_value api.host 127.0.0.1; }
api_port()    { get_config_value api.port 8080; }
base_url()    { echo "http://$(api_host):$(api_port)"; }

model_runtime() { get_config_value model.runtime mlx; }
model_name()    { get_config_value model.name ""; }
model_path()    { get_config_value model.path ""; }

vllm_base_url()  { get_config_value vllm_mlx.base_url http://127.0.0.1:8000; }
vllm_health_url() { echo "$(vllm_base_url | sed 's:/*$::')/health"; }

parse_url_part() {
  local url="$1" part="$2"
  "$PYTHON_BIN" - <<'PY' "$url" "$part"
import sys
from urllib.parse import urlparse
url, part = sys.argv[1], sys.argv[2]
parsed = urlparse(url)
if part == "host":
    print(parsed.hostname or "")
elif part == "port":
    print(parsed.port or (443 if parsed.scheme == "https" else 80))
PY
}

vllm_host() { parse_url_part "$(vllm_base_url)" host; }
vllm_port() { parse_url_part "$(vllm_base_url)" port; }

# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

is_pid_running() {
  kill -0 "$1" >/dev/null 2>&1
}

process_command() {
  ps -p "$1" -o command= 2>/dev/null | sed -e 's/^[[:space:]]*//' || true
}

process_title() {
  process_command "$1"
}

is_managed_process() {
  local cmd
  cmd="$(process_command "$1")"
  [[ -n "$cmd" ]] || return 1
  # New compact title
  [[ "$cmd" == "aster@"* ]] && return 0
  # Fallback: match by config path + entry-point
  [[ "$cmd" == *"$CONFIG_PATH_ABS"* ]] || return 1
  [[ "$cmd" == *"server.py"* || "$cmd" == *"-m aster"* ]] || return 1
}

discover_pids_by_port() {
  lsof -tiTCP:"$(api_port)" -sTCP:LISTEN 2>/dev/null || true
}

discover_managed_pids() {
  local pid
  for pid in $(discover_pids_by_port); do
    if is_managed_process "$pid"; then
      echo "$pid"
    fi
  done
}

current_pid() {
  local pid
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if [[ -n "${pid:-}" ]] && is_pid_running "$pid" && is_managed_process "$pid"; then
      echo "$pid"
      return 0
    fi
  fi
  pid="$(discover_managed_pids | head -n 1)"
  if [[ -n "${pid:-}" ]]; then
    echo "$pid"
  fi
}

is_running() {
  local pid
  pid="$(current_pid || true)"
  [[ -n "${pid:-}" ]] && is_pid_running "$pid"
}

sync_pid_file() {
  local pid
  pid="$(current_pid || true)"
  if [[ -n "${pid:-}" ]]; then
    echo "$pid" > "$PID_FILE"
  else
    rm -f "$PID_FILE"
  fi
}

port_in_use() {
  [[ -n "$(discover_pids_by_port | head -n 1)" ]]
}

port_in_use_by_other_process() {
  local pid
  for pid in $(discover_pids_by_port); do
    if ! is_managed_process "$pid"; then
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

print_runtime_info() {
  echo "root:     $ASTER_ROOT"
  echo "config:   $CONFIG_PATH_ABS"
  echo "url:      $(base_url)"
  echo "runtime:  $(model_runtime)"
  echo "pid_file: $PID_FILE"
  echo "log_file: $LOG_FILE"
  if [[ "$(model_runtime)" == "vllm_mlx" ]]; then
    echo "vllm_url: $(vllm_base_url)"
  fi
}
