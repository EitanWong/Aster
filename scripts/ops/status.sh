#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

require_venv
require_config

sync_pid_file
print_runtime_info

if is_running; then
  pid="$(current_pid)"
  echo "process:  running (pid $pid)"
  echo "title:    $(process_title "$pid")"
else
  echo "process:  not running"
  if port_in_use; then
    echo "warning:  port $(api_port) is in use but not by a managed Aster instance"
  fi
fi

echo ""
url="$(base_url)"
printf 'health:   '
if "$CURL_BIN" -fsS "$url/health" 2>/dev/null; then
  echo ""
else
  echo "unreachable"
fi

printf 'ready:    '
if "$CURL_BIN" -fsS "$url/ready" 2>/dev/null; then
  echo ""
else
  echo "unreachable"
fi

status_json="$("$CURL_BIN" -fsS "$url/v1/status" 2>/dev/null || true)"
if [[ -n "$status_json" ]]; then
  "$PYTHON_BIN" - <<'PY' "$status_json"
import json
import sys

try:
    data = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print(f"status:   {sys.argv[1]}")
    raise SystemExit(0)

print(
    "status:   "
    f"{data.get('status', 'unknown')} "
    f"model={data.get('model', 'unknown')} "
    f"running={data.get('num_running', 'n/a')} "
    f"waiting={data.get('num_waiting', 'n/a')}"
)
responses_store = data.get("responses_store")
if isinstance(responses_store, dict):
    print(
        "responses_store: "
        f"entries={responses_store.get('entries', 'n/a')} "
        f"max_entries={responses_store.get('max_entries', 'n/a')} "
        f"scope={responses_store.get('scope', 'n/a')}"
    )
PY
else
  echo "status:   unreachable"
fi

if [[ -f "$LOG_FILE" ]]; then
  echo ""
  echo "--- log (last 15 lines) ---"
  tail -n 15 "$LOG_FILE"
fi
