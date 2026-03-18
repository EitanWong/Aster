#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

require_venv
require_config

print_runtime_info

if is_running; then
  echo "process:   running (pid $(current_pid))"
else
  echo "process:   not running"
fi

url="$(base_url)"
printf 'health:    '
if "$CURL_BIN" -fsS "$url/health"; then
  echo
else
  echo "unreachable"
fi

printf 'ready:     '
if "$CURL_BIN" -fsS "$url/ready"; then
  echo
else
  echo "unreachable"
fi

printf 'models:    '
if "$CURL_BIN" -fsS "$url/v1/models" >/dev/null; then
  echo "ok"
else
  echo "unreachable"
fi

if [[ -f "$LOG_FILE" ]]; then
  echo "log_tail:"
  tail -n 20 "$LOG_FILE"
fi
