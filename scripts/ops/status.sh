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

  # Show vllm-mlx child if it's up
  if [[ "$(model_runtime)" == "vllm_mlx" ]]; then
    vllm_pid="$(lsof -tiTCP:"$(vllm_port)" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    if [[ -n "${vllm_pid:-}" ]]; then
      echo "vllm:     running (pid $vllm_pid, child of aster)"
    else
      echo "vllm:     starting... (waiting for port $(vllm_port))"
    fi
  fi
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

if [[ "$(model_runtime)" == "vllm_mlx" ]]; then
  printf 'vllm:     '
  if "$CURL_BIN" -fsS "$(vllm_health_url)" >/dev/null 2>&1; then
    echo "ok"
  else
    echo "unreachable"
  fi
fi

if [[ -f "$LOG_FILE" ]]; then
  echo ""
  echo "--- log (last 15 lines) ---"
  tail -n 15 "$LOG_FILE"
fi
