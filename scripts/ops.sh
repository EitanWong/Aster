#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-}"
shift || true

case "$cmd" in
  start) bash "$(dirname "$0")/start.sh" "$@" ;;
  stop) bash "$(dirname "$0")/stop.sh" "$@" ;;
  restart) bash "$(dirname "$0")/restart.sh" "$@" ;;
  status) bash "$(dirname "$0")/status.sh" "$@" ;;
  health) bash "$(dirname "$0")/health.sh" "$@" ;;
  smoke) bash "$(dirname "$0")/smoke_test.sh" "$@" ;;
  logs) bash "$(dirname "$0")/tail_logs.sh" "$@" ;;
  metrics) bash "$(dirname "$0")/metrics_summary.sh" "$@" ;;
  *)
    cat <<'EOF'
Usage: bash scripts/ops.sh <command> [config]

Commands:
  start
  stop
  restart
  status
  health
  smoke
  logs
EOF
    exit 1
    ;;
esac
