#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPS_DIR="$SCRIPT_DIR/ops"

cmd="${1:-}"
shift || true

case "$cmd" in
  start) bash "$OPS_DIR/start.sh" "$@" ;;
  stop) bash "$OPS_DIR/stop.sh" "$@" ;;
  restart) bash "$OPS_DIR/restart.sh" "$@" ;;
  status) bash "$OPS_DIR/status.sh" "$@" ;;
  health) bash "$OPS_DIR/health.sh" "$@" ;;
  smoke) bash "$OPS_DIR/smoke_test.sh" "$@" ;;
  logs) bash "$OPS_DIR/tail_logs.sh" "$@" ;;
  metrics) bash "$SCRIPT_DIR/dev/metrics_summary.sh" "$@" ;;
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
  metrics
EOF
    exit 1
    ;;
esac
