#!/usr/bin/env bash
# Uninstall the Aster macOS launchd service and stop any remaining local process.
set -euo pipefail

ASTER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$ASTER_ROOT/scripts/ops"
LAUNCHD_LABEL="com.local.aster.daemon"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"
CLEAN_LOGS=0

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo "Usage: bash scripts/ops/uninstall.sh [--clean-logs]"
      echo ""
      echo "Options:"
      echo "  --clean-logs   Remove logs/aster.log and logs/aster.error.log too"
      exit 0
      ;;
    --clean-logs)
      CLEAN_LOGS=1
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: bash scripts/ops/uninstall.sh [--clean-logs]" >&2
      exit 1
      ;;
  esac
done

resolve_python() {
  if [[ -x "$ASTER_ROOT/.venv/bin/python" ]]; then
    echo "$ASTER_ROOT/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

PYTHON_BIN="$(resolve_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "error: no usable Python interpreter found" >&2
  exit 1
fi

echo "Removing Aster background service..."

if [[ -f "$LAUNCHD_PLIST" ]]; then
  "$PYTHON_BIN" "$SCRIPTS_DIR/daemon.py" uninstall || true
else
  echo "launchd plist not found: $LAUNCHD_PLIST"
fi

# Also stop any directly managed local process that may still be running.
bash "$SCRIPTS_DIR/stop.sh" || true

rm -f "$ASTER_ROOT/run/aster.pid" "$ASTER_ROOT/logs/aster.pid"

if [[ "$CLEAN_LOGS" == "1" ]]; then
  rm -f "$ASTER_ROOT/logs/aster.log" "$ASTER_ROOT/logs/aster.error.log"
  echo "Removed log files"
fi

if [[ -f "$LAUNCHD_PLIST" ]]; then
  echo "warning: launchd plist still exists at $LAUNCHD_PLIST" >&2
  exit 1
fi

echo "Aster service uninstalled"
