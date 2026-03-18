#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:-configs/config.yaml}"
bash "$(dirname "$0")/stop.sh" "$CONFIG_PATH" || true
bash "$(dirname "$0")/start.sh" "$CONFIG_PATH"
