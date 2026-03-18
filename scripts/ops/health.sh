#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

require_venv
require_config
url="$(base_url)"

echo "GET $url/health"
"$CURL_BIN" -fsS "$url/health"
echo

echo "GET $url/ready"
"$CURL_BIN" -fsS "$url/ready"
echo

echo "GET $url/metrics (first 30 lines)"
"$CURL_BIN" -fsS "$url/metrics" | head -n 30
