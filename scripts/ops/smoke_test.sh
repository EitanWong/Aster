#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

require_venv
require_config
url="$(base_url)"
model_name="$($PYTHON_BIN - <<'PY' "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
print((data.get('model') or {}).get('name', 'Qwen3.5-9B'))
PY
)"

echo "Smoke test against $url using model $model_name"

echo "1) health"
"$CURL_BIN" -fsS "$url/health"
echo

echo "2) models"
"$CURL_BIN" -fsS "$url/v1/models"
echo

echo "3) non-stream chat completion (15s timeout)"
if "$CURL_BIN" -fsS --max-time 15 "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$model_name\", \"stream\": false, \"max_tokens\": 32, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: Aster smoke test OK\"}]}"; then
  echo
  echo "Non-stream smoke test passed."
else
  echo
  echo "Non-stream smoke test timed out or failed; trying streaming fallback..."
  bash scripts/chat.sh --stream --timeout 30 "Reply with exactly: Aster smoke test OK"
fi

echo "Smoke test finished."
