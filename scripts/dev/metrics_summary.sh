#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-configs/config.yaml}"

require_venv
require_config

url="$(base_url)"
raw="$($CURL_BIN -fsS "$url/metrics")"

"$PYTHON_BIN" - <<'PY' "$raw"
import sys

text = sys.argv[1]
values = {}
for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#'):
        continue
    parts = line.split()
    if len(parts) != 2:
        continue
    name, raw_value = parts
    try:
        values[name] = float(raw_value)
    except ValueError:
        continue

def first(*suffixes: str):
    for key in sorted(values.keys()):
        for suffix in suffixes:
            if key.endswith(suffix):
                return key
    return None

def val(*suffixes: str, default='n/a'):
    key = first(*suffixes)
    if key is None:
        return default
    return values.get(key, default)

def avg(sum_suffix: str, count_suffix: str):
    sum_key = first(sum_suffix)
    count_key = first(count_suffix)
    if sum_key is None or count_key is None:
        return 'n/a'
    count = values.get(count_key, 0.0)
    if count <= 0:
        return 'n/a'
    return round(values.get(sum_key, 0.0) / count, 4)

request_count = val('_request_latency_seconds_count')
request_avg = avg('_request_latency_seconds_sum', '_request_latency_seconds_count')
first_token_avg = avg('_first_token_latency_seconds_sum', '_first_token_latency_seconds_count')
prefill_avg = avg('_prefill_latency_seconds_sum', '_prefill_latency_seconds_count')
decode_avg = avg('_decode_latency_seconds_sum', '_decode_latency_seconds_count')
batch_avg = avg('_batch_size_sum', '_batch_size_count')
queue_depth = val('_queue_depth')
kv_pages_used = val('_kv_pages_used')
prefix_hits = val('_prefix_cache_hits_total', default=0.0)
prefix_misses = val('_prefix_cache_misses_total', default=0.0)
worker_restarts = val('_worker_restarts_total', default=0.0)
spec_accept_avg = avg('_speculative_acceptance_ratio_sum', '_speculative_acceptance_ratio_count')

error_total = 0.0
for key, value in values.items():
    if '_errors_total' in key:
        error_total += value

hit_rate = 'n/a'
if (prefix_hits + prefix_misses) > 0:
    hit_rate = round(prefix_hits / (prefix_hits + prefix_misses), 4)

print('--- Aster Metrics Summary ---')
print(f'request_count              {request_count}')
print(f'request_latency_avg_s      {request_avg}')
print(f'first_token_avg_s          {first_token_avg}')
print(f'prefill_avg_s              {prefill_avg}')
print(f'decode_avg_s               {decode_avg}')
print(f'batch_size_avg             {batch_avg}')
print(f'queue_depth                {queue_depth}')
print(f'kv_pages_used              {kv_pages_used}')
print(f'prefix_cache_hits_total    {prefix_hits}')
print(f'prefix_cache_misses_total  {prefix_misses}')
print(f'prefix_cache_hit_rate      {hit_rate}')
print(f'spec_acceptance_avg        {spec_accept_avg}')
print(f'worker_restarts_total      {worker_restarts}')
print(f'errors_total               {round(error_total, 4)}')
PY
