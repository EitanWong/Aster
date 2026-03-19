# Aster Debugging Guide

This guide is the practical debugging playbook for Aster.

It focuses on the real failure modes and operator workflows already encountered during development.

---

## Quick triage checklist

When Aster seems broken, check in this order:

1. **Is the process up?**
   ```bash
   bash scripts/status.sh
   ```

2. **Is the service healthy and ready?**
   ```bash
   bash scripts/health.sh
   ```

3. **Can it answer a minimal request?**
   ```bash
   bash scripts/smoke_test.sh
   ```

4. **What do the logs say?**
   ```bash
   bash scripts/tail_logs.sh
   ```

5. **What do the metrics say?**
   ```bash
   bash scripts/metrics_summary.sh
   ```

6. **Can the local chat CLI reproduce it?**
   ```bash
   bash scripts/chat.sh --repl --stream
   ```

---

## Core operator tools

### Process and health

```bash
bash scripts/start.sh
bash scripts/stop.sh
bash scripts/restart.sh
bash scripts/status.sh
bash scripts/health.sh
```

### Logs and metrics

```bash
bash scripts/tail_logs.sh
bash scripts/metrics_summary.sh
```

### Functional testing

```bash
bash scripts/smoke_test.sh
bash scripts/chat.sh "Reply with exactly: Aster is online"
bash scripts/chat.sh --stream "Reply with exactly: Aster is online"
```

---

## How to read service health

### `/health`
`/health` answers whether the service is broadly healthy.

Important fields in `details`:

- `engine_healthy`
- `worker_heartbeat_age_s`
- `worker_healthy`
- `scheduler_running`
- `monitor_running`
- `restart_count`
- `degraded`

### `/ready`
`/ready` is stricter. It should be used to answer:

> Is Aster currently ready to serve requests?

Aster currently treats readiness as requiring:

- healthy inference engine
- healthy worker heartbeat
- running scheduler

If `/health` is okay but `/ready` is not, the service is alive but not trustworthy for requests.

---

## How to read logs

Aster now emits request-level JSON logs across route, scheduler, and inference layers.

### Route layer log events

- `chat_request_start`
- `chat_stream_start`
- `chat_scheduler_submit`
- `chat_non_stream_finish`
- `completion_request_start`
- `completion_stream_start`
- `completion_scheduler_submit`
- `completion_non_stream_finish`
- `chat_request_failed`
- `completion_request_failed`

### Scheduler log events

- `scheduler_submit_start`
- `scheduler_submit_enqueued`
- `scheduler_batch_window_start`
- `scheduler_batch_dispatch`
- `scheduler_infer_start`
- `scheduler_infer_finish`
- `scheduler_infer_failed`

### Inference log events

- `infer_start`
- `infer_encoded`
- `infer_prefill_ready`
- `infer_generation_start`
- `infer_first_token`
- `infer_finish_reason`
- `infer_finish`
- `infer_failed`

### Streaming log events

- `stream_start`
- `stream_encoded`
- `stream_generation_start`
- `stream_first_chunk`
- `stream_finish_reason`
- `stream_finish`
- `stream_failed`

### Request tracing

Aster supports `X-Request-Id` and uses `request_id` in logs.

To trace one request end-to-end:

1. send a request with `X-Request-Id`
2. search logs for that id
3. verify route → scheduler → inference continuity

Example:

```bash
curl -si http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: debug-123' \
  -d '{"model":"Qwen3.5-9B","stream":false,"max_tokens":16,"messages":[{"role":"user","content":"Reply with exactly: ok"}]}'
```

Then inspect logs for `debug-123`.

---

## How to read chat CLI metrics

### Recommended interactive mode

```bash
bash scripts/chat.sh --repl --stream
```

### Useful fields

- `ttft_s` — time to first token
- `latency_s` — total request latency
- `completion_tokens` — true output token count when available
- `exact_output_tok_s` — service-side generation speed when available
- `prompt_tps` — effective prompt/prefill speed
- `cache_hit`
- `prefill_cache_hit`
- `generation_cache_reuse`
- `speculative`
- `speculative_path_mode`
- `peak_memory_gb`
- `request_id`

### Interpretation notes

- **Good TTFT but poor total latency** usually means first token is fine but decode throughput is slow.
- **`cache_hit: true`** means repeated-prefix reuse is helping.
- **`prefill_cache_hit: true` with `generation_cache_reuse: false`** means prefix matching helped prefill, but decode did not continue from a reused generation cache path.
- **`exact_output_tok_s`** is more trustworthy than rough character-based token estimates.

---

## OpenAI-compatible vs Aster debug behavior

By default, streaming must stay OpenAI-compatible.

That means a default stream should look like:

```text
data: {"object":"chat.completion.chunk", ...}
data: [DONE]
```

Aster-specific stream diagnostics are only enabled when sending:

```http
X-Aster-Debug: 1
```

`chat.sh` / `chat_cli.py` use this header automatically in streaming mode so they can display real service-side stats.

External OpenAI-compatible clients should not rely on Aster debug events.

See also:
- `docs/OPENAI_COMPAT.md`

---

## Known resolved issue: non-stream chat path hanging

### Symptom

- streaming chat worked
- non-stream chat timed out or appeared hung

### Root cause

The scheduler assumed `request.prompt` always existed:

- non-stream chat requests use `messages`
- calling `.split()` on `request.prompt` failed when `prompt` was `None`
- scheduler execution broke on chat requests
- streaming path bypassed the scheduler, so it still appeared healthy

### Fix

Scheduler request token estimation now handles both:

- prompt-style requests
- message-style chat requests

### Practical lesson

If streaming works but non-stream hangs, inspect the scheduler path first.

---

## When streaming works but non-stream fails

Check this sequence:

1. `chat_request_start`
2. `chat_scheduler_submit`
3. `scheduler_submit_start`
4. `scheduler_batch_dispatch`
5. `scheduler_infer_start`
6. `infer_start`
7. `infer_finish`
8. `chat_non_stream_finish`

If route logs appear but scheduler logs do not, the problem is likely above or inside scheduler submission.

If scheduler logs appear but inference finish never appears, the problem is inside inference or model execution.

If inference finishes but route never returns, inspect response assembly and exception handling.

---

## When generation feels slow

Separate the issue into these buckets:

### 1) TTFT problem

Check:
- `ttft_s`
- `metrics_first_token_delta_s`
- prefill behavior
- prompt size

### 2) Decode throughput problem

Check:
- `exact_output_tok_s`
- `completion_tokens`
- total `latency_s`
- `peak_memory_gb`

### 3) Cache benefit missing

Check:
- `cache_hit`
- `prefill_cache_hit`
- `generation_cache_reuse`
- prefix hit/miss totals in `metrics_summary.sh`

### 4) Scheduler / batching effects

Check:
- `queue_depth_after`
- average batch size from `metrics_summary.sh`
- scheduler logs around batch window and dispatch

---

## Metrics summary interpretation

Run:

```bash
bash scripts/metrics_summary.sh
```

Important lines:

- `request_latency_avg_s`
- `first_token_avg_s`
- `prefill_avg_s`
- `decode_avg_s`
- `batch_size_avg`
- `queue_depth`
- `prefix_cache_hits_total`
- `prefix_cache_misses_total`
- `prefix_cache_hit_rate`
- `worker_restarts_total`
- `errors_total`

### What they usually mean

- high `request_latency_avg_s` + low `first_token_avg_s` → decode is the likely bottleneck
- high `prefix_cache_misses_total` with repeated workloads → cache reuse is not materializing as expected
- non-zero `worker_restarts_total` → service stability issue, inspect logs immediately
- non-zero `errors_total` → look for repeated exception patterns

---

## Typical debugging workflows

### A. Service is down

```bash
bash scripts/status.sh
bash scripts/start.sh
bash scripts/tail_logs.sh
```

### B. Service is up, but requests fail

```bash
bash scripts/health.sh
bash scripts/smoke_test.sh
bash scripts/tail_logs.sh
bash scripts/metrics_summary.sh
```

### C. Streaming works, non-stream fails

```bash
bash scripts/chat.sh --stream "Reply with exactly: ok"
bash scripts/chat.sh "Reply with exactly: ok"
```

Then inspect route + scheduler logs.

### D. Output feels slow

```bash
bash scripts/chat.sh --stream "Explain prefix caching briefly"
bash scripts/metrics_summary.sh
```

Compare:
- TTFT
- exact output tok/s
- cache hit fields
- metrics averages

---

## Recommended future additions

Useful next improvements for debugging:

- `--quiet-stream` in `chat_cli.py` for clean JSON-only streaming stats
- richer rolling request history for recent latencies
- stricter distinction between prefill cost and decode cost in service summaries
- dedicated dashboard or TUI pane for live operational stats

---

## Related docs

- `README.md`
- `docs/OPERATIONS.md`
- `docs/OPENAI_COMPAT.md`
