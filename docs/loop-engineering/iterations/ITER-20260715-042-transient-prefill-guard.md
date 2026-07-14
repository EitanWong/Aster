# Iteration 042 — Transient prefill guard

Date: 2026-07-15

## Objective

Close the manual runtime's prefill-memory gap against OMLX without changing
decode batching, cache ownership, or MLX stream ownership.

## Reference behavior

`examples/omlx/omlx/memory_monitor.py` prices one unfused SDPA call as its
score matrix plus fp32 output. Its prefill guard uses that transient cost
before an unsafe model call, rather than relying solely on persistent KV
estimates or a post-failure recovery path.

## Implementation

- `ModelRunner` derives an immutable `PrefillTransientProfile` from the loaded
  model's text config: query-head count, head dimension, and score dtype size.
- The profile is read through the existing one-thread runner executor during
  request preparation. The scheduler performs only pure arithmetic on that
  immutable data, preserving MLX thread-local stream ownership.
- Before each prefill, the scheduler uses binary search to find the largest
  chunk whose `query_tokens * kv_tokens` score/output estimate fits the current
  memory budget after active requests and prefix snapshots.
- If even one token cannot fit, the request is rejected before `prefill_to` is
  invoked, then follows the existing memory-pressure cleanup path.

## Tests

- The model-runner estimate matches the OMLX-style one-call formula for a
  nested Qwen-like hybrid-attention config.
- A constrained fake runtime clamps a 16-token chunk to 9 tokens.
- A growing-KV request is safely chunked to completion with unchanged text.
- An unaffordable one-token prefill is rejected before any model prefill call.
- The existing single-runner-thread test remains green, protecting MLX stream
  affinity.

## Runtime probes

Prefix-off Qwen3.5-0.8B 4-bit MLX, greedy sampling, macOS arm64, MLX 0.32.0,
mlx-lm 0.31.3:

| Workload | Concurrency | Elapsed | Completion tok/s | Peak MLX | Swap delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| long, 1,205 prompt tokens | 1 | 1.756s | 72.88 | 1.492 GB | 0 |
| long, 4,820 prompt tokens | 4 | 4.022s | 127.29 | 1.626 GB | 0 |
| mixed, 1,416 prompt tokens | 1 | 2.428s | 118.60 | 1.541 GB | 0 |
| mixed, 1,416 prompt tokens | 4 | 2.351s | 122.48 | 1.541 GB | 0 |

These are post-change smoke probes, not a speedup claim: the 0.8B workload
had sufficient headroom and did not clamp, and a matching interleaved
guard-off control was not run. They establish that the normal path completed
without swap growth or admission failures.

## Decision

Retain the guard as resource/correctness infrastructure. It is not counted as
a performance optimization. The next core iteration must use an interleaved
control experiment or a real pressure workload before making a latency or
throughput claim.

