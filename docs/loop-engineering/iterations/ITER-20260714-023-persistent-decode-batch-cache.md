# Iteration 023: Persist Merged Decode Cache Across Stable Batches

- Iteration ID: `ITER-20260714-023-persistent-decode-batch-cache`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `c92f67c`
- Code commit: `b721554`
- Evidence: `iterations/artifacts/ITER-20260714-023-persistent-decode-batch-cache/current/`

## Problem And Hypothesis

The manual runtime already grouped concurrent decode requests, but every
multi-request step called `merge` on all request-local KV caches and then
called `extract` for every row. Those copies scale with context length and can
cost more than the batched model forward. Keeping the merged cache alive while
the request membership and order stay unchanged should remove those repeated
copies without changing model inputs or sampling.

## Design And Reference

The implementation follows Aster's existing `InferenceEngine` ownership model:
the engine remains responsible for scheduling and membership, while
`ModelRunner` owns the temporary merged cache context. `DecodeWorkItem` carries
the stable request ID. The runner returns a private reference containing the
merged context and row index. When the next batch has the same ordered IDs, it
reuses the context. When membership changes, references are materialized once,
the batch is rebuilt, and the new context becomes current.

The installed `mlx-lm 0.31.3` cache implementation was used as the API
reference, specifically `BatchKVCache.merge`, `BatchKVCache.extract`, and
`BatchKVCache.update_and_fetch`; no dependency source was copied.

## Correctness And Benchmark

All benchmarks used greedy sampling (`temperature=0.0`), manual runtime, four
concurrent requests, and zero prefix-cache reuse unless stated otherwise.
Raw JSON records are in the artifact directory.

### 0.8B, 8K Prompt, No Prefix Cache

| Path | Elapsed median | Completion tok/s median | Peak MLX memory |
| --- | ---: | ---: | ---: |
| Batch=4, before | `26.310s` | `19.460` | `1.829 GB` |
| Batch=4, persistent | `17.371s` | `29.476` | `1.829 GB` |

Persistent merged-cache reuse improved batch=4 throughput by `51.5%` and
reduced elapsed time by `34.0%` versus the pre-change path. The current
batch=1 median was `21.726s / 23.568 tok/s`; current batch=2 was
`27.732s / 18.462 tok/s`, so batch=4 was also `59.7%` faster than batch=2
under this no-prefix workload.

### 9B, 512-Word Prompt, No Prefix Cache

Randomized order was `batch2, batch4, batch4, batch2`.

| Path | Elapsed median | Completion tok/s median | Peak MLX memory |
| --- | ---: | ---: | ---: |
| Batch=2 | `37.715s` | `13.576` | `6.256 GB` |
| Batch=4 | `22.025s` | `23.247` | `6.220 GB` |

Batch=4 improved throughput by `71.2%` and reduced elapsed time by `41.6%`.
All eight requests completed, with zero failures and zero swap delta.

### Parity And Membership Changes

Real greedy batch=1 and batch=4 responses had identical text SHA-256 values,
completion token counts, and `length` finish reasons for all four requests.
The batch=4 diagnostic reported 33 successful batched calls, zero fallbacks,
28 persistent cache reuses, and 5 cache rebuilds. Separate mixed and staggered
workloads exercised different request lengths and membership changes; all four
requests completed in each run with zero failures and zero swap growth.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests
.venv/bin/pip check
.venv/bin/ruff check --ignore I001,E402 aster/inference/model_runner.py aster/inference/engine.py tests/test_model_runner.py tests/test_engine_runtime.py
git diff --check
```

Result: `418 passed, 9 skipped, 1 warning` across 427 collected tests. The
focused runner/engine/runtime tests passed `60/60` before the full suite.

## Memory, Power, Risks, And Rollback

Peak MLX memory remained flat in the 0.8B A/B and slightly decreased in the 9B
A/B. Process RSS and allocator peaks were recorded in the raw artifacts; swap
delta was zero in every successful comparison. `powermetrics` remains
unavailable without superuser privileges, so energy-per-token is unavailable.

The persistent context is limited to ordered request identities and native
manual decode. Paged/direct cache paths retain their existing batch-size
restrictions. On unsupported cache types, anonymous work items, batch errors,
or membership changes, the runner falls back to the existing merge/extract or
single-request path.

Rollback is `git revert b721554`; restore the local ignored configuration's
`max_decode_batch` if a model-specific memory budget requires the prior value.

## Decision And Next Priority

Keep the persistent merged-cache path and recommend `max_decode_batch=4` for
the validated native deployment profile. Next, measure prefill batching and
long-context multi-request pressure before raising the batch size further or
integrating paged/direct caches with batching.
