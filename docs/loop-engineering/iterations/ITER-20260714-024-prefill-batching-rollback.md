# Iteration 024: Roll Back Naive Native Prefill Batching

- Iteration ID: `ITER-20260714-024-prefill-batching-rollback`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `9fb0842`
- End commit: `9fb0842` (no code commit; experiment rolled back)
- Evidence: `iterations/artifacts/ITER-20260714-024-prefill-batching-rollback/current/`

## Problem And Hypothesis

After persistent decode batching removed the main decode merge/extract cost,
long concurrent requests still spent most of their wall time in serial
prefill. The hypothesis was that equal-offset, equal-chunk requests could be
merged into one model forward and then split back into request-local caches.

## Design And Test-First Work

The temporary design added a `PrefillWorkItem`, a native `prefill_batch_to`
runner path using MLX-LM cache `merge`/`extract`, and strict Engine grouping
only for equal cache offsets and equal token chunks. Tests covered the batch
input shape, extracted results, and Engine grouping.

The implementation was intentionally not committed because the live result
failed the performance and memory gates. All temporary source and tests were
restored to `9fb0842` after measurement.

## Baseline And Failed Benchmark

Workload: 0.8B Qwen3.5, manual runtime, no prefix cache, four concurrent
requests, 8,373 prompt tokens each, 128 completion tokens, greedy sampling.

| Path | Elapsed | Completion tok/s | Peak MLX memory | Swap delta |
| --- | ---: | ---: | ---: | ---: |
| Serial prefill baseline | `17.390s` | `29.442` | `1.829 GB` | `0` |
| Naive prefill batch=4 | `23.423s` | `21.859` | `12.886 GB` | `+0.93 GiB` |
| Naive prefill batch=2 | `20.892s` | `24.507` | `3.282 GB` | `0` in that run |

Batch=4 was `34.7%` slower than baseline and introduced swap growth. Batch=2
reduced the memory impact but remained `20.1%` slower and used `79.5%` more
peak MLX memory. All requests completed, but completion success alone is
insufficient for this optimization.

The separate fresh-cache 106-token probe measured individual prefill `0.331s`
and batched prefill `0.223s` (`0.674x` ratio). Greedy argmax tokens matched for
all four rows; maximum floating-point logit difference was `0.1875`. This
microprobe validated cache shape compatibility but did not represent
long-context serving behavior.

## Root Cause And Decision

The `[B, S]` model forward creates large temporary activations and merged cache
storage. At the current 1024-token prefill budget, the memory cost dominates
the saved model launches. The path was rolled back rather than hidden behind a
default because it regressed both latency and memory on the target workload.

## Verification And Next Priority

```text
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests
.venv/bin/pip check
uv lock --check
git diff --check
```

After rollback: `418 passed, 9 skipped, 1 warning` across 427 tests; pip/lock,
compile, and diff checks passed. `powermetrics` remains unavailable without
superuser privileges.

Next, benchmark memory-aware prefill microbatches at chunk sizes
`128/256/512/1024` and batch sizes `1/2/4`, recording per-step peak memory and
swap. Do not implement another prefill default until that matrix identifies a
stable region that clears the 3% gate.
