# Iteration 021: Opt-In Direct Paged Attention Bridge

- Iteration ID: `ITER-20260714-021-direct-paged-bridge`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `d8c351e`
- Code end commits: `9415777`, `74f6a94`
- Evidence: `iterations/artifacts/ITER-20260714-021-direct-paged-bridge/current/`

## Problem And Hypothesis

The token-parallel pool kernel was fast in isolation but was not consumed by
the model. A Qwen3.5-specific, separately gated bridge could use it for
decode while retaining native SDPA for long prefill, provided cache evaluation
and pool ownership did not reintroduce the previous memory regressions.

## Change

- Added `engine.paged_cache_direct_attention_enabled`, disabled by default.
- Added a Qwen3.5-only bridge for `qwen3_next.scaled_dot_product_attention`.
- Direct pool attention is used only for causal `Q<=8`; long prefill and
  unsupported masks materialize from the pool and use native SDPA.
- Direct caches prefill in contiguous storage, promote once at decode init, and
  then release the dense fallback.
- Removed persistent pool-row views from `CacheBlock.cache_data`; the pool
  object is the sole owner of MLX pool storage.
- Added configuration, bridge fallback, lifecycle, and two-stage tests.

## Correctness And Benchmark

Current direct-vs-native greedy parity matched text, prompt tokens, completion
tokens, and finish reason. The randomized 8K order was
`direct, native, native, direct, direct, native`.

| Path | Prompt | Elapsed | Completion tok/s | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: |
| Native | 2,229 | `2.688s` | `47.62` | `1.677 GB` |
| Direct paged | 2,229 | `2.777s` | `46.10` | `1.654 GB` |
| Native | 8,373 | `5.410s` | `23.66` | `2.297 GB` |
| Direct paged | 8,373 | `5.451s` | `23.48` | `2.286 GB` |

Randomized 8K medians were native `5.4423s` versus direct `5.4561s` (`+0.25%`)
and `23.520` versus `23.460` completion tok/s (`-0.25%`). Peak memory was
`2.297 GB` versus `2.286 GB` (`-0.46%`). All six requests completed 128 tokens
with zero swap delta. This is a memory-neutral opt-in bridge, not a speedup
claim and not a default change.

## Failed Experiments Preserved

- Routing all prefill queries through the decode kernel reached `10.59 GB`
  peak and `9.61s` at 8K.
- Keeping pool writes active during every prefill chunk reached `10.68 GB`.
- One-time promotion while retaining pool row views reached `27.22 GB` and
  caused swap growth.

The final two-stage promotion and metadata-only pool ownership removed these
failure modes; raw records remain in the artifact directory.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests scripts/dev
.venv/bin/pip check
git diff --check
```

Result: `417 passed, 9 skipped, 1 warning` across 426 collected tests.

## Decision And Next Priority

Keep direct paged attention as an explicit Qwen3.5-only opt-in. It is
functionally correct and removes the small contiguous-memory overhead, but it
does not meet the 3% speed gate. Next, either optimize decode-only bridge
overhead or evaluate whether the memory benefit justifies broader model and
batch support; never enable it by default without a positive end-to-end gate.
