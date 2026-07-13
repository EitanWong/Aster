# Iteration 017: Reuse Contiguous Paged KV Fallback

- Iteration ID: `ITER-20260714-017-reuse-contiguous-paged-fallback`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `9414013`
- Code end commit: `0e69890`
- Docs end commit: recorded by the following documentation commit

## Problem And Hypothesis

The opt-in hybrid paged path was functionally correct but 39% slower at 8K in
the first single-run probe. `PagedKVCacheLayer.update_and_fetch()` rebuilt a
contiguous K/V result from every physical block on every append. The hypothesis
was that a per-layer geometrically grown contiguous fallback, updated only for
new tokens, would remove the dominant repeated concatenation cost.

## Changes

- Added a per-layer contiguous fallback buffer with geometric capacity growth.
- `update_and_fetch()` now writes only `[start:end]` into that buffer.
- `state` returns a view of the cached prefix instead of concatenating all
  blocks after every update.
- Added trim-then-append support, including COW before truncating a shared
  partial physical block.
- `nbytes` includes both persistent block storage and contiguous fallback so
  admission/memory reporting does not undercount the experimental path.

## Correctness And Tests

The affected tests cover persistent pool reuse, fallback buffer reuse, trim and
append, hybrid recurrent state copying, full KV COW, and engine cleanup release.

```text
.venv/bin/pytest -q tests/test_paged_kv_adapter.py
.venv/bin/pytest -q tests/test_model_runner.py::test_opt_in_paged_prompt_cache_preserves_hybrid_list_shape_and_owner tests/test_engine_runtime.py::test_engine_cleanup_releases_owned_prompt_cache
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests scripts/dev
.venv/bin/pip check
git diff --check
```

## Benchmark

The production-shaped workload used Qwen3.5-0.8B 4-bit, greedy sampling, one
request, 128 output tokens, manual runtime, and 8,373 prompt tokens. Six
process-level measurements were interleaved in order
`paged, native, native, paged, paged, native`; each process included model
load and warmup. Raw and labeled records are in
`iterations/artifacts/ITER-20260714-017-reuse-contiguous-paged-fallback/current/`.

| Path | Median elapsed | Median completion tok/s | Peak MLX memory |
| --- | ---: | ---: | ---: |
| Native, 3 runs | `5.448s` | `23.50` | `2.297 GB` |
| Opt-in paged, 3 runs | `5.425s` | `23.60` | `2.471 GB` |

The `-0.4%` elapsed difference is below the required 3% gate and does not
establish a performance win. The paged path retains roughly `7.6%` more peak
memory because it keeps both the physical pool and contiguous fallback.

For the shorter 2,229-token single probe, the optimized opt-in path measured
`2.919s` versus native `2.638s` (`+10.6%`) and `1.654 GB` versus `1.677 GB`
peak. Both paths completed successfully with zero swap growth.

## Verification Result

The final full suite passed: `408 passed, 9 skipped, 1 warning` across 417
collected tests. Compileall, pip check, and diff check also passed.

## Decision And Next Priority

Keep the fallback-buffer optimization and opt-in boundary because it removes a
real repeated allocation/copy cost and reaches timing parity at 8K, but do not
change the default path. Next, reduce duplicate pool-plus-contiguous memory or
route a specialized attention kernel directly over the pool. Any default
change requires randomized multi-trial evidence meeting the 3% gate without a
memory regression.
