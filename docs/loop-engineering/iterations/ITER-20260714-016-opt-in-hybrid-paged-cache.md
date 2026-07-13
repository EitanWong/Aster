# Iteration 016: Opt-In Hybrid Paged Cache Boundary

- Iteration ID: `ITER-20260714-016-opt-in-hybrid-paged-cache`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `c6421e0`
- Code end commit: `3d8d131`
- Docs end commit: recorded by the following documentation commit

## Problem And Hypothesis

Qwen3.5 uses a hybrid cache layout: six full-attention `KVCache` layers and
eighteen recurrent `ArraysCache` layers in the 24-layer 0.8B model. The prior
bundle only represented full-attention layers, so it could not be placed at an
actual prompt-cache construction boundary. The hypothesis was that a
list-compatible hybrid owner could preserve model behavior while giving full
KV layers block COW and explicit release.

## Changes

- Added `EngineSettings.paged_cache_enabled`, defaulting to `false`.
- Added a list-compatible `PagedKVCacheList` owner wrapper.
- `ModelRunner` constructs the wrapper only when opt-in is enabled; it requires
  `prefix_cache_enabled=false` and `max_decode_batch=1`.
- Full `KVCache` layers use `PagedKVCacheLayer`; recurrent `ArraysCache` layers
  remain in the list and are deep-copied by bundle fork.
- `InferenceEngine._cleanup_request` invokes `release()` when present, covering
  completed, cancelled, and failed requests.
- Added runner, engine cleanup, and hybrid fork tests. Default runtime behavior
  remains native and unchanged.

## Production Baseline And A/B

Both paths used the same Qwen3.5-0.8B 4-bit model, greedy sampling, one request,
128 output tokens, manual runtime, and the same repeated-word prompts. Raw
records are in
`iterations/artifacts/ITER-20260714-016-opt-in-hybrid-paged-cache/current/`.

| Prompt tokens | Path | Elapsed | Completion tok/s | MLX peak | Active | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2,229 | native | `2.638s` | `48.52` | `1.677 GB` | `0.999 GB` | success |
| 2,229 | opt-in paged | `3.163s` | `40.47` | `2.285 GB` | `1.202 GB` | success |
| 8,373 | native | `5.279s` | `24.25` | `2.297 GB` | `1.277 GB` | success |
| 8,373 | opt-in paged | `7.336s` | `17.45` | `10.681 GB` | `9.850 GB` | success |

The opt-in path was approximately `19.9%` and `39.0%` slower, with peak-memory
increases of approximately `36%` and `365%`. It fails the 3% gate and is not a
default performance improvement.

## Correctness

The native and opt-in paths received the same 10-token prompt and greedy
32-token completion. Both produced identical text, 10 prompt tokens, 32
completion tokens, and `finish_reason=length`; raw responses are in
`native-parity.json` and `paged-parity.json`.

Hybrid fork tests also verified that recurrent `ArraysCache` state is copied,
not shared, while full KV COW remains isolated. All opt-in requests completed
with zero swap growth.

## Verification

```text
.venv/bin/pytest -q tests/test_model_runner.py::test_opt_in_paged_prompt_cache_preserves_hybrid_list_shape_and_owner tests/test_engine_runtime.py::test_engine_cleanup_releases_owned_prompt_cache tests/test_paged_kv_adapter.py
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests
.venv/bin/pip check
git diff --check
```

The focused tests and final full suite passed: `406 passed, 9 skipped, 1
warning` across 415 collected tests.

## Decision And Next Priority

Keep the opt-in boundary because it proves hybrid ownership and cleanup, but
do not enable it by default. The dominant bottleneck is repeated contiguous
materialization from `PagedKVCacheLayer.update_and_fetch`, plus transient pool
growth. Next, profile and reduce those allocations, then run randomized
multi-trial end-to-end A/B. If the 3% gate still fails, retain the design only
as a storage/lifecycle boundary rather than a serving attention path.
