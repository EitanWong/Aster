# Iteration 014: Persistent GPU Block Pool

- Iteration ID: `ITER-20260714-014-persistent-gpu-block-pool`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `f89a4e5`
- Code end commit: `f062efc` (with tiled dispatch fix in `0927844`)
- Docs end commit: recorded by the following documentation commit

## Problem And Hypothesis

The experimental paged KV adapter still packed every logical block into a new
`mx.stack` result before invoking a block-indexed kernel. The hypothesis was
that a persistent per-layer physical pool plus a tiled SIMD attention kernel
would remove that packing overhead and share Q/K work across value dimensions.
Success required exact block-order/GQA/causal parity and at least a 3% median
latency improvement in a stable A/B workload. Any correctness regression or
long-context slowdown would keep the path disabled.

## References

- `examples/mlx/mlx/backend/metal/kernels/sdpa_vector.h`: vector attention
  layout and online-softmax accumulation.
- `examples/mlx/mlx/backend/common/metal_kernel.cpp`: MLX custom-kernel
  dispatch semantics, including thread and threadgroup positions.
- `examples/omlx/omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/msa.py`:
  SIMD-lane Q/K fragments for decode attention.

## Changes

- Added `_PagedKVBlockPool` with geometrically grown MLX key/value storage and
  physical block IDs passed directly to `PagedAttentionView`.
- Preserved per-layer COW data when one shared multi-layer block table splits;
  the manager records the source physical block for each COW allocation.
- Added a tiled 32-lane Metal kernel that performs one Q/K reduction per token
  and updates online-softmax state once per SIMD group. Unsupported dimensions
  retain the generic/decode fallbacks.
- Fixed the tiled dispatch after a targeted failure: `metal_kernel` grids are
  thread-count based, so each query now receives 32 threads.
- Added `scripts/dev/benchmark_paged_attention.py` with fixed seeds,
  randomized A/B order, warmup, parity, median/p95 timing, and allocator peak
  memory output.
- No serving path was changed. Native MLX-LM attention remains the production
  path.

## Verification

Commands:

```text
.venv/bin/pytest -q tests/test_paged_kv_adapter.py tests/test_metal_paged_attention.py
.venv/bin/pytest -q
.venv/bin/python scripts/dev/benchmark_paged_attention.py
.venv/bin/python -m compileall -q aster tests
.venv/bin/pip check
git diff --check
```

The focused tests and final full suite passed after the dispatch fix:
`401 passed, 9 skipped, 1 warning` across 410 collected tests.

## Benchmark

The recorded artifact is
`iterations/artifacts/ITER-20260714-014-persistent-gpu-block-pool/current/qwen35-tiled-correct-ab.json`.
It uses Qwen3.5-shaped FP16 tensors `[B=1,Hq=8,Q=1,D=256]`, `Hkv=2`, 64-token
blocks, one warmup per path, seven interleaved measurements per path, and a
reversed logical-to-physical block table. The recorded run produced:

| KV tokens | max abs diff | native median | tiled paged median | paged/native |
| ---: | ---: | ---: | ---: | ---: |
| 512 | `3.05e-05` | `0.000673s` | `0.001051s` | `1.56x` |
| 2,048 | `3.05e-05` | `0.000916s` | `0.003133s` | `3.42x` |
| 8,192 | `3.05e-05` | `0.001198s` | `0.008917s` | `7.44x` |

Peak allocator readings were approximately `0.0021/0.0021 GB` at 512,
`0.0084/0.0084 GB` at 2K, and `0.0338/0.0336 GB` at 8K for native/paged
respectively. These are synthetic kernel measurements, not end-to-end model
latency or energy measurements.

## Failed Experiment And Root Cause

The first tiled benchmark accidentally launched only one eighth of the
required threads for the Qwen-shaped query. The resulting output contained
zeros and showed large parity errors (`0.25` to `3.55`). A D=32 regression test
exposed the issue. The fix multiplies the tiled grid by the 32-lane SIMD width;
the corrected parity values above are the only performance evidence retained.

## Dependencies

On 2026-07-14, the target direct packages were checked with `pip list
--outdated --not-required` and `pip index versions`. `mlx 0.32.0`,
`mlx-lm 0.31.3`, `mlx-audio 0.4.5`, `fastapi 0.139.0`, `uvicorn 0.51.0`,
`pydantic 2.13.4`, and `numpy 2.5.1` were current. `transformers 5.12.1` is
intentionally capped below 5.13 because of the current `mlx-audio` constraint;
the available 5.13.1 release was not installed. `pip check` passed, so no
dependency declaration change was justified.

## Decision And Next Priority

Keep the pool and tiled kernel as an experimental boundary. The implementation
removed one known allocation source and improved the short decode kernel, but
it did not meet the 3% gate and degrades with context length. Do not enable it
in the manual or batched runtime. Next, measure a production-shaped long
context path and then prioritize either a specialized decode kernel or the
missing hybrid-cache bundle fork/release and pool reclamation lifecycle.
