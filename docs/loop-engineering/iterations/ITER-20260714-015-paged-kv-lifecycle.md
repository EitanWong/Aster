# Iteration 015: Paged KV Bundle Lifecycle

- Iteration ID: `ITER-20260714-015-paged-kv-lifecycle`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `7f72417`
- Code end commit: `c5c2f6b`
- Docs end commit: recorded by the following documentation commit

## Problem And Hypothesis

The persistent pool removed repeated block packing but did not own its memory
lifecycle. `PagedCacheManager.remove_table()` released physical IDs while
`CacheBlock.cache_data` and the layer pool could still retain MLX arrays. The
hypothesis was that a bundle-level owner count plus explicit release would
reclaim full-attention KV memory without breaking fork COW or ordinary prefix
cache semantics.

## Baseline

The manual runtime was measured with the existing Qwen3.5-0.8B 4-bit model,
greedy sampling, one request, 128 output tokens, and the production config
shape. Raw records are under
`iterations/artifacts/ITER-20260714-015-paged-kv-lifecycle/current/`.

| Prompt tokens | Elapsed | Completion tok/s | MLX peak | MLX active | Swap delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,229 | `2.638s` | `48.52` | `1.677 GB` | `0.999 GB` | `0 B` |
| 8,373 | `5.279s` | `24.25` | `2.297 GB` | `1.277 GB` | `0 B` |

These are native manual-runtime baselines, not evidence that the experimental
paged path is faster.

## Changes

- Added reference-counted ownership to `_PagedKVBlockPool`.
- Added `PagedKVCacheBundle` for full-attention layer sets with `fork()` and
  idempotent `release()`.
- Added `discard_cache_data` as an explicit `PagedCacheManager.remove_table`
  option; ordinary removal keeps its prior prefix-cache behavior.
- Cleared block metadata references when the final bundle table reference is
  released, allowing MLX arrays to become reclaimable.
- Added `scripts/dev/benchmark_paged_kv_lifecycle.py` and two fork/release
  regression tests.
- Mixed recurrent/full-attention bundles are deliberately rejected; no serving
  path was changed.

## Verification

```text
.venv/bin/pytest -q tests/test_paged_kv_adapter.py tests/test_metal_paged_attention.py
.venv/bin/pytest -q
.venv/bin/python scripts/dev/benchmark_paged_kv_lifecycle.py
.venv/bin/python -m compileall -q aster tests scripts/dev/benchmark_paged_kv_lifecycle.py
.venv/bin/pip check
git diff --check
```

Results: focused `13 passed`; full suite `403 passed, 9 skipped, 1 warning`
across 412 collected tests. Compileall, pip check, and diff check passed.

## Lifecycle Evidence

Artifact: `iterations/artifacts/ITER-20260714-015-paged-kv-lifecycle/current/qwen35-paged-kv-lifecycle.json`.

For `[B=1,H=2,T=512,D=256]` FP16 KV, the persistent pool was
`2,097,152` bytes before and after a child fork. Releasing the child left the
source pool alive at `2,097,152` bytes. Releasing the source reclaimed the
pool to `0` bytes and left `0` manager allocated blocks. After an explicit
`mx.clear_cache()` in the isolated process, active memory was `16` bytes.

The key implementation detail is clearing `CacheBlock.cache_data` only when
the requested table release makes the block's reference count zero. This
avoids retaining pool rows while preserving blocks still referenced by a fork
or ordinary prefix-cache users.

## Risks And Decision

The bundle is currently full-attention only. Recurrent layers may have different
state semantics and must not be shallow-copied or released with full-attention
blocks. The pool lifecycle is not wired into the production manual runtime,
and the Metal kernel remains slower than native attention at 2K/8K.

Keep the implementation as an opt-in experimental boundary. Next, integrate it
at a controlled cache construction point, then add a hybrid bundle contract and
end-to-end A/B against the native 0.8B baseline before any default change.
