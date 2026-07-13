# Iteration 012: Lossless Paged KV Adapter Boundary

- Iteration ID: `ITER-20260713-012`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Start commit: `47edadc`
- End commit: `39502be`
- Runtime: Python 3.14.5, MLX `0.32.0`, MLX-LM `0.31.3`
- Model: Qwen3.5-0.8B-4bit
- Status: `SUCCESS` for the adapter boundary; `INVESTIGATING` for true paged attention

## Problem and Hypothesis

Aster's existing `PagedCacheManager` allocated and indexed Python block
metadata, but no model cache wrote tensor data into those blocks. The manual
runtime therefore remained entirely contiguous, while the batched runtime only
created and removed unused block tables.

Hypothesis: a lossless block-backed object can implement MLX-LM's
`update_and_fetch()` contract, preserve exact attention inputs, and expose a
block table for a future block-indexed Metal kernel. Until such a kernel is
available, the object must explicitly materialize contiguous K/V tensors.

## References and Design

- Installed MLX-LM `mlx_lm/models/cache.py`: `KVCache.update_and_fetch()` and
  `BatchKVCache.merge()` define the current Python cache contract.
- Installed MLX-LM `qwen3_5.py`: full-attention layers consume the returned K/V
  tensors through native scaled dot-product attention; linear layers use
  `ArraysCache` and are preserved unchanged.
- `examples/vllm-mlx/vllm_mlx/paged_cache.py` and `prefix_cache.py`: block
  ownership, chain tables, COW, and reconstruction reference.
- `examples/omlx/omlx/custom_kernels/common/csrc/kernels/steel_attention_block_token.h`:
  block-indexed attention requires a kernel-level block index contract; this is
  not available through the current native MLX Python attention call.

The new `PagedKVCacheLayer` stores each layer's K/V fragments in fixed-size
blocks, shares block tables through `replace_kv_cache_layers()`, and performs
COW before a write to a shared partial block. `PagedAttentionView` exposes
physical block IDs, block size, and logical sequence length. Its
`materialize()` method is intentionally explicit. Batch `merge()` currently
converts to MLX-LM's native `BatchKVCache`, preserving correctness but making
the experimental path unsuitable as a claimed paged-throughput optimization.

## Changes

- Added `aster/inference/paged_kv_adapter.py`.
- Added manager access, table fork, writable-block, and shared-block accounting
  operations in `aster/inference/paged_cache.py`.
- Added MLX tensor tests for block order, COW isolation, attention parity, and
  hybrid-cache layer replacement in `tests/test_paged_kv_adapter.py`.
- No default runtime configuration or production cache factory was changed.

## Correctness Evidence

- Adapter tests: `5 passed`.
- Qwen3.5 0.8B, 2,048 prompt tokens in 256-token chunks, native versus paged
  materializing cache: `max_abs_logit_difference = 0.0`.
- A direct MLX scaled-dot-product attention comparison also matched exactly.
- Six full-attention layers were block-backed; Qwen3.5 linear `ArraysCache`
  layers remained native.

## Performance Evidence

All benchmark runs used the same model, prompt token sequence, 256/512-token
chunks, greedy-independent prefill, one request, and one warmup per path.

| Prompt | Native median | Paged materializing median | Ratio | Blocks |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.7787s | 0.7887s | 1.0129x | 32 |
| 8,192 | 2.5365s | 2.5373s | 1.0003x | 128 |

The 2K and 8K timing differences are below the loop's 3% acceptance gate and
do not establish a speedup. Peak allocator readings varied with MLX/system
state and are not treated as a memory win. The adapter is therefore retained
for ownership and kernel integration work, not enabled as a default
performance path.

Reproduction commands are represented by the JSON artifacts in
`artifacts/ITER-20260713-012-paged-kv-adapter/current/`; they were generated
with local `.venv` Python and the model path recorded in each file.

## Verification

```text
pytest -q: 395 passed, 9 skipped, 1 warning
python -m compileall -q aster tests: passed
git diff --check: passed
jq empty <four benchmark JSON artifacts>: passed
```

## Decision and Risks

Keep `39502be` as an experimental, lossless adapter boundary. Do not enable it
in the manual or batched production runtime: materialization still rebuilds a
contiguous K/V view, and batch merge falls back to native contiguous cache
objects. The implementation does prove the ownership/COW contract and exact
native attention parity needed for the next layer of work.

## Next Priority

Implement or validate a block-indexed MLX/Metal attention entry point that can
consume `PagedAttentionView.block_ids` without materializing all K/V. Then add
bundle-wide fork/release semantics for hybrid caches and compare that kernel
against the tuned contiguous baseline at 12K, 16K, and 30K.
