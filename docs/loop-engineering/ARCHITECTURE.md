# Current Runtime Architecture

The public inference contract is implemented by `aster/inference/engine.py`.
The manual runtime owns one asyncio scheduler loop and delegates MLX/model work
to a single runner executor through `RuntimeKernel`.

## Scheduler Order

Each scheduler step now runs:

1. cancellation processing;
2. one decode batch;
3. waiting-request admission, when active capacity is available;
4. prefill work, with newly admitted requests placed ahead of existing prefill continuations.

This preserves decode priority while reducing the extra prefill turn that a new
short request previously waited through. The implementation is in
`InferenceEngine._scheduler_step` and `_prioritize_new_prefill_admissions`.

## Cache And Memory Boundaries

Admission estimates request bytes before reserving active capacity. Prefix
snapshots are coordinated by `PrefixStore`; live request cache state remains
owned by the runtime kernel. `PagedCache` exists as a separate component, but
the production manual kernel does not yet expose a complete paged KV execution
path.

`aster.inference.paged_kv_adapter.PagedKVCacheLayer` defines an experimental
lossless boundary: full-attention K/V fragments are owned by a persistent
per-layer physical pool, block-table forks use reference counts and COW, and
`PagedAttentionView` exposes the pool plus logical-to-physical indices. The
current MLX-LM attention API still requires contiguous K/V, so the adapter
materializes a fallback view and is not enabled in production. Pool capacity is
grown geometrically and is currently owned by the layer; bundle-level release
and reclamation are available through `PagedKVCacheBundle`. Full-attention
layers use block-table COW while recurrent `ArraysCache` layers are deep-copied
on fork; other recurrent state types still require an explicit contract.

`EngineSettings.paged_cache_enabled` exposes this boundary only as an opt-in
manual-runtime mode. The mode returns a list-compatible owner, releases it in
`InferenceEngine._cleanup_request`, disables prefix snapshots, and still
requires `max_decode_batch=1` because paged snapshot trimming is not yet
integrated with the batched cache contract. The default native MLX-LM cache
path supports persistent decode batching.

The fallback materialization path now keeps a geometrically grown contiguous
buffer per full-attention layer and writes only appended tokens. This removes
repeated block-table concatenation, but it intentionally duplicates pool and
contiguous storage until a block-indexed attention path can consume the pool.

`aster.inference.metal_paged_attention` provides a block-indexed proof path and
a tiled 32-lane SIMD path. The tiled path shares Q/K work across value lanes and
updates online-softmax state once per SIMD group, while preserving the generic
fallback for unsupported dimensions. `PagedBatchAttentionView` and
`paged_batch_block_attention` add a benchmark-only singleton-pool boundary:
multiple request rows carry two-dimensional block tables and sequence lengths
without constructing `[B,Hkv,K,D]` K/V. The view borrows bundle ownership and
retains no block or pool references.

I086 validates B2/B4/B8, unequal lengths, partial-block CoW, numerical parity,
non-materialization, and release. At 10,334/B8 the metadata shape reduces
estimated batch construction by 86.79%, but five fresh processes confirm a
17.70% median-of-process p95 regression for the locked Qwen3.5
`Hq=16/Hkv=4/D=256` shape. No model-runner or attention-bridge call site uses
this batch view; the complete paged boundary stays outside production serving.

## Reference Comparison

The local `examples/vllm-mlx/vllm_mlx/scheduler.py` and
`examples/vllm-mlx/vllm_mlx/mllm_scheduler.py` use waiting-to-running admission
before execution and batch insertion. Aster keeps its explicit scheduler and
manual fallback until the MLX-LM adapter passes lifecycle, cancellation,
correctness, and live performance gates.
