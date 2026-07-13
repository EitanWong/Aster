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

`aster.inference.paged_kv_adapter.PagedKVCacheLayer` now defines an experimental
lossless boundary: full-attention K/V fragments are owned by fixed physical
blocks, block-table forks use reference counts and COW, and
`PagedAttentionView` exposes the physical layout. The current MLX-LM attention
API still requires contiguous K/V, so the adapter materializes a fallback view
and is not enabled in production. A future MLX/Metal block-indexed kernel must
consume that view directly before this boundary can reduce active KV memory or
improve long-context throughput.

The first block-indexed proof kernel now exists in
`aster.inference.metal_paged_attention`, but it is deliberately outside the
production runtime. It validates the ABI and causal/GQA semantics; it does not
yet provide the tiled execution or persistent GPU pool needed for a useful
performance result.

## Reference Comparison

The local `examples/vllm-mlx/vllm_mlx/scheduler.py` and
`examples/vllm-mlx/vllm_mlx/mllm_scheduler.py` use waiting-to-running admission
before execution and batch insertion. Aster keeps its explicit scheduler and
manual fallback until the MLX-LM adapter passes lifecycle, cancellation,
correctness, and live performance gates.
