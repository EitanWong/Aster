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

## Reference Comparison

The local `examples/vllm-mlx/vllm_mlx/scheduler.py` and
`examples/vllm-mlx/vllm_mlx/mllm_scheduler.py` use waiting-to-running admission
before execution and batch insertion. Aster keeps its explicit scheduler and
manual fallback until the MLX-LM adapter passes lifecycle, cancellation,
correctness, and live performance gates.
