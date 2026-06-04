# Aster Architecture

## Runtime Shape

Aster now runs text inference through one engine-owned MLX runtime.

Top-level runtime concepts:

- `InferenceEngine`: admission, scheduling, cancellation, streaming, metrics
- `ModelRunner`: model load, tokenization, prefill, one-step batched decode
- `PrefixStore`: immutable prompt checkpoints under a byte budget
- `RequestState`: explicit per-request lifecycle and decode state

There is no sidecar, worker supervisor, paged KV allocator, or separate scheduler
process in the serving path.

## Request Lifecycle

Requests move through:

- `submitted`
- `admitted`
- `prefix_lookup`
- `prefill_wait`
- `prefilling`
- `decode_ready`
- `decoding`
- `completed` or `cancelled` or `failed`

The engine owns every transition. API handlers only submit requests and consume
responses or stream chunks.

## Scheduler Model

The scheduler is intentionally small and lives in `engine.py`.

- Submission queue admission is bounded by `api.max_queue_depth`
- Active execution is bounded by `engine.max_active_requests`
- Decode is scheduled before prefill on every engine loop iteration
- Decode requests are served round-robin from `_decode_queue`
- Each decode step advances up to `engine.max_decode_batch` requests together
- Prefill remains chunked and cooperative, but decode no longer advances as
  independent per-request generators

This is the smallest change that makes the engine materially more batch-capable
without introducing a second scheduler abstraction.

## Decode Execution

Decode batching is handled by `ModelRunner.decode_batch_step(...)`.

- Each request contributes its live cache, next input token, sampler, detokenizer,
  and token budget
- For multi-request decode, caches are merged into a temporary batch cache
- One model forward pass advances the batch by one token
- Per-request samplers run on each row of logits
- Updated per-request caches are extracted back out of the merged cache
- Finished requests are removed by the engine after the step

If a cache type cannot be merged safely, the runner falls back to sequential
single-request decode for that step instead of adding more architectural
machinery.

## Prefix Reuse

Prefix reuse is checkpoint-based.

- Reuse entries are immutable cache snapshots
- Requests never share mutable live decode KV
- The store keeps exact checkpoints at safe boundaries:
  - full prompt completion
  - chat-template boundaries that remain exact prefixes
  - periodic long-prefill checkpoints
- Lookup is longest-prefix exact matching over known checkpoint lengths

This keeps reuse deterministic and useful for repeated system prompts, recurring
chat scaffolds, and branch-from-shared-history workloads without importing
page-table complexity.

## What Was Intentionally Avoided

- no `vllm-mlx` runtime dependency
- no fake distributed workers
- no CUDA-style paged KV subsystem
- no speculative decode in the core runtime
- no separate scheduler package for policy that already lives cleanly in the engine
- no persistent opaque batch object owning request lifecycle

## Current Tradeoff

The engine now has true batched decode stepping, but it still reconstructs a
temporary batch cache for each decode step. That is simpler and more explicit
than a long-lived batch executor, but it may become the next performance ceiling
once live MLX benchmarking is available.
