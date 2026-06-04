# Native MLX Engine Rewrite

## Current Runtime

Aster text inference now runs through one engine-owned MLX path:

- `InferenceEngine`
- `ModelRunner`
- `PrefixStore`
- `RequestState`

This replaced the older queue-plus-backend structure and removed the
`vllm-mlx` runtime path from serving.

## What Changed In This Phase

This phase turns the clean rewrite baseline into a more production-shaped
runtime without making it materially larger.

### Decode execution

- Removed per-request decode iterators as the main execution primitive
- Added one-step batched decode in `ModelRunner`
- The engine now assembles a decode batch and advances it in one scheduler step
- Unfinished requests are requeued fairly by the engine

### Scheduler behavior

- Active execution is bounded by `engine.max_active_requests`
- Decode is scheduled ahead of prefill
- Prefill remains chunked, but no longer defines the throughput ceiling by
  forcing decode into per-request stepping
- Cancellation and cleanup remain engine-owned

### Prefix reuse

- Prefix lookup now uses exact digest lookup over known checkpoint lengths
  instead of linear tuple scanning
- Chat requests now expose multiple exact reuse points rather than only one
  boundary
- Reuse instrumentation now reports:
  - attempts
  - hits
  - tokens reused

### Observability

Added lightweight operational counters and gauges for:

- prefill queue depth
- active decode set size
- prefill step count
- decode step count
- generated decode tokens
- queue wait latency
- prefix reuse attempts and reused tokens

## Deliberate Non-Changes

The following were intentionally not added:

- no `vllm-mlx` resurrection
- no sidecar or worker process
- no paged physical KV subsystem
- no speculative decoding
- no KV-cache quantization
- no new scheduler package
- no persistent opaque batch executor

## Current Tradeoff

Decode is now materially more batch-capable, but each batch step still builds a
temporary merged cache and extracts per-request caches back out. That keeps the
runtime compact and explicit, but it may become the next measurable bottleneck
under higher concurrency.

## Validation State

Structurally validated in-repo:

- syntax compilation
- request-state tests
- prefix-store tests
- engine scheduler tests using a fake runner

Still requires a real Apple Silicon MLX environment for truth:

- end-to-end MLX correctness
- real tokens/sec and TTFT
- long-context memory behavior
- cache-merge cost under sustained decode batching
- repeated-prefix win size on live models
