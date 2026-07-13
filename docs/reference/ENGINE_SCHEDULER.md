# Engine Scheduler And Decode Model

## Baseline Problem

After the first rewrite, Aster had one engine-owned runtime, but decode was
still effectively cooperative per request:

- the engine owned queue order
- decode requests were round-robin
- each request was still advanced by its own generator under the runner lock

That kept the architecture clean, but it limited throughput because the hottest
path still behaved like `N` independent model invocations.

## Minimal Options Considered

### Option 1: Scheduler-only refinement

Keep per-request decode iterators and only tune queue policy.

Rejected because:

- fairness improves, but the core decode path remains unbatched
- throughput ceiling does not move much
- it adds policy discussion without fixing the real hot path

### Option 2: Persistent batch executor

Create a dedicated decode-batch object that owns active batch state across steps.

Rejected for this phase because:

- it duplicates lifecycle authority already in `InferenceEngine`
- it adds another stateful subsystem to debug
- it is harder to prove correct before live MLX benchmarking

### Option 3: Batched decode step inside `ModelRunner`

Keep scheduling in `engine.py`, but let `ModelRunner` advance multiple decode
requests in one model forward pass.

Chosen because:

- it materially changes the hot path
- it preserves single-engine ownership
- it stays small enough to reason about

## Chosen Model

### Scheduling

- The submission queue holds admitted-but-not-yet-owned work
- The engine only drains submissions while active work is below
  `engine.max_active_requests`
- Decode always runs before prefill
- Decode fairness is round-robin through `_decode_queue`
- Each engine decode step pulls up to `engine.max_decode_batch` requests

### Decode batching

Each active decode request carries:

- live prompt cache
- next input token
- sampler
- detokenizer
- stop-token set
- completion progress

The runner receives a list of these request-local decode work items.

For a multi-request step, the runner maintains a persistent merged cache when
the active request membership is unchanged:

1. Merge caches into a batch cache when the batch is first formed
2. Run one model forward pass on a `[batch, 1]` token input
3. Sample one token per row
4. Return lightweight per-request references to the merged cache
5. Reuse the merged cache on the next stable batch step
6. Extract and re-merge once when membership changes, then continue

Single-request steps and batches without request identities retain the direct
request-local cache path. A failed batch clears the persistent context and
falls back to per-item decode so unsupported cache types remain recoverable.

The engine then:

- updates request-local state
- streams any text segments
- completes or requeues each request

### Prefill coordination

- Decode is always attempted first
- Prefill still runs every loop iteration when work remains
- Prefill is chunked under `engine.prefill_token_budget`
- If the machine is truly idle, the prompt fits within
  `engine.idle_prefill_token_limit`, and the engine is not using the pressure
  budget, prefill can complete in one pass
- Under memory pressure, the engine drops to
  `engine.pressure_prefill_token_budget`

### Prefix checkpoints

Checkpoints are created only at exact boundaries:

- safe chat prefix boundaries
- periodic long-prefill boundaries
- full prompt completion before decode starts

This gives real reuse value without page tables or mutable cache sharing.

## Failure Model

- Cancellation is cooperative at scheduler boundaries
- A batch decode failure fails that batch explicitly instead of hiding the error
- A batch OOM triggers one recovery path:
  - evict cold snapshots
  - preserve older decode requests
  - fail the newest decode in the batch before sacrificing the whole active set

## Next Likely Bottleneck

If live MLX benchmarks show decode is still leaving performance on the table,
the next high-value optimization is likely a persistent batch-cache lane that
avoids merge-and-extract on every decode step. That is intentionally deferred
until this smaller design is measured.

## BatchGenerator Migration Assessment

The latest `examples/vllm-mlx` uses `mlx_lm.BatchGenerator` as the persistent
active-batch owner. That would probably outperform Aster's current
merge-and-extract decode step under sustained concurrency because cache state
stays inside the batch generator instead of being rebuilt every token.

## RuntimeKernel Boundary

Aster now keeps `InferenceEngine` behind a small runtime-kernel boundary:

- `manual`: the current default. It preserves Aster's explicit admission,
  prefill queue, prefix checkpointing, stream flushing, and per-step
  merge/extract decode batching.
- `batch_generator`: an experimental adapter boundary for a future
  `mlx_lm.BatchGenerator` backend. It is intentionally not enabled for serving
  yet; selecting it fails fast with a configuration error.

`SimpleKernel` is not modeled as a third `RuntimeKernel` yet. vllm-mlx's
`SimpleEngine` is an end-to-end serving path optimized for single-user latency;
it bypasses most continuous batching machinery. Aster's current
`RuntimeKernel` only owns model/cache calls inside the existing scheduler, so a
real SimpleKernel would need a higher-level engine strategy boundary:

- `scheduled`: current admission/prefill/decode queues
- `simple`: one request at a time, direct prefill/decode, lower task-switch and
  merge/extract overhead
- `batch_generator`: continuous batch owner backed by `mlx_lm.BatchGenerator`

Until benchmarks show poor single-user latency, the safer path is to keep
SimpleKernel as an assessment item rather than expose a config value that cannot
actually bypass the scheduler.

The boundary is deliberately narrower than vllm-mlx's scheduler. It owns model
and cache operations only:

- prompt encoding
- request and cache memory estimation
- prefix-cache clone/trim
- chunked prefill
- decode initialization
- batched decode step
- detokenizer finalization
- model fingerprinting for cache namespace safety
- strict chat-prefix warmup rendering

`InferenceEngine` still owns request lifecycle, admission control,
prefix-store policy, streaming, cancellation, and metrics. That keeps the
current thread-ownership fix stable while making it possible to swap the decode
kernel later.

The benchmark gate for enabling `batch_generator` is:

- higher completion tokens/sec on `mixed` and `reuse` workloads at concurrency
  2/4/8
- lower or equal p95 latency under the same request mix
- no regression in cancellation cleanup
- no loss of prefix cache hit rate or reused-token counts
- clean behavior for unsafe cache types where trim/reuse must be rejected

Prefix cache entries are also partitioned by model fingerprint, not only by
served model name. That prevents same-name model swaps from reusing stale KV
state and is required before adding SSD persistence or warm-prompt preload.

It is not the right next patch yet. Moving to `BatchGenerator` would change the
runtime ownership boundary:

- request insertion/removal moves from `InferenceEngine` into a batch executor
- cancellation must be deferred to safe scheduler boundaries
- prefix cache restore/store has to align with BatchGenerator cache ownership
- MLX stream binding becomes part of the engine loop contract
- prompt-cache compatibility differs across KV, rotating KV, and hybrid cache
  types

The 2026-07-14 API audit initially found that the experimental `BatchedEngine`
recorded prefix hits without passing stored caches to
`BatchGenerator.insert(caches=...)`. Iteration 027 corrected that boundary by
capturing the `(prompt_len-1, prompt_len)` prefill snapshot, cloning cache
ownership before insertion, and releasing pins on all terminal paths. Exact
and strict-prefix reuse now pass deterministic 0.8B/9B smoke gates, but
divergent hybrid-cache LCP rewind and the separate `BatchGeneratorRuntimeKernel`
adapter remain outside the eligible serving boundary.

The current manual runtime now binds mlx-lm/mlx-vlm generation streams at every
runner-entry boundary. This is separate from single-thread ownership: the
executor ensures all model/cache calls run on one thread, while stream binding
ensures module-level MLX generation streams point at that worker thread.

Warm prompt support follows vllm-mlx's strict-prefix approach. At warmup, Aster
can render a chat template twice with different user probes, find the string LCP
where user content begins, and submit that strict prefix as a raw prompt. That
pre-populates the prefix store without relying only on message-boundary reuse.
Prefix store entries can also be persisted to disk with
`engine.prefix_cache_persist_path`; this is a local pickle format intended as a
first persistence step before SSD tiering or KV quantization.

## Parser And Safety Parity

Aster now carries vllm-mlx-inspired low-coupling parser and safety building
blocks before deeper runtime integration:

- structured JSON schemas are normalized before prompting/validation:
  `$ref` is inlined, unsupported metadata is stripped, `type: [...]` becomes
  `anyOf`, nested unions are flattened, and object schemas default to
  `additionalProperties: false`
- auto tool parsing can recognize common local-model formats such as Qwen XML,
  Qwen bracket calls, Llama function tags, Nemotron parameter tags, MiniMax
  invoke tags, Mistral `[TOOL_CALLS]`, and raw JSON tool call objects. Raw JSON
  tool calls must include both `name` and `arguments`, matching vllm-mlx's
  guard against hijacking structured output objects that merely contain a
  `name` field.
- MCP command validation is available as a standalone runtime module so future
  external tool servers start from a deny-by-default command/env/url safety
  surface
- reasoning extraction now covers the portable vllm-mlx parser families:
  Qwen/DeepSeek `<think>`, GLM box-wrapped `<think>`, GPT-OSS/Harmony channel
  messages, and Gemma4 thought/response channels. `AutoReasoningParser` is used
  by local API response encoding so OpenAI-compatible `reasoning_content` and
  Responses API `response.reasoning_text.*` events are not limited to Qwen tags.
- audio endpoints enforce upload and TTS input limits before invoking ASR/TTS
  services, matching vllm-mlx's resource-protection pattern
- API auth and in-memory rate limiting are available as FastAPI middleware:
  `/health`, `/ready`, and `/metrics` remain public, while model endpoints can
  require `Authorization: Bearer <api.api_key>` and enforce
  `api.rate_limit_per_minute`
- embeddings, transcription, and speech endpoints reject request-time model
  loading unless the requested model is the configured model or a configured
  audio alias; vllm-mlx's alias allowlists are retained as policy data
- a thinking-aware logits processor boundary exists for reasoning-token budget
  enforcement and safe transition into content-phase constrained decoding,
  but it is not yet wired into the manual MLX decode loop
- source-separation has an optional `mlx-audio` service boundary so future audio
  routes can add separation without folding backend-specific code into the API

These are still API/runtime interfaces, not a complete model-family parser
matrix. Missing vllm-mlx parity remains in model-specific streaming tool
parsers such as Hermes, Kimi, Granite, xLAM, GLM4.7, Functionary, Gemma4, and
Harmony. Those should be added through `ToolParserManager` rather than by
expanding prompt emulation indefinitely.

## Gap Refresh After vllm-mlx Review

The latest review corrected a few earlier assumptions:

- Reasoning was not just a Qwen/DeepSeek concern. vllm-mlx has a meaningful
  parser matrix for GPT-OSS/Harmony, Gemma4, and GLM, and some of those formats
  use channel tokens rather than `<think>` tags. Aster now has equivalent
  Aster-native parsers, but still lacks per-model parser selection in model
  metadata.
- The largest remaining runtime gap is still `BatchGenerator`, but vllm-mlx's
  `SimpleEngine` is also important. It provides a direct single-user path with
  lower scheduler overhead, system-prefix KV snapshots, seeded logits
  processors, MTP/specprefill hooks, and processor retirement after thinking.
  Aster has a `RuntimeKernel` boundary but no real `simple` engine strategy yet.
- Multimodal remains much thinner in Aster. vllm-mlx handles URL/base64/path
  media, SSRF/private-network blocking, video frame extraction, VLM processor
  batching, pixel/audio cache keys, and disables unsafe cache reuse for audio.
  Aster has a request model and rejection path, not a usable MLLM processor.
- Cache parity is not just disk persistence. vllm-mlx has paged KV/block-table
  ownership, COW/fork behavior, SSD tiering, memory cache helpers, and
  media-specific cache stores. Aster's `PrefixStore` is safer than the old
  implementation but remains a lightweight prefix snapshot store.
- Audio parity is split across endpoint policy, STT/TTS runtime behavior, and
  source separation. Aster now has resource limits, alias policy, and a
  source-separation service boundary, but no separation route and no full
  vllm-mlx STT/TTS backend feature coverage.

The current plan is to keep the explicit Aster scheduler until live benchmarks
show the temporary cache merge/extract path is the dominant cost. The safe
intermediate steps are now in place: one dedicated MLX runner thread, safer
prefix matching, LCP chat boundaries, generation stream binding, strict-prefix
warmup, disk persistence, and cache rewind checks. Those make a future
`BatchGenerator` migration lower risk because request lifecycle and cache
semantics are now more explicit.

Use `BatchGenerator` migration as the next major runtime project if either of
these shows up in measurement:

- decode throughput flattens as concurrency rises even though GPU utilization
  has headroom
- profiles show per-token cache merge/extract taking a material share of decode
  time
