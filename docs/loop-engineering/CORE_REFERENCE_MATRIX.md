# Core Inference Engine Reference Matrix

Updated: 2026-07-15

This matrix is the implementation boundary for the core-first loop.  The
repositories under `examples/` are treated as executable reference material,
not as code to copy wholesale.  Aster adopts a behavior only after it has a
local correctness test, a deterministic baseline, and a resource-aware A/B
measurement.

## Comparison

| Core area | Mature reference practice | Aster today | Gap or risk | Candidate adoption | Required gate |
| --- | --- | --- | --- | --- | --- |
| MLX execution ownership | `mlx-lm` uses a thread-local generation stream; vLLM-MLX explicitly tests that scheduler/model work stays on the owning thread. | Production manual runtime serializes MLX work through one runner executor. Experimental lane streams are user-owned and uncommitted. | Moving work across threads can fail with missing thread-local streams or create nondeterministic cache state. | Keep the single owner as the default. Treat extra streams as an experiment, never as an implicit optimization. | Same-thread assertions, exact hashes, cancellation cleanup, and no stream errors under concurrent load. |
| Decode batching | `mlx-lm`/vLLM-MLX run one decode step over active requests, retain a stable merged cache, and filter finished rows. | Manual runtime already merges stable decode caches and rebuilds only after membership changes. | Membership changes and hybrid cache shapes can invalidate naive mid-batch prefill or cache merges. | Preserve the stable-batch boundary; optimize merge/rebuild cost before expanding admission. | Batch 1 vs batch N exact parity, mixed/staggered lifecycle tests, speed and peak-memory gates. |
| Continuous request lifecycle | Rapid-MLX and vLLM-MLX separate waiting, prompt processing, generation, finished, and cancellation paths; generator close/release is tested. | Aster has explicit request phases, cancellation, admission retry, prefix pinning, and runtime cache recovery. | Every terminal path must release snapshots, cache references, and runtime objects; leaks can be silent until long runs. | Expand lifecycle probes and metrics before changing scheduling semantics. | Repeated cancel/finish/recovery runs leave zero pinned entries and bounded memory. |
| Chunked prefill | `mlx-lm.BatchGenerator` admits pending prompts, processes them in chunks, then moves completed prompts into decode; OMLX exposes `prefill_step_size` and chunked-prefill controls. | Aster has an explicit prefill queue and configurable token budget, with pressure fallback. | The budget is mostly a total-memory heuristic; activation memory for the actual attention route is not priced explicitly. | Add configuration-aware transient prefill accounting/admission as a narrow core improvement. | Unit tests for estimator and admission decisions; long-context no-OOM/swap regression; A/B must not regress short or mixed workloads. |
| Prefill memory safety | OMLX estimates unfused SDPA score/output tensors from query tokens, KV length, heads, and head dimension, and rejects/evicts before a dangerous call. | Aster catches `MemoryError` after the call and then evicts/rejects; `estimate_request_bytes` prices persistent KV/state only. | A long prompt can exceed memory through transient SDPA activations even when persistent admission passes. | Reuse the same model-config dimensions for a conservative preflight estimate; add kernel-route detection only with measured evidence. | Exact config coverage for full/hybrid attention, typed diagnostics, deterministic rejection tests, and resource benchmark. |
| Prefix cache index | SGLang uses a radix tree with page alignment, explicit reference protection, hit/recency metadata, and eviction; Rapid-MLX uses a trie plus LRU/pinned entries and deep-copy-on-fetch. | Aster uses token-sequence lookup with bounded full snapshots, pins, LCP/exact stats, and recent/sparse retention policies. | Full snapshot cloning and flat lookup can make long/divergent histories expensive; no structural sharing or page-level ownership yet. | First measure lookup/clone cost and lifecycle pressure. Consider radix/page sharing only after a minimal compatible design is proven. | Hash/token parity, clone isolation, eviction/ref protection, memory slope, and no regression for exact/append/branch reuse. |
| Cache lifecycle | vLLM-MLX closes/replaces generators, periodically clears MLX cache, and uses incremental evaluation during cleanup to avoid peaks. | Aster has runtime-cache clear/recovery and explicit prefix-store eviction. | Cleanup behavior across all terminal/error paths needs continuous stress evidence. | Keep cleanup centralized; add tests/metrics rather than scattering `mx.clear_cache`. | Long cancellation/restart loop, zero pinned state, bounded RSS/MLX allocator, no output drift. |
| Scheduling policy | Rapid-MLX uses waiting/running queues and one-token decode steps; OMLX separates scheduler limits for sequences and batched tokens. | Aster prioritizes decode, rotates yielding prefill continuations, and has active-request/admission limits. | Long prefill fairness and decode latency compete; changing priority without measurement can worsen staggered p95. | Profile queue wait, prefill chunk duration, and decode starvation before tuning policy. | Mixed/staggered p50+p95, prompt/decode throughput, and exact lifecycle parity. |
| Paged/KV storage | SGLang and OMLX use explicit page/block ownership and eviction; vLLM-MLX has paged-cache experiments. | Aster has experimental paged storage/Metal boundaries, disabled because end-to-end speed did not clear the gate. | Storage indirection can add overhead without a serving win; ownership is more complex than contiguous MLX cache. | Keep disabled; only revisit after a block-indexed consumer and release lifecycle are production-shaped. | At least 3% end-to-end speed win or material memory reduction with exact parity and zero swap growth. |
| Speculative decoding | The three local DFlash repos provide draft/verify, rollback, and MLX/Metal kernel reference designs. | Aster has not integrated DFlash and the current priority is manual runtime foundation. | Speculation can hide core cache/scheduling defects and introduces rollback correctness risk. | Use DFlash only after baseline cache ownership, prefill, and decode lifecycle are stable. | Draft/target parity, rollback tests, acceptance-rate telemetry, and end-to-end speed/resource win. |

## Decisions for the next loop

1. Keep the manual runtime and its single MLX executor as the production
   foundation.
2. Do not re-enable manual prefill batching or integrate DFlash in this phase;
   the hybrid `ArraysCache + KVCache` batch-state parity issue remains a hard
   boundary.
3. Transient-aware prefill admission is implemented in the manual runtime.
   It addresses a reference-proven correctness/resource gap without changing
   cache ownership or decode semantics.
4. Its retention is based on preflight safety, lifecycle tests, and zero-swap
   smoke probes, not a throughput claim. Any future tuning must establish an
   interleaved control and preserve deterministic output hashes.

## Source anchors

- `examples/mlx-lm/mlx_lm/generate.py` — generation stream, wired-memory
  limit, and `BatchGenerator` prompt/decode lifecycle.
- `examples/vllm-mlx/vllm_mlx/mllm_batch_generator.py` — active-batch
  filtering, prompt admission, cache extraction, and terminal cleanup.
- `examples/vllm-mlx/tests/test_engine_core_stream_safety.py` and
  `tests/test_memory_stability.py` — stream ownership and cleanup gates.
- `examples/Rapid-MLX/vllm_mlx/prefix_cache.py` and
  `tests/test_prefix_cache_pressure_eviction.py` — trie/LRU/pinning model.
- `examples/sglang/python/sglang/srt/mem_cache/radix_cache.py` and
  `cpp_radix_tree/tree_v2_impl.h` — radix matching and protected references.
- `examples/omlx/omlx/memory_monitor.py` — transient SDPA estimate and
  prefill memory guard.
- `aster/inference/engine.py`, `aster/inference/model_runner.py`, and
  `aster/inference/runtime_kernel.py` — current Aster scheduling, admission,
  and runtime boundaries.
