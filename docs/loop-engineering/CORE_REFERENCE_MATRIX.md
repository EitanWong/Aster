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
| MLX execution ownership | `mlx-lm` uses a thread-local generation stream; vLLM-MLX explicitly tests scheduler/model ownership; Uzu encodes one native Metal command buffer and exposes explicit submit/wait timing. | Production manual runtime serializes MLX work through one runner executor. Experimental lane streams are user-owned and uncommitted. | Moving work across threads can fail with missing thread-local streams or create nondeterministic cache state. | Keep the single owner as the default. Treat extra streams as an experiment, never as an implicit optimization. | Same-thread assertions, exact hashes, cancellation cleanup, and no stream errors under concurrent load. |
| Decode batching | `mlx-lm`/vLLM-MLX run one decode step over active requests, retain a stable merged cache, and filter finished rows. | Manual runtime already merges stable decode caches and rebuilds only after membership changes. | Membership changes and hybrid cache shapes can invalidate naive mid-batch prefill or cache merges. | Preserve the stable-batch boundary; optimize merge/rebuild cost before expanding admission. | Batch 1 vs batch N exact parity, mixed/staggered lifecycle tests, speed and peak-memory gates. |
| Continuous request lifecycle | Rapid-MLX and vLLM-MLX separate waiting, prompt processing, generation, finished, and cancellation paths; generator close/release is tested. | Aster has explicit request phases, cancellation, admission retry, prefix pinning, and runtime cache recovery. | Every terminal path must release snapshots, cache references, and runtime objects; leaks can be silent until long runs. | Expand lifecycle probes and metrics before changing scheduling semantics. | Repeated cancel/finish/recovery runs leave zero pinned entries and bounded memory. |
| Chunked prefill | `mlx-lm.BatchGenerator` admits pending prompts, processes them in chunks, then moves completed prompts into decode; OMLX exposes `prefill_step_size` and chunked-prefill controls. | Aster has an explicit prefill queue and configurable token budget, with pressure fallback. | The budget is mostly a total-memory heuristic; activation memory for the actual attention route is not priced explicitly. | Add configuration-aware transient prefill accounting/admission as a narrow core improvement. | Unit tests for estimator and admission decisions; long-context no-OOM/swap regression; A/B must not regress short or mixed workloads. |
| Prefill memory safety | OMLX combines static SDPA pricing with recent measured chunk growth and a safety multiplier before admitting the next chunk. | Aster now records each chunk's MLX peak over the previous active baseline, keeps the highest per-token growth, and combines it with static full-attention pricing. | A long prompt can exceed memory through transient SDPA activations even when persistent admission passes. | Keep the per-request tracker conservative; add kernel-route detection only with measured evidence. | Exact config coverage for full/hybrid attention, typed diagnostics, deterministic rejection tests, and resource benchmark. |
| Prefix cache index | SGLang uses a radix tree with page alignment, explicit reference protection, hit/recency metadata, and eviction; Rapid-MLX uses a trie plus LRU/pinned entries and deep-copy-on-fetch; LM Studio selects the nearest snapshot and clones it before trimming. | Aster uses bounded full snapshots, pins, LCP/exact stats, recent/sparse retention, and a sorted distinct-length index for direct longest-prefix probes. | No structural sharing or page-level ownership yet; a full trie/radix would add memory and ownership complexity at Aster's default 256-entry bound. | Keep the length index after its high-cardinality branch-miss win. Reconsider radix/page sharing only if sustained real traces exceed the bounded index or require page ownership. | Hash/token parity, clone isolation, eviction/ref protection, memory slope, and no regression for exact/append/branch reuse. |
| Cache lifecycle | vLLM-MLX closes/replaces generators, periodically clears MLX cache, and uses incremental evaluation during cleanup to avoid peaks. | Aster has runtime-cache clear/recovery and explicit prefix-store eviction. | Cleanup behavior across all terminal/error paths needs continuous stress evidence. | Keep cleanup centralized; add tests/metrics rather than scattering `mx.clear_cache`. | Long cancellation/restart loop, zero pinned state, bounded RSS/MLX allocator, no output drift. |
| Scheduling policy | Rapid-MLX uses waiting/running queues and one-token decode steps; OMLX separates scheduler limits for sequences and batched tokens. | Aster prioritizes decode, rotates yielding prefill continuations, and has active-request/admission limits. | Long prefill fairness and decode latency compete; changing priority without measurement can worsen staggered p95. | Profile queue wait, prefill chunk duration, and decode starvation before tuning policy. | Mixed/staggered p50+p95, prompt/decode throughput, and exact lifecycle parity. |
| Paged/KV storage | SGLang and OMLX use explicit page/block ownership and eviction; vllm-metal fuses K/V scatter, exposes token-contiguous pages to one varlen kernel, and wraps both operations as lazy MLX C++ Primitives. | Aster's experimental block pool and Metal kernel are disabled because end-to-end speed did not clear the gate, despite strong 8K kernel results. | Storage indirection and graph boundaries can erase the kernel win; private MLX C++ ABI raises build and packaging risk. | Keep Aster's kernel math. Reproduce fused scatter plus lazy Primitive in isolation; do not import vllm-metal's M5-regressing split-KV gate. | Exact parity across layout/mask/batch corners, standalone boundary win, then at least 3% end-to-end speed or material memory gain with zero swap growth. |
| Speculative decoding | The three local DFlash repos provide draft/verify, rollback, and MLX/Metal kernel reference designs. | Aster has not integrated DFlash and the current priority is manual runtime foundation. | Speculation can hide core cache/scheduling defects and introduces rollback correctness risk. | Use DFlash only after baseline cache ownership, prefill, and decode lifecycle are stable. | Draft/target parity, rollback tests, acceptance-rate telemetry, and end-to-end speed/resource win. |

## Decisions for the next loop

1. Keep the manual runtime and its single MLX executor as the production
   foundation.
2. Do not re-enable manual prefill batching or integrate DFlash in this phase;
   the hybrid `ArraysCache + KVCache` batch-state parity issue remains a hard
   boundary.
3. Transient-aware prefill admission is implemented in the manual runtime.
   It combines static pricing with a per-request observed-growth tracker,
   addressing a reference-proven correctness/resource gap without changing
   cache ownership or decode semantics.
4. Its retention is based on a two-pair 9B/30K control: peak MLX memory fell
   3.53% at the median with identical greedy output hashes. It is a
   long-context resource optimization, not a global throughput claim.
5. Prefix lookup now iterates distinct retained lengths and directly probes
   token keys. The 256-entry / 8,192-token divergent-branch microbenchmark
   improved `21.8x`; a full radix owner is not justified at the current bound.
6. The vllm-metal split-KV gate is rejected on this M5 after same-binary A/B.
   Its fused cache write and lazy MLX Primitive remain the next bounded paged
   integration candidate; Aster's current kernel math is retained.

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
- `examples/lmstudio-mlx-engine/mlx_engine/cache_wrapper.py` — nearest-cache
  selection, snapshot cloning, and exact/near-hit trimming.
- `examples/omlx/omlx/memory_monitor.py` — transient SDPA estimate and
  prefill memory guard.
- `examples/vllm-metal/vllm_metal/metal/paged_ops.cpp` and
  `attention/impls/sdpa.py` — lazy MLX Primitive, fused K/V scatter, varlen
  paged attention, and occupancy-gated split-KV reference.
- `examples/uzu/crates/backend-uzu/src/backends/metal/command_buffer.rs` and
  `engine/language_model/` — native Rust/Metal command ownership, timing, and
  DFlash-integrated generation reference.
- `aster/inference/engine.py`, `aster/inference/model_runner.py`, and
  `aster/inference/runtime_kernel.py` — current Aster scheduling, admission,
  and runtime boundaries.
