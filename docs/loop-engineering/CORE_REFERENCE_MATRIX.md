# Core Inference Engine Reference Matrix

Updated: 2026-07-29

This matrix is the implementation boundary for the core-first loop.  The
repositories under `examples/` are treated as executable reference material,
not as code to copy wholesale.  Aster adopts a behavior only after it has a
local correctness test, a deterministic baseline, and a resource-aware A/B
measurement.

## Comparison

| Core area | Mature reference practice | Aster today | Gap or risk | Candidate adoption | Required gate |
| --- | --- | --- | --- | --- | --- |
| MLX execution ownership | `mlx-lm` uses a thread-local generation stream; vLLM-MLX explicitly tests scheduler/model ownership; Uzu encodes one native Metal command buffer and exposes explicit submit/wait timing. | Production manual runtime serializes MLX work through one runner executor. Experimental lane streams are user-owned and uncommitted. | Moving work across threads can fail with missing thread-local streams or create nondeterministic cache state. | Keep the single owner as the default. Treat extra streams as an experiment, never as an implicit optimization. | Same-thread assertions, exact hashes, cancellation cleanup, and no stream errors under concurrent load. |
| Input tokenization ingress | [Gigatoken](https://github.com/marcelroed/gigatoken) `34a1599` (MIT) uses Rust SIMD pretokenization, merge-aware caching, and parallel CPU encoding; its compatibility API targets HuggingFace-style tokenizers. | `ModelRunner` uses the model tokenizer for raw/chat prompt encoding, stop/thinking fragments, and the streaming detokenizer. | Faster CPU encoding can improve queueing and TTFT only when ingress is measured as material. A drop-in replacement can drift on special tokens, chat templates, truncation, or incremental decode; it cannot accelerate MLX GPU forward/prefill/decode itself. | Keep the model tokenizer as the authority for templates and detokenization. Evaluate Gigatoken only as an opt-in ingress encoder after an exact Qwen3.5 compatibility probe; do not install or route it in production by default. | Exact IDs for public prompts, API/chat templates, BOS/EOS, stop/thinking/structured fragments, truncation, and randomized Unicode; retained original streaming detokenizer; queue-aware TTFT/e2e improvement >=3% with no decode/RSS/swap regression. |
| Decode batching | `mlx-lm`/vLLM-MLX run one decode step over active requests, retain a stable merged cache, and filter finished rows. | Manual runtime already merges stable decode caches and rebuilds only after membership changes. | Membership changes and hybrid cache shapes can invalidate naive mid-batch prefill or cache merges. | Preserve the stable-batch boundary; optimize merge/rebuild cost before expanding admission. | Batch 1 vs batch N exact parity, mixed/staggered lifecycle tests, speed and peak-memory gates. |
| Decode graph synchronization | MLX-LM groups sampled tokens/logprobs under async evaluation, leaves decode cache state lazy until its next use, and periodically clears allocator cache; prefill evaluates cache explicitly. LM Studio also separates sampling from grouped materialization. | Aster keeps the same lazy cache boundary, clears after 512 generated tokens globally, and now async-submits MLX samples from heterogeneous rows before one group wait. Python-valued custom samplers retain conversion semantics while logits provide a model/KV barrier; post-sample failures do not replay rows. | Host token IDs are still required for stop/stream processing. Tensorizing processor work can misalign random state, structured constraints, or cache lanes when membership changes. | Retain grouped sample synchronization and explicit prefill evaluation. Profile logsumexp and each processor class before considering homogeneous tensorization or backend sampler graphs. | Exact token/text/final-cache bytes; processor/sampler order; Python sampler compatibility; non-replaying failures; membership replace/reorder; stop-aware structured/random/penalty B2/B4/B8; 6K prompts; balanced AB/BA independent processes; sustained allocator/RSS/swap; >=3% core floor. |
| Continuous request lifecycle | Rapid-MLX and vLLM-MLX separate waiting, prompt processing, generation, finished, and cancellation paths; generator close/release is tested. | Aster has explicit request phases, cancellation, admission retry, prefix pinning, and runtime cache recovery. | Every terminal path must release snapshots, cache references, and runtime objects; leaks can be silent until long runs. | Expand lifecycle probes and metrics before changing scheduling semantics. | Repeated cancel/finish/recovery runs leave zero pinned entries and bounded memory. |
| Chunked prefill | `mlx-lm.BatchGenerator` admits pending prompts, processes them in chunks, then moves completed prompts into decode; OMLX exposes `prefill_step_size` and chunked-prefill controls. | Aster has an explicit prefill queue and configurable token budget, with pressure fallback. | The budget is mostly a total-memory heuristic; activation memory for the actual attention route is not priced explicitly. | Add configuration-aware transient prefill accounting/admission as a narrow core improvement. | Unit tests for estimator and admission decisions; long-context no-OOM/swap regression; A/B must not regress short or mixed workloads. |
| Prefill memory safety | OMLX combines static SDPA pricing with recent measured chunk growth and a safety multiplier before admitting the next chunk. | Aster now records each chunk's MLX peak over the previous active baseline, keeps the highest per-token growth, and combines it with static full-attention pricing. | A long prompt can exceed memory through transient SDPA activations even when persistent admission passes. | Keep the per-request tracker conservative; add kernel-route detection only with measured evidence. | Exact config coverage for full/hybrid attention, typed diagnostics, deterministic rejection tests, and resource benchmark. |
| Prefix cache index | SGLang uses a radix tree with page alignment, explicit reference protection, hit/recency metadata, and eviction; Rapid-MLX uses a trie plus LRU/pinned entries and deep-copy-on-fetch; LM Studio selects the nearest snapshot and clones it before trimming. | Aster uses bounded full snapshots, pins, LCP/exact stats, recent/sparse retention, and a sorted distinct-length index for direct longest-prefix probes. | No structural sharing or page-level ownership yet; a full trie/radix would add memory and ownership complexity at Aster's default 256-entry bound. | Keep the length index after its high-cardinality branch-miss win. Reconsider radix/page sharing only if sustained real traces exceed the bounded index or require page ownership. | Hash/token parity, clone isolation, eviction/ref protection, memory slope, and no regression for exact/append/branch reuse. |
| Cache lifecycle | vLLM-MLX closes/replaces generators, periodically clears MLX cache, and uses incremental evaluation during cleanup to avoid peaks. | Aster has runtime-cache clear/recovery, explicit prefix-store eviction, and a 512-generated-token decode clear budget with attempts/failures in diagnostics. | Cleanup behavior across all terminal/error paths still needs continuous stress evidence; prefill and explicit clear must reset the decode budget. | Keep cleanup centralized and token-normalized rather than clearing every scheduler step. | Long cancellation/restart loop, zero pinned state, bounded RSS/MLX allocator, no output drift. |
| Scheduling policy | Rapid-MLX uses waiting/running queues and one-token decode steps; OMLX separates scheduler limits for sequences and batched tokens. | Aster prioritizes decode, rotates yielding prefill continuations, and has active-request/admission limits. | Long prefill fairness and decode latency compete; changing priority without measurement can worsen staggered p95. | Profile queue wait, prefill chunk duration, and decode starvation before tuning policy. | Mixed/staggered p50+p95, prompt/decode throughput, and exact lifecycle parity. |
| Paged/KV storage | SGLang and OMLX use explicit page/block ownership and eviction; vllm-metal fuses K/V scatter, exposes token-contiguous pages to one varlen kernel, and wraps both operations as lazy MLX C++ Primitives. | Aster's experimental block pool and Metal kernel are disabled because end-to-end speed did not clear the gate, despite strong 8K kernel results. Native attention and Aster-layout scatter Primitives were separately rejected after confirmation. | Storage writes and indirection can erase the kernel win; private MLX C++ ABI adds exact-version build and nanobind coupling. vllm-metal's scatter wins in its scheduler-owned layout, but Aster's transfer regressed 64-token batch 4/8. | Retain Aster's pool writes, kernel math, and public `mx.fast` boundary. Profile the complete real-model paged graph before selecting another operator or changing layout ownership. | Exact block/COW/model parity; measured whole-graph bottleneck; >=3% end-to-end speed or material memory gain with zero swap growth. |
| Compressed-domain KV | OMLX/mlx-vlm route TurboQuant caches directly through fused decode kernels and preserve hybrid recurrent layers; Open-TQ-Metal and gemma4metal specialize int4 long-context attention. | Aster has no compressed-domain cache path. Native MLX FP16 remains default; the older generic 4-bit/8-bit prototypes and the new TurboQuant reproduction are disabled. | Isolated compression can improve capacity while losing default-path latency and model quality. Public Open-TQ tests are materially weaker than Aster's token/PPL gate. | Reject measured 4-bit TurboQuant. Preserve its artifacts as a capacity ceiling; reconsider only a model-specific 6/8-bit route with exact greedy, >=99% top-1, <=0.5% PPL change, and no decode regression. | Same-model prefill/decode, 2K/8K/32K stress, token/PPL/top-1, full hybrid-cache bytes, peak/swap, and >=3% default-path gain or an explicit no-regression capacity profile. |
| Speculative decoding | The three local DFlash repos provide draft/verify, rollback, and MLX/Metal kernel reference designs. | Aster has not integrated DFlash and the current priority is manual runtime foundation. | Speculation can hide core cache/scheduling defects and introduces rollback correctness risk. | Use DFlash only after baseline cache ownership, prefill, and decode lifecycle are stable. | Draft/target parity, rollback tests, acceptance-rate telemetry, and end-to-end speed/resource win. |

## 2026-07-29 Engine Gap Assessment

This assessment separates an implementation capability difference from a
measured performance difference. A reference feature is not itself a reason to
import it: Aster's public matrices are comparable only with direct MLX-LM so
far. I069 resolves the high-level public QMSUM component trace, but broader
workload, lower-level attribution, and reference-engine coverage remain
incomplete.

### Current position

- Aster is not a thin wrapper around `mlx-lm`. Its production path owns an
  explicit request lifecycle, admission accounting, prefix pinning,
  cancellation/recovery, bounded chunked prefill, and one-token continuous
  decode scheduling. `Engine._step_decode` batches live rows and
  `ModelRunner._decode_batch` preserves a stable merged cache while grouping
  sampled-token materialization.
- The nearest apples-to-apples public result is I069's four-block QMSUM ABBA
  component trace, which extends the I066/I067 crossed-pair foundation:
  1,380 locked public workloads on Aster and direct MLX-LM in each matrix,
  with identical model/Tokenizer, effective prompt tokens, greedy output-token
  hashes, metric coverage, and zero swap. All nine crossed gates pass, and
  each engine is first for 1,380 public records. I069's 1,600 QMSUM engine
  records pass source/model/execution, cross-block token parity, ABBA,
  component-trace, and zero-swap gates. Its common decode-driver time is
  `+8.791%`/`+8.655%` (Aster/direct) and aggregate decode throughput is
  `-8.177%`/`-8.025%`, by first-engine stratum. Prefill and TTFT stay within
  3%. Cache merge/rebuild, processor dispatch, and result delivery are ruled
  out for B1, but the lazy MLX completion barrier prevents a comparable
  low-level component cause or general engine ranking.
- Aster's production prefill is intentionally per request:
  `ModelRunner.prefill_to` forwards one `[1, tokens]` tensor at a time.
  This is a structural distinction versus the MLX batch-generator references,
  but I068 shows it does not dominate the public single-request QMSUM gap.
  The prior Aster prefill-microbatch attempt failed hybrid-cache parity and
  end-to-end gates, so it remains a conditional arrival-load candidate rather
  than an accepted rewrite.

| Reference group | Confirmed Aster position | Real gap or boundary | Earliest valid next test |
| --- | --- | --- | --- |
| direct MLX-LM | Same-model public adapter is complete; I069's 1,600 public QMSUM records pass all state/component/ABBA gates and Aster exposes an explicit request lifecycle and a batched decode boundary. | Aster has a reproducible `8.0%~8.2%` decode-throughput deficit and `5.5%~5.9%` end-to-end increase. B1 cache merge/rebuild, processor dispatch, and delivery are immaterial; I070 shows that source-call tracing perturbs the timed path and cannot resolve the private lazy completion substep. | I071: establish Aster public-source arrival/load behavior before selecting scheduler, prefill, or cache work. |
| vLLM-MLX and Rapid-MLX | Aster already has waiting/prefill/decode queues, cancellation, bounded prefix snapshots, decode batching, and memory admission. | Their `BatchGenerator`-style prompt batching and page/radix ownership are not equivalent to Aster's one-request prefill and snapshot model. No same-model adapter is locally runnable yet. | Add an adapter only after I067; measure arrival-driven B1/B4/B8 public load with matched model, tokenizer, prompt IDs, output contract, RSS, swap, and request p50/p95. |
| LM Studio MLX Engine | Aster has clone/trim snapshots, prefix lookup, and a model-owned batch path. | LM Studio's nearest-cache restore is a useful small-cache design reference, not proof that a new cache layer is faster. | Profile Aster prefix hit rate, clone time, and retained bytes under real shared-prefix traffic. Reconsider only if the bounded 256-entry snapshot policy becomes the measured limiter. |
| OMLX | Aster already adopted the useful transient-prefill principle: static plus observed activation growth before the next chunk. | OMLX has broader page/block ownership, pressure reclamation, and optional SSD/cache-compression paths. Aster's current snapshot cache does not provide page-level sharing. | First demonstrate capacity, eviction, or high-prefix-reuse pressure that the present bounded store cannot satisfy. Do not re-enable paged or compressed paths merely for architectural parity. |
| vllm-metal | Aster has already reproduced the split-KV, attention-Primitive, and fused-scatter mechanisms, including exact kernel correctness checks. | The complete Aster graph did not retain a >=3% speed win; storage/layout integration erased kernel-level gains. Private MLX C++ ABI and packaging are additional costs. | Collect a real-model per-operator trace. Retry one native subgraph only when attention/KV write is the measured dominant share and the whole graph can pass parity and end-to-end gates. |
| Uzu and llama.cpp | They expose the longer-term native-runtime ceiling: explicit command ownership, backend timing, and low-level kernels/sampling. | They are different runtime/format architectures, not current same-model MLX comparators. Aster has no native backend boundary or GPU-timeline trace yet. | Treat them as a separate backend program after a trace identifies irreducible Python/MLX dispatch overhead; do not infer a direct throughput deficit from heterogeneous model formats. |
| Gigatoken | Aster correctly keeps its model tokenizer authoritative for templates, special tokens, and streaming detokenization. | Gigatoken can affect CPU prompt ingress only; it cannot accelerate MLX prefill or decode. Token-ID/template/detokenizer drift is the primary integration risk. | Instrument render/tokenize time. Evaluate an opt-in encoder only if it is material to queue-aware TTFT, then require exact public/chat/special-token/Unicode IDs and retained original detokenization. |

### Ordered path to match and exceed

1. **I067 result: reject the unstable baseline.** The full reversed public
   core matrix passes comparability, but long-context/QMSUM direction reversals
   prevent a production bottleneck claim.
2. **I068 result: reject the prefill hypothesis for QMSUM.** Four fresh,
   ABBA-ordered public blocks retain a stable decode/end-to-end difference in
   both order strata; TTFT and prefill remain inside the no-op band.
3. **I069 result: rule out high-level B1 candidates.** The source-bound trace
   reproduces the common decode-driver gap in both order strata. It rules out
   cache merge/rebuild, processor dispatch, and delivery, but the dominant
   Aster completion field includes a lazy MLX barrier with no direct private
   counterpart.
4. **I070 result: retain the lower-level attribution limit.** The fresh public
   smoke has exact output, source, and zero-swap parity but fails every 3%
   observer no-op metric. Do not rerun QMSUM ABBA or treat the private lazy
   completion label as an optimization target.
5. **Run a public arrival matrix before scheduler work.** Use the same locked
   workloads with controlled B1/B4/B8 arrivals, staggered long-prefill traffic,
   shared-prefix traffic, and cancellation. This selects between prefill
   microbatching, queue policy, and cache ownership rather than optimizing one
   synthetic cell.
6. **Admit one conditional core candidate.** If trace data shows concurrent
   prefill is dominant, repair `BatchKVCache`/hybrid-cache parity and test
   prompt microbatches. If prefix clone/retention dominates, test structural
   sharing. If post-model decode work dominates, profile processor classes
   before tensorizing. Each branch keeps exact output and lifecycle gates.
7. **Use SIMD at the correct layer.** CPU SIMD/Gigatoken is an ingress/TTFT
   experiment. Metal simdgroup or native Primitive work is a GPU-subgraph
   experiment. Neither is promoted from a standalone kernel or tokenizer
   benchmark; each must beat the native MLX full path by the normal gate.
8. **Defer compression and speculation.** The measured TurboQuant path fails
   default latency and quality parity, while speculation adds rollback/cache
   complexity. They become candidates only after the baseline, tracing, and
   core load matrix are stable.

For an engine-level `match` claim, Aster must stay inside the 3% no-op band
or better in every declared public workload/length/concurrency bin while
preserving the same model, tokenizer, generation contract, output semantics,
RSS, and swap behavior. For an `exceed` claim, the advantage must survive
reversed order, bootstrap bounds, p50 and p95 request metrics, and the full
public matrix for that named engine scope. A single decode-token or isolated
kernel win is evidence for a hypothesis, not an engine result.

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
6. The vllm-metal split-KV gate and a same-math native attention Primitive are
   rejected on this M5. Five-process main testing against a guarded `mx.fast`
   control did not establish a >=3% interval; the two nominal >=3% medians then
   fell below or reversed in confirmation. All 32K/64K intervals crossed zero.
   Aster's current kernel math and `mx.fast` boundary are retained.
7. vllm-metal fused scatter is valid in its own layout: exact pinned builds
   cleared the gate at 1/4/8/16/64/128 tokens. Pure MLX combined storage did not
   transfer the gain, and no Aster-layout cell repeated a >=3% gain across both
   groups; 64-token batch-4/8 instead regressed in both and cleared the
   confirmation regression gate. Retain the current pool writes and move
   measurement to the complete real-model paged graph.
8. Complete real-model profiling rejects direct paged attention as a speed
   path and identifies sampled-token synchronization as the dominant decode
   boundary. TurboQuant 4-bit reduces the hybrid cache but fails default MLX
   speed, teacher top-1, PPL, and 8K token parity. Retain neither runtime
   change.
9. Post-sample cache-eval dependency semantics are now proven for the retained
   boundary. A 142-process archive plus native/recurrent/paged 10,000-step
   stress supports lazy decode cache state and a 512-generated-token allocator
   clear budget. Production improves `9.51%~17.90%` across native/direct batch
   1/2/4. A fixed 512-scheduler-step policy is explicitly rejected because
   long batch 4 accumulated `481.42 MB` free-cache within one interval.
10. Batch sampling now preserves per-row policy while crossing one grouped MLX
    completion barrier. Twenty-four final-source short processes improved
    `+9.89%~+18.06%`; long cells improved `+5.37%~+12.51%`; sustained mixed
    B8 improved `+14.26%`. Exact dynamic-row/cache semantics and zero swap
    growth hold. Eager grouping, scalar concatenation, and greedy-only argmax
    remain rejected.
11. SIMD work is selected by measured layer: Gigatoken is a CPU ingress
    tokenization reference, while Metal simdgroup work is a separate GPU kernel
    path with its own parity and end-to-end gates. The Gigatoken reference is
    pinned remotely at `34a1599` and is not an installed dependency or a
    production tokenizer until an exact Qwen3.5 compatibility and queue-aware
    public-workload measurement proves a material benefit.

## Source anchors

- `examples/mlx-lm/mlx_lm/generate.py` — generation stream, wired-memory
  limit, grouped token/logprob evaluation, and `BatchGenerator` lifecycle.
- `examples/vllm-metal/vllm_metal/v1/sampling_batch.py` — tensorized
  heterogeneous sampling policy reference.
- `examples/omlx/omlx/patches/mlx_lm_mtp/batch_generator.py` — request-UID
  row ownership and membership realignment.
- `examples/lmstudio-mlx-engine/mlx_engine/model_kit/batched_vision/batch_generator.py`
  — sample construction and grouped materialization boundary.
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
- `examples/omlx/omlx/patches/turboquant_attention.py` and pinned
  `mlx_vlm/turboquant.py` — hybrid-cache conversion and compressed-domain
  decode/prefill dispatch.
- local `examples/gemma4metal/lib/turboquant.metal` — Open-TQ-Metal int4
  reproduction and its deliberately weaker public test boundary.
- `examples/vllm-metal/vllm_metal/metal/paged_ops.cpp` and
  `attention/impls/sdpa.py` — lazy MLX Primitive, fused K/V scatter, varlen
  paged attention, and occupancy-gated split-KV reference.
- `examples/uzu/crates/backend-uzu/src/backends/metal/command_buffer.rs` and
  `engine/language_model/` — native Rust/Metal command ownership, timing, and
  DFlash-integrated generation reference.
- `aster/inference/engine.py`, `aster/inference/model_runner.py`, and
  `aster/inference/runtime_kernel.py` — current Aster scheduling, admission,
  and runtime boundaries.
- [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken) at
  `34a1599f0c0ae7d7cd0d1c530e6522320158b360` — CPU SIMD pretokenization,
  token-cache hierarchy, and HuggingFace compatibility reference for a future
  ingress-only probe.
