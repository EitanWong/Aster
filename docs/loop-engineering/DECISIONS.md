# Decisions

## 2026-08-21: Reject I096 Runtime Attribution And Gate Host Quiescence

- Decision: retain benchmark-only telemetry, make no production inference
  change, and advance to a predeclared rolling host-quiescence control.
- Evidence: all 32 paired fresh-process rows pass source, exact output/finish,
  terminal, fallback, swap, prewarm, telemetry, and MLX allocator contracts.
  Aster paired decode medians are `+0.012%` in B4-short and `-1.492%` in
  B4-mixed; direct MLX-LM is `+0.748%` and `+1.266%`. At least one order
  stratum in every cell exceeds `1%`, including Aster short `+5.263%` and
  mixed `-4.459%`.
- Interpretation: a fixed two-second idle exposes but does not control the
  remaining state. System-CPU delta correlations (`r=-0.822` Aster,
  `r=-0.798` MLX-LM) are observational and include child work. The derived
  child-normalized values remain diagnostic, not causal evidence.
- Rollback: remove the benchmark-only telemetry/control files and archived
  artifact; no serving path imports them and no production default changed.
- Next experiment: require rolling two-second pre-launch CPU median `<=6%`,
  p95 `<=12%`, available memory `>=20%`, stable swap, and retained timeout
  evidence before rerunning the 32-row control. MTP and speculation remain
  foundation-gated.

## 2026-08-21: Reject I095 Decode Attribution Until Control State Is Stable

- Decision: keep the sampled observer benchmark-only and make no production
  inference change. The common B4 decode boundary remains confounded by
  fresh-process/control state.
- Evidence: 16 new off/off control rows reuse the locked I094 Qwen3.5-9B,
  public B4 workload, 32-token cap, cache-off state, declared warmup, and
  balanced engine/control order. Exact output/finish, source, terminal,
  fallback, output-cap, and warmup contracts all pass; all 16 new control rows
  have zero workload swap growth. A later audit found `458,752` bytes of
  host-global swap growth in one reused I094 observer-off row, which was not a
  hard gate in the historical I095 summary. Aster B4-mixed
  control-first decode TPS is `+25.825%` versus `+1.464%` observer-off-first;
  the retained paired control deltas include `+48.771%`. MLX-LM control
  strata are `-1.403%/+1.274%`.
- Interpretation: the I094 mixed observer delta cannot be assigned to observer
  work, an Aster decode stage, or a runtime candidate. The predeclared `1%`
  control gate fails even though semantic/resource gates pass.
- Rollback: remove the benchmark-only control harness and its artifact; no
  serving path imports it and no production config changed.
- Next experiment: trace explicit host idle/thermal/allocator state at the
  same boundary, or establish a lower-level kernel control with an equivalent
  state contract before evaluating MTP or speculative decoding.

## 2026-07-13: Admit Before Prefill Continuation

- Decision: retain the manual scheduler and process waiting admissions after decode but before prefill.
- Reason: request timelines and existing tests showed short requests could wait behind a long prefill even when decode was idle.
- Alternative rejected: enabling `batch_generator` immediately. Its adapter is still marked unavailable and would bypass required compatibility evidence.
- Tradeoff: newly admitted prompts can preempt an existing prefill continuation, increasing fairness and short-request responsiveness while delaying the long prompt by one scheduler turn.
- Rollback: revert commit `32addf1`.
- Scope: mixed scheduling and admission latency; no claim about single-request decode kernel speed or global throughput.
- Superseded by the randomized A/B re-evaluation and rollback decision below.

## 2026-07-13: Keep Paged KV as an Experimental Materialization Boundary

- Decision: retain `39502be` as a lossless block ownership/COW adapter, but do
  not enable it in either production runtime.
- Evidence: Qwen3.5-0.8B 2K chunked prefill matched native logits exactly; the
  materializing adapter was `1.29%` slower at 2K and statistically flat at 8K,
  so it did not pass the 3% performance gate or reduce retained KV memory.
- Reason: current MLX-LM attention consumes contiguous K/V and cannot consume a
  block table directly. Enabling the adapter now would add copies without
  proving a user-visible gain.
- Next experiment: validate a block-indexed MLX/Metal kernel against
  `PagedAttentionView`, then add hybrid-cache bundle lifecycle and batch merge
  support.
- Rollback: remove `39502be` and the associated experimental tests/docs; the
  default native cache path is unchanged.

## 2026-07-13: Validate, But Do Not Enable, Block-Indexed Metal Attention

- Decision: retain `31f47cf` as a correctness-first kernel contract and keep it
  disabled in serving paths.
- Evidence: Qwen3.5-shaped FP16 block attention matched native MLX within
  `6.1e-05` maximum absolute difference, but measured `85.96x` slower.
- Reason: the proof kernel assigns one thread to each output value and repeats
  the Q/K reduction; it proves the layout, not a viable execution strategy.
- Next experiment: persistent GPU block pool plus tiled/simdgroup attention,
  with 512/2K/8K parity and performance gates.
- Rollback: remove `31f47cf`; no production cache or attention path depends on
  the experimental module.

## 2026-07-13: Randomized A/B Re-evaluation

- Result: do not accept `32addf1` as a default performance profile yet.
- Evidence: seven interleaved baseline/current trials with greedy sampling gave current elapsed median `+2.86%` and completion throughput `-2.78%`; 95% bootstrap intervals included zero.
- Interpretation: the earlier grouped `-13.6%` result was affected by execution order or workload shape. The scheduler behavior may still help only when short requests arrive during an active long prefill.
- Next experiment: use a staggered arrival workload that submits the long prompt first and short requests after prefill has begun.

## 2026-07-13: Roll Back Admission Scheduling Candidate

- Decision: revert `32addf1` with `5f2b952`.
- Evidence: seven interleaved staggered trials per side with identical 272-token outputs showed short-request p95 `2.4447s -> 2.7454s` (`+12.3%`), aggregate elapsed `4.2352s -> 4.1885s` (`-1.1%`), and completion throughput `64.223 -> 64.940 tok/s` (`+1.1%`). All bootstrap intervals crossed zero.
- Reason: aggregate throughput did not improve materially and the protected short-request metric regressed in the measured workload. Unit-level fairness behavior is insufficient to retain a runtime policy without a stable end-to-end gain.
- Retained work: keep the deterministic/resource-aware/staggered benchmark harness and all raw artifacts for future scheduler candidates.

## 2026-07-14: Keep Persistent Pool and Tiled Attention Experimental

- Decision: retain `094fc43`, `0927844`, and `f062efc` as an experimental
  persistent-pool and block-indexed attention boundary; do not route serving
  traffic through it.
- Evidence: the pool removes per-call block packing and the tiled path matches
  native FP16 attention within `3.1e-05` on the 512/2K/8K Qwen3.5-shaped
  probes. The corrected randomized A/B run measured paged/native median ratios
  of `1.56x`, `3.42x`, and `7.44x`, respectively, so no 3% performance gate
  was met.
- Root cause of the intermediate correctness failure: MLX custom-kernel grids
  are expressed in threads, while the tiled kernel indexes threadgroup
  positions. The dispatch now launches 32 threads per query, with a regression
  test covering the tiled dimension path.
- Dependency result: the target MLX and serving packages were already at their
  current compatible PyPI versions; upgrading Transformers to `5.13.1` would
  violate the current `mlx-audio` compatibility bound, so no package commit
  was made.
- Rollback: revert `094fc43`, `0927844`, and `f062efc`; the production manual
  runtime remains unchanged.
- Next experiment: compare a production-shaped long-context workload against
  native attention, then choose between a more specialized decode kernel and
  hybrid-cache bundle lifecycle/reclamation.

## 2026-07-14: Add Full-Attention Paged KV Bundle Reclamation

- Decision: retain `c5c2f6b` as an experimental lifecycle boundary for a set of
  full-attention layers; do not enable it in serving yet.
- Evidence: a deterministic lifecycle probe retained `2,097,152` pool bytes
  while a fork remained alive, then reclaimed the pool and all manager blocks
  after the source bundle released. The 0.8B manual runtime baseline completed
  2,229 and 8,373 token prompts without swap growth, providing a production
  reference for a future opt-in integration.
- Design: pool owners are reference-counted across bundle forks. Bundle release
  uses `discard_cache_data=True` only when the final table reference is gone,
  preserving ordinary prefix-cache manager behavior.
- Boundary: mixed recurrent/full-attention bundles are rejected because
  recurrent state has no proven fork/release contract yet.
- Rollback: revert `c5c2f6b`; the native manual runtime remains unchanged.
- Next experiment: integrate the full-attention bundle behind an opt-in cache
  boundary, then add and verify hybrid-layer state ownership.

## 2026-07-14: Keep Hybrid Paged Cache Opt-In Only

- Decision: retain `3d8d131` as an opt-in manual-runtime boundary. Keep native
  MLX-LM caches as the default.
- Design: `PagedKVCacheList` remains list-compatible for model execution;
  `ArraysCache` layers are deep-copied on bundle fork, full KV layers use
  physical block COW, and engine cleanup invokes `release()`.
- Correctness: same-model greedy output matched exactly for the 10-token /
  32-token parity smoke; 2.2K and 8.4K requests both completed with zero swap
  growth.
- Performance: opt-in elapsed time was `19.9%` slower at 2.2K and `39.0%`
  slower at 8.4K; peak memory was `36%` and `365%` higher. The 3% gate failed.
- Restrictions: prefix snapshots are disabled and decode batch size must be
  one because clone/merge ownership is not yet safe for paged bundles.
- Rollback: disable `engine.paged_cache_enabled` or revert `3d8d131`.
- Next experiment: attack repeated contiguous materialization and allocation
  overhead, then rerun randomized multi-trial end-to-end A/B.

## 2026-07-14: Reuse Contiguous Paged KV Fallback, Keep Opt-In

- Decision: retain `0e69890` as an experimental optimization; do not enable
  `paged_cache_enabled` by default.
- Change: each paged layer now owns a geometrically grown contiguous fallback
  and writes only the appended token range. Trim performs COW before modifying
  a shared partial block.
- Evidence: the 8K opt-in median improved from the prior `7.336s` single run to
  `5.420s`; randomized 3×3 A/B measured native `5.448s` versus paged `5.425s`,
  only `0.4%` apart and below the 3% gate. Peak memory remained `2.471 GB`
  versus native `2.297 GB`.
- Interpretation: the time bottleneck was repeated materialization, but the
  remaining duplicate pool plus contiguous storage is a memory regression.
- Rollback: revert `0e69890`; the opt-in boundary remains functionally safe.
- Next experiment: reduce duplicate storage or use the pool directly in a
  faster attention path, with randomized multi-trial evidence.

## 2026-07-14: Keep Lazy Paged Pool Promotion Opt-In

- Decision: retain `6772425` as the storage-only default inside the existing
  opt-in paged boundary; keep native MLX-LM caches as the production default.
- Change: the model-runner path now keeps geometrically grown contiguous KV
  storage and promotes into the persistent physical pool only when
  `PagedAttentionView.block_pool()` is explicitly requested. Per-layer token
  accounting and COW cover hybrid layers and forks.
- Correctness: 16 paged-adapter tests passed, full suite passed with `411
  passed, 9 skipped`, and native/storage-only greedy output matched exactly.
- Performance: randomized 8K 3×3 A/B measured native `5.4541s` versus paged
  `5.4526s` (`-0.03%`), below the 3% gate; paged peak memory was `3.38%`
  higher. This is a memory-architecture improvement, not a speedup claim.
- Dependency result: relevant runtime packages were already current within
  project constraints; `transformers 5.13.1` remains excluded by the
  `mlx-audio` compatibility bound `<5.13.0`. No package upgrade was made.
- Rollback: disable `engine.paged_cache_enabled` or revert `6772425`.
- Next experiment: benchmark direct attention over the persistent pool, with
  the same parity, lifecycle, memory, and randomized 3% gates.

## 2026-07-14: Bound Paged KV Fallback Growth

- Decision: retain `89dc086` inside the existing opt-in paged boundary; do not
  change the production default.
- Root cause: geometric doubling expanded the final 8K chunked-prefill fallback
  from 8,192 to 16,384 tokens, creating unnecessary memory pressure.
- Change: grow by `max(step, overflow)` so normal appends reuse capacity while
  the final partial step does not double.
- Evidence: randomized 8K 3×3 A/B measured paged/native elapsed medians of
  `5.4259s/5.4353s` (`-0.17%`), throughput `23.591/23.550` tok/s (`+0.17%`),
  and peak memory `2.286/2.297 GB` (`-0.46%`). All requests completed with
  zero swap delta and current greedy parity remained exact.
- Gate: the 3% speed improvement gate was not met, so this is recorded as a
  memory and allocation improvement rather than a global performance win.
- Rollback: disable `engine.paged_cache_enabled` or revert `89dc086`.
- Next experiment: improve the direct pool attention kernel and integrate it
  only if it clears parity, memory, lifecycle, and randomized end-to-end gates.

## 2026-07-14: Keep Token-Parallel Paged Attention Experimental

- Decision: retain `9d577a8` as the explicit `PagedAttentionView` kernel
  boundary; do not route model serving through it yet.
- Change: partition each aligned long KV scan across 32 simdgroups and reduce
  per-group online-softmax states in threadgroup memory. Short/non-aligned
  shapes retain fallback kernels.
- Evidence: Qwen3.5-shaped 512/2K/8K kernel median ratios were `0.880x`,
  `0.976x`, and `0.733x` versus native SDPA. Max absolute differences were
  `0`, `0`, and `3.05e-05`; the causal multi-query regression passed.
- Boundary: this is not an end-to-end serving result. `Qwen3NextAttention`
  still calls native MLX-LM SDPA, and a bridge must handle masks, cache
  evaluation, decode parity, and fallback behavior before any opt-in serving
  flag is enabled.
- Rollback: revert `9d577a8`; the current model runner remains unchanged.
- Next experiment: add a separate direct-attention opt-in bridge and rerun
  production-shaped parity, memory, lifecycle, and randomized A/B gates.

## 2026-07-14: Keep Direct Paged Attention Opt-In

- Decision: retain `9415777` and `74f6a94` as a Qwen3.5-only opt-in bridge;
  keep native model attention as the default.
- Design: prefill remains contiguous, one-time promotion creates the pool at
  decode init, and only causal `Q<=8` attention calls use the token-parallel
  pool kernel. Unsupported masks fall back to native SDPA after materializing.
- Correctness: current greedy text/token/finish parity matched native, all
  direct 2K/8K requests completed, and the full suite passed with `417
  passed, 9 skipped`.
- Performance: randomized 8K direct/native medians were `5.4561s/5.4423s`
  (`+0.25%`) and `23.460/23.520` completion tok/s (`-0.25%`); peak memory was
  `0.46%` lower. The 3% speed gate was not met.
- Failed paths: all-Q direct prefill, per-chunk pool writes, and retained pool
  row views produced 10.59/10.68/27.22 GB peaks; these are preserved as raw
  evidence and are not used by the final bridge.
- Rollback: leave `engine.paged_cache_direct_attention_enabled=false` or
  revert `9415777` and `74f6a94`.
- Next experiment: reduce decode bridge overhead or test broader batch/model
  support only after new end-to-end evidence clears the same gates.

## 2026-07-14: Cache Direct Paged-Attention Block Indices

- Decision: retain `be48448` as a small internal optimization; keep direct
  paged attention disabled by default.
- Change: cache the `uint32` logical block-index tensor and invalidate it on
  table growth, COW remapping, trim, reset, or pool promotion changes.
- Correctness: the persistent-pool identity test and the full suite passed;
  current result is `417 passed, 9 skipped, 1 warning`.
- Evidence: fresh randomized 8K 3x3 A/B measured native/direct elapsed medians
  of `5.4306s/5.4597s` (`+0.54%`) and completion throughput of
  `23.570/23.445 tok/s` (`-0.53%`). Peak MLX memory stayed at
  `2.297/2.286 GB`, with zero failed requests and zero swap delta.
- Interpretation: reusing the tensor removes repeated small allocations but
  does not produce a statistically meaningful end-to-end speedup. It remains
  below the 3% gate and is not a default-path change.
- Dependency maintenance: refreshed `uv.lock` to 72 compatible packages and
  declared `pydub` plus `python-multipart` in `pyproject.toml`, keeping the
  lock and runtime requirements aligned. `uv lock --check`, `pip check`, and
  compileall pass. `transformers 5.13.1` remains excluded by the current
  `<5.13.0` compatibility bound.
- Rollback: disable `engine.paged_cache_direct_attention_enabled` or revert
  `be48448`; dependency-only rollback is `86ed15c`.
- Next experiment: measure decode-only bridge overhead or broaden model/batch
  support, preserving the same parity, lifecycle, memory, and A/B gates.

## 2026-07-14: Persist Merged Decode Cache Across Stable Batches

- Decision: retain `b721554` and raise the validated native decode batch
  recommendation to `4`; keep paged/direct cache modes at their existing
  explicit restrictions.
- Root cause: the original batched path merged every request-local cache and
  extracted every row on every decode step. Those O(context length) copies
  erased the benefit of a batched model forward, especially for Qwen3.5.
- Change: `ModelRunner` now passes request identities into `DecodeWorkItem`,
  retains the merged cache for an unchanged batch membership, returns private
  cache references, and materializes/remerges only when membership changes.
  Single-request, anonymous, unsupported, and failed-batch paths retain their
  existing fallback behavior.
- Correctness: full suite passed with `418 passed, 9 skipped, 1 warning`.
  Real greedy batch=1 versus batch=4 output hashes, completion counts, and
  finish reasons matched exactly. Mixed and staggered workloads completed all
  four requests with zero failed/cancelled requests and zero swap delta.
- Performance: randomized 0.8B no-prefix 4-request A/B measured batch=4
  baseline/current medians of `19.460/29.476 tok/s` (`+51.5%`) and
  `26.310/17.371s` (`-34.0%`), with `1.829 GB` peak in both paths. Randomized
  9B 2x2 A/B measured batch=2 `13.576 tok/s / 37.715s` versus batch=4
  `23.247 tok/s / 22.025s` (`+71.2%` throughput, `-41.6%` elapsed), with
  peak memory `6.256/6.220 GB` and no swap growth.
- Reference: the installed `mlx-lm 0.31.3` cache contract in
  `mlx_lm.models.cache` (`BatchKVCache.merge`, `BatchKVCache.extract`, and
  `update_and_fetch`) informed the persistent-context boundary; no reference
  code was copied.
- Configuration: the tracked example now recommends `max_decode_batch=4`;
  the ignored local `configs/config.yaml` was also set to 4 for the validated
  9B deployment profile.
- Rollback: revert `b721554` and restore `max_decode_batch` to the previous
  local value if a different model or memory budget regresses. The persistent
  cache automatically falls back on membership changes and batch errors.
- Next experiment: measure prefill batching and long-context multi-request
  memory pressure before considering larger decode batches or broader cache
  integration.

## 2026-07-14: Roll Back Naive Native Prefill Batching

- Decision: roll back the uncommitted prefill-batching implementation; keep
  serial chunked prefill as the production path and retain `b721554`'s native
  decode batching optimization.
- Hypothesis: merging equal-offset request caches and running one `[B, S]`
  model forward would reduce serial prefill wall time for concurrent long
  prompts.
- Evidence: with 0.8B, four concurrent 8K prompts, no prefix reuse, and native
  decode batch 4, baseline was `17.390s / 29.442 tok/s` with `1.829 GB` peak.
  Naive prefill batch 4 was `23.423s / 21.859 tok/s`, with `12.886 GB` peak
  and `0.93 GiB` swap growth. Batch 2 reduced peak to `3.282 GB` but still
  took `20.892s / 24.507 tok/s`, `20.1%` slower than baseline.
- Prototype: a fresh-cache 106-token microprobe showed batched prefill at
  `0.223s` versus individual `0.331s`; greedy argmax tokens matched for all
  four rows, but this short result did not predict long-context memory behavior
  and is not a serving claim.
- Root cause: the merged `[B, S]` forward creates activation and cache
  allocation pressure proportional to batch and chunk size; the extra merge
  and extract work does not compensate at the current 1024-token budget.
- Rollback: all prefill implementation changes were discarded; the full suite
  returned to `418 passed, 9 skipped, 1 warning`. Raw baseline and failed
  benchmark JSON records remain under the iteration artifact directory.
- Next experiment: measure chunk sizes `128/256/512/1024` against batch sizes
  `1/2/4` with per-step peak memory and swap data, then only implement an
  adaptive microbatch selector if it clears the 3% speed gate without memory
  regression.

## 2026-07-14: Roll Back Memory-Aware Prefill Microbatching

- Decision: roll back the second prefill-batching attempt; keep serial native
  prefill as the production path. Do not enable a small-chunk exception without
  resolving deterministic hybrid-cache parity first.
- Measurement: with cache-only evaluation (no full `[B,S,V]` logits eval), the
  isolated 0.8B matrix showed batch=4 was useful at chunk 128/256, but not at
  512/1024. In end-to-end 0.8B 4x8K trials with `prefill_token_budget=256`,
  randomized serial/batch medians were `22.818/20.182s` (`-11.5%`) and
  `22.439/25.369 tok/s` (`+13.1%`); peak memory was `1.662/1.931 GB`, with
  zero swap delta.
- Model coverage: a one-shot 9B 4-request 512-word probe improved
  `25.355s -> 24.358s` and `20.193 -> 21.020 tok/s`, but peak memory rose
  `5.997 -> 6.622 GB`. This is supportive but not sufficient evidence for a
  default change.
- Correctness failure: 0.8B greedy serial-vs-batched parity differed for one
  of four prompts at the text SHA level, despite matching completion length
  and finish reason. The hybrid `ArraysCache + KVCache` batched prefill state
  therefore fails the project's deterministic correctness gate.
- Root causes found: forcing `mx.eval(logits)` caused a separate severe memory
  regression (`12.886 GB` peak and `0.93 GiB` swap at batch=4); removing that
  mistake exposed the remaining model/cache numerical divergence.
- Rollback: all temporary prefill batching source and tests were discarded;
  full suite returned to `418 passed, 9 skipped, 1 warning`. Matrix and parity
  raw evidence remain in the iteration artifact directory.
- Next experiment: audit the model-native `mlx_lm.BatchGenerator` path and its
  cache ownership/cancellation behavior rather than reimplementing hybrid
  prefill batching inside the manual runner.

## 2026-07-14: Keep BatchGenerator Experimental After API Audit

- Decision: do not enable or broaden `BatchedEngine` in this iteration; keep
  the manual runtime as the production default.
- Package result: `uv lock --upgrade` resolved 72 packages with no lockfile
  changes. `mlx 0.32.0`, `mlx-lm 0.31.3`, `mlx-audio 0.4.5`, and the serving
  stack are current within the declared compatibility set. `transformers
  5.13.1` is rejected because `mlx-audio 0.4.5` requires `<5.13.0`.
- Runtime evidence: the existing BatchGenerator wrapper completed four
  concurrent 0.8B requests and a cancellation smoke with no failures.
- Blocking correctness issue: prefix lookup pins a stored entry but the
  adapter does not pass its cache through `BatchGenerator.insert(caches=...)`.
  Two identical sequential requests therefore recomputed the 196-token prompt
  and reported no cache hit; response cache flags are also hardcoded false.
- Next experiment: add a cache-shape/ownership adapter with deterministic
  serial-vs-batched parity and explicit restore/store/cancel tests before any
  runtime default or package-bound change.

## 2026-07-14: Restore Prompt-Boundary Caches in BatchedEngine

- Decision: retain `68b0a2b` in the experimental `engine_type=batched` path;
  keep manual runtime as the production default.
- Root cause: the previous wrapper discarded `BatchGenerator` prompt responses,
  never passed stored caches to `insert()`, and stored cache state after
  generation instead of at the reusable prompt boundary.
- Change: consume `BatchGenerator.next()`, capture the final prefill response
  at `(prompt_len-1, prompt_len)`, deep-copy the cache before insertion, pass
  only uncached tokens, and release prefix pins on every terminal path.
- Evidence: 0.8B four-request hot-prefix throughput improved `28.2%` with
  identical output hashes and unchanged peak memory. Exact 12K reuse improved
  elapsed by `91.6%` on 0.8B and `90.6%` on 9B, with zero swap delta. An
  append-only Agent prompt reused 481 tokens with exact no-cache parity.
- Safety: divergent LCP matches that require hybrid-cache rewind remain
  rejected; cancellation, streaming, and follow-up probes left zero pinned
  entries. No default runtime or package bound changed.
- Rollback: revert `68b0a2b`.
- Next experiment: complete BatchedEngine mixed/reuse/staggered concurrency
  2/4/8 matrix and evaluate safe hybrid append-only LCP coverage.

## 2026-07-14: Guard Heterogeneous BatchGenerator Profiles

- Decision: retain `17f20ee` in the experimental BatchedEngine path, but do
  not expose it as the production runtime strategy.
- Root cause: Qwen3.5 hybrid `ArraysCache + KVCache` batches changed greedy
  output hashes when prompt lengths or cache offsets differed. Forcing
  prefill batch size to one fixed only part of the failure; decode merging
  still diverged.
- Change: active requests must share prompt length, cache/no-cache mode, and
  cache token offset before entering the same BatchGenerator profile. Other
  requests wait rather than being mixed.
- Evidence: corrected 30-record 0.8B on/off matrix at concurrency 2/4/8 had
  exact response-hash parity, zero errors, and zero swap delta. Warm cache-on
  elapsed improved `9%~34%`; mixed C=8 improved `16.0%`, long C=8 improved
  `33.5%`. The mixed peak was `3.1%` higher than off, so this is a measured
  latency/cache-reuse tradeoff, not a universal memory win.
- Additional fixes: structured schema argument order, effective EOS stops,
  and special-token filtering now make structured requests valid JSON with a
  clean `stop` finish reason.
- Rollback: revert `17f20ee`.
- Next experiment: use separate per-profile BatchGenerator lanes or an
  equivalent scheduler boundary, with the same token parity and memory gates.

## 2026-07-14: Keep Per-Profile BatchGenerator Lanes Experimental

- Decision: retain `d791253` behind `engine.batch_generator_max_lanes`, but
  keep the default at `1` and do not change the manual production runtime.
- Hypothesis: independent BatchGenerator instances would let heterogeneous
  prompt/cache profiles make progress without merging hybrid cache state.
- Change: lane-local generators, UID maps, request ownership, prefix
  extraction, finish, cancellation, and cleanup; all MLX calls remain
  sequential on one engine-loop owner. Empty lanes are recycled when a new
  profile arrives.
- Evidence: with lane `2`, simultaneous mixed records preserved exact hashes
  and improved elapsed time by `2.90%~5.78%` at unchanged `1.495 GB` peak and
  zero swap delta. Structured output was valid for 6/6 responses, prefix reuse
  hit on both rounds, and cancellation/follow-up left zero running or pinned
  entries.
- Failure: staggered arrival produced hash drift in all four lane-2 records
  and was `3.46%~5.80%` slower. The first request starts a profile lane before
  later requests arrive; subsequent membership changes alter BatchGenerator's
  greedy output. This fails the token-parity gate.
- Rollback: set `engine.batch_generator_max_lanes: 1` or revert `d791253`.
- Next experiment: deterministic cohort admission or a BatchGenerator state
  isolation strategy that makes output independent of arrival timing.

## 2026-07-14: Seal Isolated BatchGenerator Cohorts

- Decision: retain `9dbfc7d` as an opt-in safety improvement; keep one lane
  and a zero admission window as the default profile.
- Root cause addressed: a secondary lane could execute with one request and
  then accept later arrivals, changing BatchGenerator batch membership and
  greedy hashes.
- Change: isolated secondary lanes wait for a bounded first-step window,
  then seal permanently until drained. Simultaneous backlog skips the window;
  later requests cannot join a sealed lane. Configurations with more than one
  lane and no positive window are rejected.
- Evidence: lane `2` with `160ms` dynamic cohort admission restored exact
  response-hash parity across 8 mixed/staggered records, with zero errors,
  zero swap delta, and `1.495 GB` peak. Elapsed improved `0.19%~4.99%`.
  Structured, streaming, cancellation, follow-up, and pinned-entry cleanup
  passed.
- Tradeoff: staggered p95 increased `9%~12%`; `100ms` and `140ms` windows
  still missed the final cohort request, while `160ms` and `200ms` had zero
  mismatches. The window is therefore not promoted to the default.
- Rollback: set `engine.batch_generator_max_lanes=1`, or revert `9dbfc7d`.
- Next experiment: event-driven cohort closure or fixed-batch/state isolation
  that removes the p95 wait without reopening late membership changes.

## 2026-07-14: Keep Longest-Lane Priority Experimental

- Decision: retain `5807641` as an explicit BatchGenerator scheduling control;
  keep the default quantum at `1` and the production runtime unchanged.
- Change: a sealed active lane with the longest prompt profile may run a
  bounded number of generator steps before the next lane. Cohort target size
  is independently configurable and defaults to automatic behavior.
- Evidence: lane `2`, a `160ms` window, target size `3`, and quantum `2`
  preserved exact response hashes across 8 mixed/staggered records, with zero
  errors, zero swap growth, and `1.495 GB` peak. Staggered p95 improved
  `3.82%~6.42%` versus quantum `1`.
- Tradeoff: versus lane `1`, staggered elapsed remained `18.13%~18.77%`
  slower and p95 remained `3.72%~4.78%` higher. This is a tail-latency
  improvement inside the safe multi-lane mode, not a global throughput win.
- Rollback: leave `batch_generator_longest_lane_step_quanta=1`, use
  `batch_generator_max_lanes=1`, or revert `5807641`.
- Next experiment: event-driven cohort closure or model-native fixed-batch
  isolation to remove the remaining multi-lane scheduling cost without
  reopening arrival-dependent membership.

## 2026-07-14: Do Not Promote Independent MLX Lane Streams

- Decision: retain the uncommitted `lane_streams` candidate as an explicit
  opt-in experiment only; keep the production/default lane configuration
  unchanged.
- Design: pass a dedicated `mx.new_stream(mx.default_device())` to each
  opt-in `BatchGenerator`, while preserving the single engine-loop owner and
  sequential lane stepping.
- Correctness: the matched 8-record A/B completed with zero errors, identical
  response hashes, zero swap growth, and `1.495 GB` peak MLX memory. Both
  arms' cancellation probes ended with zero running requests and zero pinned
  entries.
- Performance: matched elapsed changes were `-0.84%~-2.53%`, below the 3%
  gate. Interleaved reruns showed only about `0.5%~1.1%` mixed-workload gain;
  staggered p95 varied with arrival timing and once regressed `4.79%`.
- Interpretation: `BatchGenerator.next()` already binds execution to its
  configured stream and synchronously evaluates current tokens, while Aster
  still steps lanes sequentially. Stream assignment alone does not remove the
  scheduler/cohort bottleneck.
- Rollback: disable `batch_generator_lane_streams` or remove only the
  candidate-owned uncommitted changes after ownership is confirmed.
- Next experiment: event-driven cohort closure or model-native fixed-batch
  state isolation with arrival-independent token parity.

## 2026-07-14: Roll Back Event-Driven Cohort Closure

- Decision: remove the event-driven repeated-profile-lane candidate; keep the
  validated 160 ms admission window and one-lane default.
- Design tested: seal each lane before its first step, then create a new lane
  for a late request with the same profile instead of waiting for a cohort.
- Evidence: mixed elapsed improved `0.89%~2.77%`, but staggered elapsed became
  `24.50%~27.73%` slower, p95 became `7.98%~11.22%` higher, and throughput
  fell about 20%. Peak MLX memory stayed at `1.495 GB`, with zero errors and
  zero swap growth.
- Root cause: staggered same-profile arrivals were isolated into repeated
  single-request lanes, while the engine still stepped lanes sequentially.
  Removing the wait therefore removed batching rather than overlapping work.
- Rollback: source and test changes were reverted in the working tree; no
  pre-existing independent-stream changes were modified.
- Next experiment: fixed-shape padding/masking or a model-native state
  isolation boundary that preserves batching without late membership drift.

## 2026-07-14: Roll Back Greedy Batch Argmax Fast Path

- Decision: remove the batch-wide greedy argmax candidate from the manual
  decode path.
- Evidence: Qwen3.5-0.8B manual batch=4 mixed workload measured baseline
  median `1.6682s / 172.642 tok/s` versus candidate
  `1.6860s / 170.884 tok/s`, with unchanged `1.541 GB` peak and zero swap
  growth.
- Root cause: the extra batch reduction/evaluation did not amortize against
  the existing per-row greedy sampling on this MLX workload.
- Rollback: source and test changes were removed; no user working-tree lane
  changes were touched.

## 2026-07-14: Skip Chat Reuse Analysis When Prefix Cache Is Disabled

- Decision: retain `acf785f` in the manual runtime.
- Change: `ModelRunner.encode_request()` skips `_chat_reuse_points()` when
  `engine.prefix_cache_enabled` is false; the enabled path is unchanged.
- Evidence: real 40-turn Qwen3.5-0.8B Agent encoding improved
  `73.136ms -> 1.787ms` median. Five end-to-end 1,718-token / 16-token
  requests improved `2.7199s -> 1.8380s` median, with identical text SHA,
  prompt/completion counts, and finish reason.
- Scope: this is a default-path Agent preprocessing improvement, not a global
  decode throughput claim. Full suite passed with `439 passed, 9 skipped`.
- Rollback: `git revert acf785f`.
- Next experiment: safe fixed-shape/state isolation for multi-lane batching,
  plus prefix-enabled long-context Agent measurements.

## 2026-07-14: Keep Bounded Chat Prompt Token Cache

- Decision: retain `b82599b` in the manual runtime with default capacity 32.
- Design: an LRU scoped to one `ModelRunner` caches a hash of messages,
  thinking mode, and chat-template kwargs, returning copied token IDs and
  reuse-point metadata. KV caches are unaffected; clearing runtime caches also
  clears tokenized prompts.
- Evidence: prefix-enabled 40-turn Agent encode median improved
  `74.082ms -> 0.028ms`. Three fresh-process e2e medians improved exact hot
  reuse `0.2548s -> 0.1788s` (`-29.8%`) and append-only turns
  `0.3032s -> 0.2846s` (`-6.1%`); cold was `1.6%` faster. Hash, token-count,
  finish-reason, memory, and swap checks held.
- Tradeoff: bounded token metadata consumes memory proportional to cached
  prompt length; capacity `0` disables it and the LRU prevents unbounded
  growth.
- Rollback: `git revert b82599b` or set
  `engine.chat_prompt_cache_max_entries=0`.
- Next experiment: fixed-shape/state isolation for multi-lane batching and
  longer branching Agent workload validation.

## 2026-07-14: Bound Chat Snapshot Reuse Points

- Decision: retain `a8913e6` in the manual runtime with the default
  `engine.snapshot_max_chat_reuse_points=8`.
- Design: after chat reuse analysis, keep only the most recent eight reusable
  boundaries. A value of `0` preserves unlimited historical reuse points for
  workloads that prefer cache breadth over retained memory.
- Evidence: on a Qwen3.5-0.8B 40-turn Agent A/B, cold median improved
  `2.5375s -> 1.8549s` (`-26.9%`). Exact, append-only, and branch requests
  retained identical response hashes, cache-hit flags, and saved-token counts.
  Snapshot memory fell from `1.192 GB` / 39 entries to `0.326 GB` / 9 entries
  (`-72.7%`), with zero swap growth.
- Tradeoff: older historical branch points are no longer retained by default;
  longer or unusually branch-diverse conversations may benefit from a larger
  configured budget.
- Rollback: `git revert a8913e6` or set
  `engine.snapshot_max_chat_reuse_points=0`.
- Next experiment: validate the budget across longer and branchier Agent
  traces, then return to fixed-shape/state-isolation work for safe multi-lane
  batching.

## 2026-07-14: Add a Sparse Tier for Long-Context Chat Snapshots

- Decision: retain `0e13e8f` with recent eight-point retention plus four sparse
  older points for prompts at least `2048` tokens.
- Design: older points are selected at exponentially increasing distances from
  the recent window, plus the earliest point that meets the snapshot minimum.
  Shorter prompts remain recent-only. Unlimited behavior remains available
  with `snapshot_max_chat_reuse_points=0`.
- Evidence: the 80-turn Qwen3.5-0.8B 3x3 A/B improved cold median
  `3.6749s -> 2.2592s` (`-38.5%`) and reduced initial snapshot memory
  `3.092 GB -> 0.628 GB` (`-79.7%`). Exact, append, recent, mid, and old
  branches all retained cache hits, identical hashes, and zero swap growth.
- Tradeoff: mid/old branches saved fewer tokens and took longer than the
  unlimited baseline because they resume from a sparse ancestor; this is an
  explicit bounded-memory tradeoff, not a universal branch-latency claim.
- Rollback: `git revert 0e13e8f`, set
  `engine.snapshot_chat_reuse_sparse_points=0`, or set
  `engine.snapshot_max_chat_reuse_points=0`.
- Next experiment: sustained repeated-branch/cancellation/recovery traces,
  followed by model-native fixed-shape state isolation for batching.

## 2026-07-14: Skip Full Snapshots for Non-Exact Prefix Hits

- Decision: retain `07bd566` with
  `engine.snapshot_skip_full_prompt_on_prefix_hit=true` by default.
- Design: cold requests and exact prefix hits retain full prompt snapshots;
  non-exact prefix-hit branches reuse their ancestor and do not create another
  full-prompt KV clone. The rule is enforced in both prefill-end checkpointing
  and decode activation.
- Evidence: a Qwen3.5-0.8B 80-turn / 12-branch 3x3 A/B reduced post-recovery
  snapshot memory `1.511 GB -> 0.739 GB` (`-51.1%`) and entries `29 -> 15`.
  All scenario hashes, token counts, hit flags, and saved-token counts matched;
  cancellation left zero pinned entries and recovery hit the cache.
- Randomized follow-up: four same-seed fresh-process pairs measured candidate
  deltas of `+2.13%` cold, `+1.34%` exact, `+0.63%` append, `+0.49%/-0.17%/-0.03%`
  for branch 1/6/12, and `+0.51%` recovery. The grouped `+5.5%` branch result
  did not reproduce; final memory remained `1.511 GB -> 0.739 GB` in every
  seed. This clears the memory goal without a material branch regression.
- Rollback: `git revert 07bd566` or set
  `engine.snapshot_skip_full_prompt_on_prefix_hit=false`.
- Superseded exact-hit detail (2026-07-31): I081/I082 extend the default skip
  to exact hits after three 4 GiB and four configured-8-GiB source-bound
  windows proved that lookup already owns the LRU touch, request clone, and
  pin. Exact hits no longer create a second full-prompt reservation/store by
  default; the same false setting still restores the historical refresh path.
- Next experiment: randomized sustained branch/cancel/recovery ordering, then
  model-native fixed-shape state isolation for batching.

## 2026-07-14: Make Core Manual Runtime the Active Workstream

- Decision: prioritize Aster's manual runtime foundation before integrating
  DFlash or any other speculative-decoding reference design.
- Scope: request lifecycle, prefill/decode scheduling, KV and prefix ownership,
  state isolation, memory pressure, cancellation, correctness, and benchmark
  infrastructure. `examples/dflash-*` remains read-only reference material.
- Baseline: prefix-off Qwen3.5-0.8B manual runtime at concurrency 4 measured
  `5.307s`, `31.16` average generation tok/s, and `1.626GB` peak MLX memory on
  a 4,820-token long workload, with zero swap growth and zero failures.
- Reason: speculative decoding multiplies cache, rollback, sampling, and
  correctness surfaces; the core runtime must first have stable ownership and
  measurement gates so any later draft/verify path can be judged honestly.
- Next experiment: profile the manual long/concurrent path and test one
  model-native state-isolation or scheduling change without touching DFlash.

## 2026-07-17: Reject 4-bit TurboQuant and Keep Native MLX Attention

- Decision: retain no TurboQuant or direct-paged runtime change. Native MLX
  remains the production attention path; direct paged attention stays opt-in.
- Direct-path evidence: ten fresh Qwen3.5-0.8B controls per context measured
  direct elapsed `+0.37%/+0.20%` at 2K/8K with exact tokens, only
  `9.12/6.09 MB` lower maximum peak, and zero swap growth.
- Kernel evidence: five fresh processes, 30 warmups, and 200 interleaved calls
  per method showed 4-bit TurboQuant `3.94x` cache compression and
  `18%~60%` lower latency than Aster paged. It did not beat default MLX across
  2K/8K/32K/64K; 2K/32K/64K were `3.47%/34.13%/25.08%` slower.
- Model evidence: twenty fresh Qwen3.5/WikiText-2 cells across five distinct
  windows found decode `5.22%/5.72%` slower at 2K/8K. Only 3/5 greedy windows
  matched at either context; minimum teacher top-1 was `89.06%/93.75%`, and
  absolute PPL change reached `7.49%/3.38%`.
  Hybrid cache bytes fell `1.72x/2.67x`, but quality and no-regression gates
  failed; swap stayed flat, while the strict RSS interval gate was
  inconclusive.
- Reference handling: OMLX `e3a4fe4` and pinned mlx-vlm passed `51/51` tests.
  Open-TQ/gemma4metal remains read-only evidence because its public test
  boundary is weaker and the host lacks the separate Metal Toolchain needed
  for its native build.
- Rollback: no runtime code was introduced. Remove only Iteration 049 evidence
  files if the experiment itself must be discarded.
- Next experiment: prove or falsify whether post-sample `_eval_cache()` does
  necessary work using exact hybrid-state RAW/WAW and 10,000-step stress.

## 2026-07-17: Amortize Decode Allocator-Cache Clearing by Generated Tokens

- Decision: retain lazy decode-cache provenance and clear MLX allocator cache
  after each 512 generated tokens. Batch 1/2/4 therefore clear after
  512/256/128 successful scheduler steps.
- Design: single decode relies on sampled scalar materialization; batch decode
  retains `mx.eval(logits)` but no longer evaluates every merged cache leaf.
  Prefill still evaluates cache state explicitly. Prefill and explicit runtime
  clears reset the decode token budget.
- Reference: MLX-LM main `15b522f` does not evaluate cache state per decode and
  periodically clears allocator cache. Official MLX 0.32.0 documentation
  confirms `array.item()` evaluates its graph.
- Evidence: 60 fixed-cadence confirmation processes measured `5.10%~15.13%`
  median gains across native/direct batch 1/2/4, all with exact token, text,
  and cache digests. Twenty token-budget batch confirmations measured
  `+11.94%/+15.27%` at batch 2/4. Eighteen final-source production processes
  retained `+9.51%~+17.90%` over the archived baseline.
- Stress: native/direct 10,000-token runs improved `5.58%/5.06%`, with RSS
  below the 1% regression gate and zero swap. A fixed 512-step batch-4 policy
  was rejected at `481.42 MB` transient allocator cache; the retained
  512-token policy reduced post-first-clear cache to at most `3.05 MB` while
  improving long batch-4 throughput `14.87%`.
- Correctness: native KV WAW, recurrent sibling-state RAW, and paged-pool WAW
  each completed 10,000 synthetic steps with exact final state bytes. The
  full archive contains 142 fresh process records and 16/16 artifact tests.
- Review hardening: a failed allocator clear retains its due budget and retries
  on the next generated token. Final integration approval is jointly gated by
  the hash-bound production bridge and token-budget long-stress aggregate.
- Tradeoff: cache leaves may remain lazy until their next RAW use; final or
  snapshot boundaries that do not consume logits must continue to evaluate
  state explicitly. The 512-token budget is global to one runner, matching
  allocator ownership rather than request ownership.
- Rollback: revert the Iteration 050 commit to restore per-step cache-tree
  evaluation and allocator clearing.
- Next experiment: profile the post-change general batch sampler path. Measure
  the remaining explicit logits evaluation and per-row synchronization before
  considering another sampling change; do not repeat the rejected greedy-only
  batch argmax experiment.

## 2026-07-18: Group Heterogeneous Batch Sample Synchronization

- Decision: retain one grouped sampled-token barrier in the production manual
  batch path. Preserve each row's existing logits processors and sampler
  instead of introducing a greedy-only or policy-specific fast path.
- Design: build row graphs in request order, submit MLX-like sampled scalars
  with one `mx.async_eval`, prepare cache references while work is queued, wait
  once with `mx.eval`, and materialize results in the same order. Python-valued
  custom sampler results bypass MLX evaluation and retain the old conversion
  contract, while logits provide the model/KV barrier. Errors after sampler
  graph submission are returned without replaying processors or RNG.
- Reference: MLX-LM `15b522f` uses grouped async token/logprob evaluation;
  vllm-metal tensorizes row policies; OMLX rebuilds row ownership around stable
  request IDs; LM Studio separates sampling from grouped materialization.
- Evidence: the selected candidate passed a 100-process greedy/mixed/penalty/
  structured confirmation and a 24-process 6,169-token confirmation. The
  review-hardened final admission measured `+9.89%~+18.06%` across the eight
  adopted short core cells and `+5.37%~+12.51%` across four long cells. Every
  selected exact independent-process interval clears its workload floor. The
  weakest long B2 cell's three 1,024-pair runs measured median `+5.37%` with
  process interval `[+4.51,+6.05]`.
- Stress: three mixed B8 runs generated 8,192 timed tokens per policy and
  improved a median `14.26%` with process interval `[+13.87,+14.58]`; all 96
  blocks were positive, each policy
  performed 16 expected allocator clears, token/text/cache hashes matched,
  and swap did not grow.
- Measurement decision: preserve the noisier 124-process production matrix.
  Desktop load caused isolated large positive and negative excursions, so
  final acceptance uses two independent KV states sharing one loaded model
  and balanced adjacent AB/BA calls. Block resampling is a within-process
  stability diagnostic; independent-process end-to-end results are the
  statistical admission unit. No record was deleted and the core 3% floor was
  not relaxed.
- Correctness hardening: stop-aware structured B4 produced valid schema output
  for 4/4 lanes while membership shrank `4 -> 3 -> 1`. The composite
  `final-admission.json` requires current source hashes, all selected paired
  components, stop-aware schema validity, zero swap, and retained negative
  evidence.
- Rejected alternatives: eager grouped sampling, sampled-scalar concatenation,
  and Iteration 034's whole-batch greedy argmax. Current SIMPLE,
  FlashSampling, fused-softmax, and backend-sampler implementations remain
  reference material because their hardware, batch, or correctness contracts
  do not match Aster's B1-B8 arbitrary-processor path.
- Tradeoff: the batch path creates one extra Python list of lazy sampled values
  and still must materialize host token IDs before stop/stream processing. The
  optimization does not claim to remove all sampling overhead.
- Rollback: revert the Iteration 051 commit to restore eager logits evaluation
  and per-row sample materialization.
- Next experiment: profile post-change logsumexp and processor-specific graph
  costs before testing tensorized homogeneous groups or backend sampler graphs.

## 2026-07-23: Re-audit Iteration 051 with strict assignment-balanced evidence

- The original Iteration 051 short/long claims used too few independent
  processes and are retained only as historical screening evidence. The
  current admission uses 18 fresh processes per cell, nine odd/even
  runner-assignment replicates, round-robin cell execution, and a
  96.09%-coverage distribution-free median interval.
- Short core cells cleared the 3% lower-bound gate in balanced, baseline-first,
  and production-first strata. Structured B2/B4 are a compatibility fallback:
  their balanced lower bounds were `-0.25%/-0.72%`, above the explicit `-1%`
  no-regression floor; host-driven parser order and block stability remain
  diagnostics rather than speed gates.
- The 6,169-token screen retained a greedy B2 production-first lower-bound miss
  at `+1.75%`. A predeclared 1,024-step, 18-process confirmation resolved it
  with balanced `[+6.65,+7.44]`, baseline-first `[+6.57,+7.64]`, and
  production-first `[+6.75,+7.65]`. The screen failure is recorded in
  `final-admission.json`, not removed.
- Mixed B8 1,024-step stress passed with balanced `[+14.52,+15.41]`, both
  order strata above `+14.5%`, exact parity, and no swap growth. System swap
  reclamation is recorded separately; the memory gate rejects growth rather
  than requiring an unrelated global counter to remain byte-identical.
- The current admission binds measurement-source/model hashes from raw payloads
  and recomputes every strict aggregate. Structured validation now hashes the
  JSON processor source as well as the runner, and confirms schema-valid,
  stop-aware membership shrink `4 -> 3 -> 1`.

## 2026-07-23: Reject greedy logsumexp elision after end-to-end profiling

- Hypothesis: MLX-LM's `temperature == 0` argmax does not need the per-row
  `logsumexp` subtraction, so a semantic sampler marker could reduce decode
  cost without changing tokens.
- Evidence: a benchmark-only candidate retained exact token/text/cache parity
  and zero swap growth in ten fresh paired records. Observed medians were
  `+0.53%/+0.58%/+0.50%` for greedy B2/B4/B8, `+0.60%` for penalty B4, and
  `+1.27%` for mixed B4. A same-logits profile measured only `13~53 us` graph
  deltas at B1/B2/B4/B8.
- Decision: do not modify `ModelRunner`, request state, or sampler contracts.
  The candidate is below the 3% core gate; retain the scripts and payloads as
  evidence for future fused processor/sampler work.
- Constraint: structured and nonzero-temperature rows must continue receiving
  normalized logprobs, and any future optimization must preserve RNG order,
  parser ownership, and exact stop behavior.

## 2026-07-23: Reject broader raw-logit and neutral-processor candidates alone

- Iteration 053 retained exact shift-invariant sampler behavior but measured
  only `+1.22%` at mixed B4 and `+0.05%` at mixed B8. Do not add a sampler
  metadata contract for that isolated opportunity.
- Iteration 054 measured neutral repetition-processor removal at
  `+1.47%/+1.57%/+1.50%` for greedy B2/B4/B8. It is below the standalone gate.
- These are retained negative experiments. The neutral sentinel is used only
  as a prerequisite of Iteration 055's broader processor-context ownership;
  it is not promoted as an independent performance claim.

## 2026-07-23: Bound built-in penalty processor history to 20 tokens

- Decision: retain explicit `0`, `20`, or full-history (`None`) processor
  context metadata across decode initialization, request state, work items, and
  runner defense. Omit MLX-LM's neutral repetition processor.
- Ownership: Aster assigns a 20-token bound only where it constructs MLX-LM's
  repetition/presence/frequency processors with that exact context. Structured,
  thinking, custom, and legacy work-item processors retain full history.
- Evidence: 18 fresh 24,601-token B2 processes formed nine runner-balanced
  replicates. The balanced 96.09%-coverage median interval was
  `[+4.75%,+6.72%]`; baseline-first and production-first lower bounds were
  `+4.33%/+4.10%`. All nine replicates were stable, all output matched exactly,
  and swap stayed flat.
- Short boundary: 18-process 409-token B2/B4 matrices did not clear the speed
  gate, but balanced/order lower bounds stayed above `-1%`. They are
  no-regression controls, not positive performance claims.
- Failed evidence: the earlier 64-step matrix failed order/stability gates and
  remains archived. Increasing the predeclared measurement window to 256 steps
  produced the admitted independent matrix; no records were discarded.
- Composite gate: `final-admission.json` verifies all 54 formal processes,
  source/model signature equality, long admission, short no-regression,
  exactness, swap non-growth, and retention of the failed matrix. All nine
  gates pass.
- Tradeoff: this is an active-penalty ultra-long-context optimization. It does
  not accelerate full-history structured/custom processors or prefill.
- Rollback: remove the context metadata and engine slice helper, restore full
  processor history on every work item, and pass the prior unbounded arguments
  to MLX-LM's processor factory.

## 2026-07-24: Reject residual host and homogeneous sampling candidates

- Host profile: sampled-token materialization plus `DecodeResult` construction
  occupied only `0.225%~0.294%` of grouped batch time at B2/B4/B8.
- Homogeneous penalties: a benchmark-only `[B,20]` gather/scatter path preserved
  duplicate-token semantics, row sampler order, and exact output, but measured
  only `+0.01%~+0.66%`.
- Batch normalization: one processor-free `[B,V]` reduction preserved exact
  top-p output and RNG order, but measured `-1.16%/+1.78%/+0.11%` at B2/B4/B8.
- Decision: retain all nine payloads as negative evidence and make no production
  change. The candidates fail the 3% early gate, so independent-process and
  dynamic-membership confirmation are not warranted.
- Next experiment: avoid Python -> MLX -> Python token-history conversion only
  for Aster-owned structured/thinking processors that declare a Python-token
  input contract; preserve all unknown/custom processor behavior.

## 2026-07-24: Reuse LMFE native allowed-token lists

- Decision: retain a local fast path in `JSONSchemaLogitsProcessor` that borrows
  LMFE's cached native `list[int]`, falls back to integer conversion for other
  containers, and scans the small EOS set for membership.
- Ownership: key-context and incomplete-JSON EOS filters always create new
  lists, so Aster never mutates LMFE's cache. ModelRunner, arbitrary custom
  processors, parser ordering, masks, samplers, and RNG behavior are unchanged.
- Evidence: short structured B4 and 24,601-token B2 each ran 18 fresh
  processes / nine runner-balanced replicates. Their balanced 96.09%-coverage
  intervals were `[+31.33%,+33.40%]` and `[+24.96%,+27.63%]`; every order
  stratum cleared 20%, all outputs matched exactly, and swap did not grow.
- Correctness: stop-aware B4 produced 4/4 schema-valid results, stopped every
  lane in 17-58 tokens, and shrank active membership `4 -> 3 -> 1`.
- Rejected scope: direct Python-token delivery did not independently clear 3%,
  and an identity mask cache hit 0/320 calls. Neither enters production.
- Rollback: restore unconditional integer-list conversion and the prior
  allowed-list-driven EOS scan in `_allowed_tokens`.
- Next experiment: find and gate a cheap, capacity-bounded LMFE semantic key
  before considering reuse of the remaining roughly 6 ms structured mask.

## 2026-07-24: Reuse consecutive structured masks

- Decision: retain one prior allowed-list snapshot and its immutable MLX mask
  inside each `JSONSchemaLogitsProcessor`. A cheap length/shape/probe
  fingerprint rejects obvious changes, and full list equality verifies every
  hit before reuse.
- Capacity: exactly one entry per request. Capacity-1 matched the capacity-8
  screen's 1,076/1,152 B4 hits, improved the screen from `+83.16%` to
  `+90.61%`, and retained less MLX memory. No global cache, configuration, or
  cross-request ownership is introduced.
- Correctness: snapshot the allowed list on every miss so in-place mutation
  cannot validate a stale mask; include logits shape in the fingerprint.
  Key-context/EOS filtering, parser order, custom processors, samplers, RNG,
  stop handling, and cache state remain unchanged.
- Evidence: the 18-process short B4 and 24,601-token B2 matrices passed with
  balanced intervals `[+114.79%,+119.26%]` and `[+64.81%,+68.91%]`. Throughput
  medians improved `+112.39%/+65.94%`; all 36 outputs matched, all nine
  replicates per cell were stable, both order strata cleared the gate, and
  swap did not grow.
- Memory: conservative dual-runner MLX peak deltas versus Iteration 057 were
  7,979,008 bytes at short B4 and 3,989,504 bytes at long B2, below the
  predeclared 16/8 MiB bounds.
- Rollback: remove the three request-local fields and the fingerprint,
  equality, snapshot, and assignment block in `_mask`; the prior construction
  path then runs on every row without a migration or configuration change.
- Next experiment: profile the retained path again. Pre-cache percentages are
  invalid after this speedup and must not be used to choose the next change.

## 2026-07-24: Bound iteration state and benchmark evidence

- Decision: use `CURRENT.json` as the only mutable iteration state and the
  short iteration protocol as the operating contract. One iteration owns one
  hypothesis, one primary metric, and at most one production candidate.
- Workspace gate: the read-only checker measures changed/staged/untracked
  paths, artifact files and bytes, cross-iteration evidence, mixed index state,
  reference updates, and generated caches. The initial report fails at 1,163
  changed paths and 1,114 artifact files; this is the debt baseline, not a
  reason to delete existing user work.
- Evidence policy: exploratory raw output stays under ignored
  `run/loop-engineering/<iteration>/`. Only source, manifests, aggregate or
  admission results, and the minimum raw records needed to recompute a formal
  claim are promoted. More than 150 files or 100 MiB requires an explicit
  retention justification and compact representation.
- Scientific gate: a 3% screen only authorizes TDD. Admission still requires
  independent processes, balanced AB/BA strata, predeclared intervals, exact
  behavior, memory and swap bounds, and a recorded full-suite result.
- Tradeoff: compact evidence makes review and recovery faster, while full
  exploratory payloads are local scratch rather than permanent repository
  history. Manifests and hashes preserve provenance; formal evidence needed to
  recompute an admitted claim remains tracked.

## 2026-07-24: Cap snapshot cloning for long-context requests

- Decision: before cloning a prefix snapshot, reserve space for both the live
  cache and its clone. For requests with at least 65,536 prompt tokens, cap the
  available-memory portion of the snapshot budget at 2 GiB; shorter requests
  retain the existing budget and the configured snapshot limit still applies.
- Basis: a controlled 128K comparison changed the snapshot budget from 8 GiB
  to 2 GiB. The request completed instead of failing, snapshot stores fell from
  8 to 5, active MLX memory fell from 8.47 GiB to 3.15 GiB, and swap did not
  grow. This is directional evidence; the automatic default-config path still
  needs a fresh archived real-model reproduction.
- Verification: threshold boundary, lower-memory behavior, clone preflight,
  and both checkpoint call paths are covered; the full suite passes with 498
  tests and nine skips.
- Rollback: remove the request-aware budget helper and pass only the configured
  snapshot and available-memory minimum to clone preflight.

## 2026-07-24: Reuse exact structured EOS membership

- Decision: retain one EOS-presence boolean beside each request-local exact mask
  snapshot. Reuse requires the same probes and full allowed-list equality;
  mutations and fingerprint collisions remain verified misses.
- Semantics: incomplete JSON still rechecks completion on every call. EOS/key
  filtering creates a new list, LMFE-owned lists are not mutated, and parser,
  mask, sampler, RNG, stop, and cache ordering remain unchanged.
- Evidence: 18 fresh processes / nine runner-balanced replicates per cell gave
  balanced intervals `[+38.01%,+47.33%]` at short B4 and
  `[+21.09%,+24.47%]` at 24,601-token B2. Both order strata cleared 10%, all 36
  outputs matched exactly, 9/9 replicates were stable, and swap stayed flat.
- Correctness: stop-aware B4 produced 4/4 schema-valid results and membership
  shrank `4 -> 3 -> 1`. Mutation, completion recheck, collision, EOS filter, and
  shape invalidation have automated coverage.
- Memory correction: numeric ceilings were absent before the first formal
  matrix, so its memory data remains discovery evidence. After recording 4/2
  GiB RSS and 16/8 MiB MLX-delta limits, four fresh confirmations passed.
- Evidence retention: 50 logical files are stored in one 237,686-byte archive;
  the composite admission recomputes both strict aggregates and passes 12/12
  gates.
- Rollback: remove the cached and pending EOS booleans and perform the original
  membership scan in `_allowed_tokens` on every call.
- Next experiment: measure which LMFE prefix-state `TokenList` objects remain
  live before changing their ownership or lifetime.

## 2026-07-24: Separate inherited workspace debt from iteration growth

- Decision: keep the initial 1,164 changed / 1,114 artifact paths as an
  immutable debt baseline at HEAD `2cb14052d4a3`. The checker reports that debt
  as a warning while blocking net growth beyond 25 files / 20 artifacts and
  enforcing a 20-file / 5 MiB active-iteration artifact budget.
- Reason: treating inherited shared-worktree inventory as fresh work made every
  future iteration permanently fail; deleting or staging owner-unknown work
  would be worse. Incremental gates keep new work bounded without hiding debt.
- Constraint: the baseline must never move upward to absorb new files. Lower it
  only after owner-attributed consolidation, and continue reporting foreign
  iterations, reference updates, mixed index paths, and generated caches.

## 2026-07-28: Reuse request-local LMFE freetext lists

- Decision: retain a request-local reusable LMFE freetext `TokenList` and one
  active sequential prefix state in `JSONSchemaLogitsProcessor`.
- Root cause: native LMFE retained a new copy of the same 246,881-token JSON
  freetext allowlist for every prefix state. Source-bound profiles observed
  1,024 short B4 and 256 long B2 lists even though the measured sequences were
  append-only.
- Design: the working list is keyed by LMFE's static allowlist identity, only
  its dynamic tail is cleared before the next state, and it is excluded from
  `allowed_token_cache`. Non-monotonic callers rebuild from the root parser.
- Evidence: 18 fresh processes / nine runner-balanced replicates per cell gave
  short/long balanced lower bounds of `+16.01%/+12.52%`; every output/cache
  hash matched and swap did not grow. Two independent ownership pairs reduced
  median RSS growth by `98.68%/97.61%`. Stop-aware B4 was 4/4 schema-valid and
  released every request list after lane cleanup.
- Rejected alternatives: aggressive predecessor pruning was about 3% slower,
  a 32-state window regressed short B4 by 22.9%, and Python composite-list
  screens regressed by 15% to 59%.
- Boundary: this is an LMFE 0.11.3 private-method adaptation, not a general
  structured-output performance claim. An LMFE upgrade, broader schema/tool
  workload, or non-monotonic production caller requires a fresh review.
- Rollback: restore native `TokenEnforcer` construction, remove the active
  prefix/decode-step helpers and their tests, and retain the prior direct
  `get_allowed_tokens` path. No configuration or data migration is needed.

## 2026-07-28: Admit a scenario-scoped local MLX-LM baseline

- Decision: retain the I061 Aster/direct-MLX-LM comparison protocol and compact
  evidence as a baseline, without changing production runtime behavior.
- Equivalence: both engines used the same local Qwen3.5-0.8B 4-bit files, raw
  prompt IDs, greedy sampler, warmup, fixed completion cap, and isolated
  process boundary. All 12 pairs matched completion IDs, text, and finish
  reason with non-growing swap.
- Result: Aster is 5.743% lower at paired p50 in the 128-word/256-token case
  and 10.431% higher in the 2,048-word/64-token case. The signs differ, so the
  evidence is a workload baseline rather than a general engine ranking.
- Retention: the 31,783-byte, 38-member archive recomputes the aggregate
  without scratch; its final admission and source hashes are covered by two
  artifact tests.
- Next decision: profile short-context per-token overhead before considering a
  production change; do not transfer the long-context result into a short
  decode claim.

## 2026-07-28: Reject short-decode host-allocation and pipeline screens

- Decision: retain no I062 production change. The engine-equivalent empty
  logits-processor context removes only 0.052%/0.246% in two exact paired
  records; the full-history allocation is below the 3% floor.
- Pipeline evidence: a six-process balanced same-model MLX-LM serial/pipeline
  screen retained exact 256-token output and zero swap, but its 9.013% p50 had
  a 95% interval `[-27.195%, +38.942%]`. Serial-first and pipeline-first
  medians reversed at `-27.123%` and `+30.879%`.
- Reason: the measured effect is inseparable from order/state interaction, so
  adopting a lookahead host/graph contract would be speculative.
- Retention: 13 raw records are compressed into a 5,996-byte archive and an
  archive-only test recomputes the inconclusive aggregate.
- Next decision: improve benchmark state classification before retrying any
  asynchronous decode design.

## 2026-07-28: Reject fixed terminal-prewarm explanation for short decode

- Decision: retain no I063 production change. Clearing MLX allocator cache and
  collecting Python objects preserved the serial-first/pipeline-first timing
  reversal, and a crossed prewarm screen ruled out the fixed `serial ->
  pipeline` terminal graph as a sufficient explanation.
- Evidence: eight independent 2x2 crossed-prewarm records had exact 256-token
  output, `length` finish, identical source/model fingerprints, and non-growing
  swap. Pipeline-first p50 was +16.746%, serial-first p50 was -14.022%, and all
  four matched pipeline-first-minus-serial-first contrasts were positive
  (+12.085% to +89.357%).
- Boundary: coarse host load/frequency/thread values were retained, but MLX did
  not expose stream counts in this environment. The result rules out a specific
  warmup explanation; it does not establish a pipeline speedup or a hardware
  root cause.
- Retention: 14 raw records are compressed into a 19,498-byte archive with
  source hashes and an archive-only recomputation test.
- Next decision: measure serial/serial and pipeline/pipeline fresh-cache pairs
  to classify call-position state before reconsidering asynchronous decode.

## 2026-07-28: Reject adjacent same-process short-decode comparison

- Decision: retain no I064 production change. In a crossed eight-process
  same-variant screen, both serial and pipeline were materially slower on their
  second fresh-cache call: p50 `-25.955%` and `-28.110%`; seven of eight second
  calls regressed.
- Interpretation: I062/I063's mixed serial/pipeline sign reversal is consistent
  with call position. This rejects the lookahead pipeline as a deployable
  performance conclusion, but does not identify the underlying allocator,
  stream, thermal, or hardware mechanism.
- Boundary: serial's bootstrap interval stayed below zero; pipeline's included
  one +4.869% record. The finding is therefore a robust protocol warning, not
  a precise universal slowdown estimate.
- Retention: eight raw records are compressed into a 23,097-byte archive with
  source hashes and an archive-only recomputation test.
- Next decision: verify that I061's process-isolated Aster/direct comparison
  contains one timed decode per process after its declared prewarm, and keep all
  same-process diagnostics out of engine-ranking statistics.

## 2026-07-28: Admit the I061 isolated timed-call boundary

- Decision: retain I061 as a scenario-scoped Aster/direct baseline. An I065 AST
  and archive audit found exactly two generation calls per engine process: one
  discarded warmup and the second assigned to the sole timed result.
- Evidence: all 24 archived record PIDs are unique; both scenarios retain 3/3
  Aster-first/MLX-LM-first process balance; all pairs remain comparable; the
  38-member archive and current I061 source hashes match its admission record.
- Boundary: this validates the retained I061 comparison protocol, not a general
  engine ranking and not a claim about same-process adjacent calls.
- Next decision: attribute Aster's isolated timed decode to a measured component
  before choosing another production runtime candidate.

## 2026-07-28: Require public data and complete cross-engine coverage

- Decision: replace the unstarted I066 Aster-only attribution plan with a
  version-pinned public-data and cross-engine completeness gate. A locally
  constructed prompt may remain a unit-test fixture or historical diagnostic,
  but it cannot select a production optimization or support a general engine
  comparison.
- Sources: the tracked lock pins FastChat MT-Bench at
  `587d5cfa1609a43d192cedb8441cac3c17db105d` and LongBench data at
  `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, including SHA-256, byte size,
  license notes, and structural checks. The downloaded LongBench archive is
  `cb45b11a...857f7f64`: 34 JSONL members / 8,418 rows, of which the primary
  corpus is 21 tasks / 4,750 rows; its official 21 prompt templates and output
  limits are separately pinned.
- Workload rule: MT-Bench uses its verbatim first turn. LongBench uses only its
  official templates and output limits. Manifests carry source-record identity
  and prompt hash, not copied prompts. `cross-engine-core` has 1,380 public
  records for scoped diagnosis; `full-public` has 4,830 records and is the only
  profile eligible for a complete engine statement within its named MT-Bench
  plus LongBench-primary scope.
- Comparability rule: `validate-results` rejects missing required engines or
  workload rows, different model/Tokenizer fingerprints, generation-setting or
  prompt drift, deterministic token drift, and missing TTFT/prefill/decode/
  end-to-end/RSS/swap metrics. The source lock, download manifest, inventory,
  and results stay under ignored `run/`; only their contract and conclusions are
  tracked.
- Current availability: the 12-runtime inventory finds Aster and direct MLX-LM
  available. Exo, Ollama, llama.cpp, vLLM, SGLang, vLLM-MLX, MLC-LLM,
  mistral.rs, LM Studio MLX Engine, and OMLX are unavailable by local probes and
  remain explicit exclusions rather than omitted comparison rows.
- Next decision: implement source-bound Aster and direct-MLX-LM result adapters,
  run independent-process public-data records, and inspect the complete matrix
  before changing production runtime code.

## 2026-07-28: Admit the public core-matrix adapter, defer attribution

- Result: I066 implemented source-bound Aster and direct-MLX-LM result adapters
  and completed all 1,380 `cross-engine-core` records on both engines (2,760
  engine-records). The source lock, source-rendered input token IDs, model and
  Tokenizer hashes, greedy contract, output token IDs, metrics, and zero-swap
  results all passed the validator's eight gates.
- Adapter correction: source-bound long prompts initially exposed deterministic
  output drift because direct MLX-LM used its default 2,048-token prefill step
  while Aster used 1,024. Pinning both adapters to 2,048 restored exact output
  token parity; this is a measurement-contract correction, not a production
  runtime optimization.
- Descriptive scoped screen: paired medians are Aster/direct +15.762% decode
  throughput, -8.509% prefill throughput, -5.112% TTFT, +0.975% end-to-end
  time, and -3.304% peak RSS, with zero swap. By input length, prefill is lower
  for Aster below 8,192 tokens and higher in the 8,192-32,768 bin; directions
  also vary by workload.
- Decision: admit the public adapter and comparable scoped matrix as I066's
  foundation. Reject any global engine ranking and reject a production bottleneck
  selection: each engine/task shard was measured only once, and the scope is not
  `full-public`.
- Next decision: I067 must rerun the complete same-source core matrix with each
  shard's engine order reversed, then require matched workload/length-bin
  direction and a bootstrap interval outside the 3% no-op band before profiling
  or changing a runtime component.

## 2026-07-28: SIMD and Gigatoken reference intake

- Source reviewed: `marcelroed/gigatoken` at
  `34a1599f0c0ae7d7cd0d1c530e6522320158b360`, MIT, version `0.10.0`. Its
  Rust implementation targets CPU tokenization through SIMD pretokenization,
  cache-aware BPE encoding, and a HuggingFace-compatible API; it is not a
  model-forward, MLX, or Metal decode kernel.
- Boundary: Aster currently depends on its model tokenizer for prompt/chat
  encoding, special-token fragments, and streaming detokenization. Replacing
  it blindly would risk token, template, truncation, and streaming drift, while
  an input-only improvement cannot be reported as GPU inference acceleration.
- Decision: retain Gigatoken as a remote P1 reference for a future opt-in CPU
  ingress experiment. Do not install it as a production dependency or change
  the default tokenizer path now. I067 remains limited to the reversed public
  cross-engine matrix.
- Admission gate for a later experiment: exact Qwen3.5 token IDs over public
  prompts plus chat/BOS/EOS/stop/thinking/structured/Unicode cases; original
  streaming detokenizer retained; queue-aware TTFT and end-to-end benefit with
  no decode, RSS, swap, cancellation, or output-token regression. CPU SIMD and
  Metal simdgroup candidates remain separate measured layers.

## 2026-07-28: Reject I067 public crossed matrix as a production attribution

- Decision: retain no I067 production change and reject a global Aster/direct
  MLX-LM ranking. The second, fully reversed 1,380-record public matrix added
  2,760 engine-records; both matrices pass all eight public comparability
  gates, the crossed join passes all nine gates, each engine is first for 1,380
  records, and swap remains flat.
- Evidence: aggregate Aster/direct prefill is lower in both strata: `-10.504%`
  with Aster first (95% bootstrap `[-10.951%, -9.868%]`) and `-7.681%` with
  MLX-LM first (`[-8.455%, -7.037%]`). But the `[8192,32769)` decode result
  reverses from `-10.515%` to `+27.532%`. Public QMSUM has material sign
  reversals: decode `-11.423%/+35.203%`, end-to-end
  `+10.308%/-36.111%`, and prefill `-6.087%/+81.979%`.
- Boundary: the matrix proves source/model/input/output comparability and an
  order-sensitive measurement state, not why the state differs. Aggregate
  prefill remains a candidate hypothesis only; it does not authorize prompt
  batching, cache, SIMD, scheduler, or native-kernel changes.
- Next decision: I068 repeats all 200 public QMSUM records in four independent
  ABBA-ordered blocks and records outside-timing process/host state. It can
  classify the measurement interaction but cannot itself admit a production
  candidate.

## 2026-07-29: Classify the public QMSUM gap as decode-bound

- Decision: retain no I068 production change, but replace I067's QMSUM
  measurement-state warning with a bounded, reproducible public result.
- Evidence: four fresh ABBA blocks completed all 200 locked QMSUM records on
  both engines (1,600 engine-records). Source, model/tokenizer, execution,
  deterministic cross-block output-token parity, metrics, state trace, ABBA
  balance, and zero-swap gates pass. Aster/direct decode throughput is
  `-7.775%` with Aster first (95% `[-8.042%, -7.682%]`) and `-8.216%` with
  MLX-LM first (`[-8.408%, -8.008%]`). End-to-end time is `+5.452%` / `+5.352%`.
  Both directions repeat in both blocks per order stratum.
- Boundary: prefill (`-1.982%` / `-1.514%`) and TTFT (`-1.457%` / `-1.928%`)
  stay inside the 3% no-op band; peak RSS is not reproducible. Outside-timing
  host/process telemetry is context only and does not attribute the decode
  deficit to a specific cache, model, sampling, or delivery operation.
- Next decision: I069 adds source-bound decode component instrumentation before
  selecting a runtime candidate. Do not enable prefill microbatching,
  Gigatoken, SIMD/Metal, paged KV, compressed KV, or speculation from I068.

## 2026-07-29: Rule out high-level B1 decode candidates before optimization

- Decision: retain no I069 production change. The public QMSUM decode-driver
  gap is reproducible, but the trace does not identify a semantically comparable
  implementation-level operation that justifies a runtime rewrite.
- Evidence: four fresh ABBA blocks completed all 200 locked QMSUM records on
  both engines (1,600 engine-records / 800 paired records). Component metadata,
  source/model/execution, deterministic output tokens, state trace, ABBA, and
  zero-swap gates pass. Common decode-driver seconds per output token are
  `+8.791%` with Aster first (95% `[+8.660%, +8.964%]`) and `+8.655%` with
  MLX-LM first (`[+8.522%, +8.773%]`). Aggregate decode throughput remains
  `-8.177%` / `-8.025%`; end-to-end is `+5.906%` / `+5.504%`.
- Boundary: the single-request matrix reports zero Aster batch-cache merges and
  rebuilds. Cache resolution (`0.004%`), processor dispatch (`0.004%`), and
  result delivery (`0.082%`) are immaterial shares of Aster's driver. The
  `92.550%` sampling-completion field contains the lazy MLX completion barrier,
  so comparing it to an absent direct-MLX-LM private substep would be invalid.
- Next decision: I070 instruments only source-aligned submit and mandatory
  materialization boundaries, first through a public traced/untraced smoke and
  then through QMSUM ABBA if the smoke is exact and within the no-op band. Do
  not enable prefill microbatching, Gigatoken, SIMD/Metal, paged KV, compressed
  KV, speculation, or a native backend from I069.

## 2026-07-29: Reject the lower-level decode observer as a QMSUM attribution tool

- Decision: retain no I070 production change and do not run the planned QMSUM
  ABBA trace. The source-bound observer is not measurement-neutral on the
  locked public workload.
- Evidence: V2 ran four fresh isolated MT-Bench shards in opposite first-status
  order, covering 80 records per engine/condition. Workload/source lock,
  model/tokenizer, generation, execution outside the observer, exact
  token/text/finish parity, complete trace coverage, and zero-swap gates pass.
  Median traced/untraced movement is Aster decode `-3.635%`, end-to-end
  `+3.818%`, TTFT `+7.432%`, RSS `-7.558%`; direct MLX-LM decode `-7.277%`,
  end-to-end `+7.717%`, TTFT `+11.477%`, RSS `-13.299%`. The formal result is
  `trace-no-op-rejected-metric-movement` in I070's 7,519-byte artifact.
- Tooling correction: source fingerprints are now compared within each
  traced/untraced engine pair. This retains the common harness fingerprint while
  permitting direct MLX-LM's engine-specific installed-package fingerprint; a
  focused regression rejects a drift in that engine-local source.
- Boundary: source-call proxies preserve outputs but add material Python work to
  the timed path. A direct MLX-LM private closure cannot be split without
  modifying or duplicating reference implementation code, so its lower-level
  label is not a valid production-performance evidence boundary.
- Next decision: I071 establishes a public-source arrival/load baseline for
  Aster's actual scheduler with controlled concurrency, staggered long-prefill,
  shared-prefix, and cancellation cases. It selects a future candidate from
  measured queue, TTFT, end-to-end, memory, and lifecycle behavior rather than
  from a private decode label.

## 2026-07-29: Keep decode-aware prefill cap disabled pending resource attribution

- Decision: retain `engine.decode_active_prefill_token_budget` only as a
  nullable opt-in experiment. `configs/config.yaml` stays on its null default;
  no normal serving path changes.
- Evidence: four independently started, order-balanced locked-source staggered
  QMSUM/MT-Bench pairs preserved exact token/text/finish parity. At value 512,
  paired median short decode and end-to-end time improved `55.204%` and
  `48.276%`; paired long end-to-end and prefill time also improved `9.320%`
  and `7.986%`. Candidate cancellation accepted at a prefill checkpoint and
  cleaned up active state with zero swap growth.
- Boundary: process-level swap is non-monotonic. Candidate samples include
  `+1,043,791,872` bytes while controls peak at `+585,236,480` bytes. This
  means latency and deterministic correctness are insufficient for a default
  admission. A follow-up must attribute model/process lifecycle memory before
  rerunning the candidate under a resource gate.

## 2026-07-29: Admit decode-aware prefill cap after lifecycle attribution

- Decision: set engine.decode_active_prefill_token_budget to 512 in the
  production configuration. The runtime only applies the cap when decode work
  is queued, and null remains the direct rollback value.
- Evidence: the predeclared locked-source lifecycle screen ran four fresh
  processes in cache-on/cache-off and candidate/control order. All long and
  short outputs retained exact token IDs, text hashes, and length finishes.
  Candidate swap was zero for both cache states except for one -8,388,608-byte
  cache-off workload delta; controls were zero. Cache-on retained one
  390,103,040-byte snapshot in both variants without swap growth.
- Interpretation: the earlier +1,043,791,872-byte candidate sample did not
  recur as a candidate-only workload or lifecycle effect. This clears the
  scheduler-policy resource gate, while leaving global OS compression and
  prefix-cache lifetime attribution explicitly open.
- Compatibility: a fresh current-source Aster/direct-MLX-LM two-record 9B
  smoke matched public source lock, model/tokenizer fingerprint, greedy
  generation, common harness sources, token IDs, text hashes, finishes, and
  zero swap. It is output-compatibility evidence, not a heterogeneous timing
  claim or a replacement for I066/I067 complete public matrices.
- Next decision: I073 measures prefix-cache resource ownership before selecting
  a cache-size, eviction, or representation change.

## 2026-07-29: Reject I073 cache-policy selection

- Decision: retain no I073 cache-policy change. The arrival/load harness now
  supports a distinct locked-QMSUM plan and temporary experiment-only snapshot
  capacity overrides, but production snapshot budget, entry limit, eviction,
  checkpoint behavior, and representation remain unchanged.
- Evidence: six fresh source-bound rows retained exact terminal output identity
  across cache states and zero running/waiting/pending state. Cache-on shared
  prefix produced one exact hit and reused 10,333 tokens; the one-entry
  distinct-QMSUM row produced one existing 390,103,040-byte eviction; cache-on
  cancellation retained one 85,065,728-byte checkpoint and a deterministic
  follow-up. The compact artifact binds all raw rows by SHA-256.
- Boundary: workload-stage `psutil.swap_memory().used` changed
  +883,752,960 bytes cache-off and +364,576,768 bytes cache-on for shared
  prefix, while four distinct/cancellation rows were zero. This is a
  host-global meter and one row per state, so it is neither a process-owned nor
  repeatable cache-specific resource direction.
- Next decision: I074 adds a no-request lifecycle control and two
  order-balanced shared-prefix pairs. A cache policy remains blocked until a
  direction repeats and differs from the idle control.

## 2026-07-29: Reject I074 host-state cache attribution

- Decision: retain no I074 cache-policy change. `idle-lifecycle` is retained
  as a harness control, but no snapshot budget, entry limit, eviction policy,
  or representation default changes.
- Evidence: the empty-plan result metadata bug was corrected with a focused
  regression; nine arrival/load tests pass. Both idle cache states submitted no
  requests, retained no snapshots, and had zero workload-stage global swap.
  Two opposite-order shared-prefix pairs retained exact output identity and
  zero active state; both cache-on rows had one exact reuse of 10,333 tokens.
- Boundary: workload-stage global swap in `off,on,on,off` order was
  `0, 0, 0, +78,577,664` bytes. The only positive value was a cache-off row;
  it does not repeat by cache state or distinguish itself from idle in both
  orders. The host-global meter remains context, not cache ownership.
- Next decision: I075 measures explicit snapshot bytes and exact replay utility
  for three distinct locked QMSUM keys plus a first-record replay at 8 GiB and
  a temporary 1 GiB budget. A lower default remains blocked until it preserves
  useful replay under a predeclared tradeoff.

## 2026-07-29: Reject I075 1 GiB snapshot budget

- Decision: retain the configured 8 GiB snapshot-budget default. The temporary
  1 GiB value is rejected and has no production effect.
- Evidence: both source-bound capacity-replay rows retained exact output
  identity and zero active state. The 8 GiB control retained three snapshots
  totaling 1,415,217,152 bytes and replayed the first record as an exact hit
  with zero prefill steps and `0.279007s` TTFT. The 1 GiB candidate retained
  390,103,040 bytes after two evictions, but replay missed, needed eight
  prefill steps, and had `19.219282s` TTFT.
- Boundary: the candidate's 1,025,114,112-byte explicit retention reduction is
  real, but it loses the predeclared useful replay. Its differing host-global
  swap result is not used for the decision.
- Next decision: I076 deepens the source-bound sequence to four distinct QMSUM
  records and tests a temporary 2 GiB budget. Any replay loss remains a
  rejection; no cache default is changed before that evidence exists.

## 2026-07-29: Reject I076 2 GiB snapshot budget

- Decision: retain the configured 8 GiB snapshot-budget default. The temporary
  2 GiB candidate is rejected and has no production effect.
- Evidence: all five locked-source outputs retained exact terminal identity and
  zero active state. The 8 GiB control retained four snapshots (1,988,067,328
  bytes) and replayed the first record as a zero-prefill-step hit at `0.226819s`
  TTFT. The 2 GiB candidate finished under its final budget (1,591,541,760
  bytes), yet performed two evictions and replayed as an eight-step miss at
  `27.859819s`.
- Boundary: existing source shows clone-reserve behavior: it reserves twice the
  candidate snapshot bytes and evicts below the resulting target before clone.
  Final counters lack the per-reservation target and eviction context, so they
  cannot select a wider budget safely.
- Next decision: I077 adds bounded prompt-free per-reservation telemetry and
  first proves that observer is neutral on source-bound replay. It changes no
  clone-reserve or eviction behavior.

## 2026-07-31: Admit I077 bounded reservation observability

- Decision: retain a default-64, maximum-256 FIFO of immutable snapshot
  reservation events in engine status and cache statistics. Value 0 disables
  collection. No clone-reserve, eviction, cache budget, entry limit, or
  snapshot representation changes.
- Evidence: accepted, reserve-eviction, preflight-skip, disabled, and FIFO-drop
  behavior has focused coverage. A separate-process traced/untraced 8 GiB
  four-key replay retained exact terminal identities, zero active state, four
  snapshots / 1,988,067,328 bytes, and zero evictions. Exact replay stayed at
  zero prefill steps and TTFT moved `0.169433 -> 0.166854s` (`-1.522%`), inside
  the absolute 3% observer gate.
- Privacy/size boundary: the five captured events include request ID, logical
  length, byte budgets/reserve/targets, store state, and eviction deltas. They
  contain no prompt, text, prompt-token array, or token IDs and dropped none.
- Reference basis: current upstream vLLM carries structured KV eviction samples
  in scheduler statistics; current SGLang returns a structured per-call
  `EvictResult`. Aster adopts only that observability boundary around its
  existing policy.
- Next decision: I078 tests the trace-predicted 3 GiB four-key boundary without
  changing the production default.

## 2026-07-31: Admit I078 3 GiB only for wider retention testing

- Decision: the temporary 3 GiB value becomes a candidate for a wider
  reuse-distance matrix, not a production default. The tracked 8 GiB default
  and existing cache policy remain unchanged.
- Evidence: all five terminal identities match I077. Every effective budget is
  exactly 3 GiB and every store-before/store-after value is below the recorded
  target. The process retains four snapshots / 1,988,067,328 bytes with zero
  reservation/store evictions, zero preflight skips, zero active state, and an
  exact zero-prefill replay at `0.165729s` TTFT.
- Boundary: four ordered QMSUM keys do not represent a sustained agent reuse
  window. Timing, RSS, and host-global swap are context only in this
  retention-capacity decision.
- Next decision: I079 establishes a six-distinct-key 8 GiB traced replay
  baseline before selecting at most one lower candidate.

## 2026-07-31: Admit I079 4 GiB only to a balanced multi-window screen

- Decision: retain the configured 8 GiB production default. The temporary
  4 GiB value advances to I080 as an experiment-only candidate; 3 GiB is
  excluded by the six-key trace.
- Selection evidence: the 8 GiB control retained six snapshots / 2,846,359,552
  bytes and exact first replay. Its maximum observed store-before plus
  two-clone reserve was 3,626,565,632 bytes. That exceeds 3 GiB but leaves
  668,401,664 bytes of headroom under 4 GiB.
- Candidate evidence: the fresh 4 GiB row matched all seven control workload
  IDs, completion counts, finishes, output-token hashes, and text hashes. It
  retained the same six snapshots / 2,846,359,552 bytes, had zero reservation
  or store evictions, zero preflight skips, zero active state, and exact
  zero-prefill replay at 0.167437 seconds.
- Boundary: this is one ordered public chain and one process per budget.
  Replay timing, unmatched RSS baselines, and host-global swap are context only;
  no performance, resource, sustained-session, or cross-engine claim is made.
- Next decision: I080 uses four new disjoint six-key QMSUM windows and balanced
  8/4 versus 4/8 process order. It rejects 4 GiB on any utility or correctness
  failure and does not directly change the production default.

## 2026-07-31: Reject I080 4 GiB snapshot budget

- Decision: retain the configured 8 GiB production default. The temporary
  4 GiB candidate is rejected after passing only one of four fresh disjoint
  six-key windows. No cache-policy or runtime behavior changes in I080.
- Evidence: all eight fresh processes match their paired source, plan,
  execution contract except budget, and all seven terminal token/text
  identities. Every control retains six snapshots with zero eviction and exact
  zero-prefill replay. Candidate windows 6, 18, and 24 each evict one other
  snapshot during the final replay reservation and end with five entries;
  window 12 retains six. Total candidate eviction is three entries /
  2,110,849,024 bytes.
- Interpretation: replay correctness remains exact because lookup hits before
  the eviction. Current source then schedules another full-prefix checkpoint
  for the exact hit, reserves duplicate clone capacity, and may evict an
  unrelated unpinned entry before replacing the same logical key. This is the
  next measured lifecycle hypothesis, not authorization to weaken capacity or
  eviction gates.
- Boundary: timing, RSS, MLX peak, and host-global swap are context only. The
  result supports no memory-saving, performance, sustained-session, or
  cross-engine claim.
- Next decision: I081 uses TDD to test suppression of duplicate full-prefix
  checkpoint work after an exact hit while preserving LRU touch, pin/unpin,
  miss, strict-prefix append, cancellation, and persistence behavior. Even a
  passing 4 GiB replay screen cannot lower the production default in I081.

## 2026-07-31: Admit I081 exact-hit lifecycle to 8 GiB validation

- Decision: retain the configured 8 GiB snapshot budget and advance the
  exact-hit checkpoint predicate to a fresh production-budget validation. The
  I081 runtime/test change remains an uncommitted candidate; it is not yet a
  production release.
- Evidence: three predeclared exact-hit tests failed before the change while
  the explicit refresh rollback control passed. After the one-line predicate
  change, four focused tests, the affected 97-test cache/engine/config/
  persistence/arrival suite, and the full suite (`554 passed, 9 skipped, 1
  warning`) passed. Default exact hits retain the lookup LRU touch and pin
  ownership, perform no second reservation/clone/store, and unpin at cleanup.
- Retention screen: fresh 4 GiB offsets 6, 18, and 24 match I080 source, plan,
  execution, and seven-request terminal identities. Each retains six entries
  with six stores, one exact hit, zero evictions/preflight skips, zero replay
  prefill, six bounded trace events, zero dropped events, and zero terminal
  active/pinned state.
- Compatibility boundary: a fresh two-record 9B Aster/direct-MLX-LM smoke
  matches source/model/generation, token IDs, text hashes, length finishes,
  and zero swap. It is output-compatibility evidence only, not a timing
  ranking or a replacement for the complete public matrices.
- Next decision: I082 runs fresh order-balanced six-key windows at the
  configured 8 GiB budget and widens cancellation/persistence controls. A
  failed gate restores exact-hit refresh behavior; a pass may authorize a
  small production commit without changing cache budget or eviction policy.

## 2026-07-31: Admit I082 exact-hit lifecycle for production commit

- Decision: admit the I081 exact-hit predicate for a minimal production
  commit. Keep the configured 8 GiB budget, two-clone reservation, eviction
  policy, persistence behavior, and snapshot representation unchanged.
- Evidence: four fresh configured-8-GiB windows at offsets 6, 12, 18, and 24
  match their I080 source/plan/execution/terminal controls. Every row retains
  six entries with six stores and one exact hit, zero evictions/preflight
  skips, exact zero-prefill replay, six bounded prompt-free trace events, and
  zero active/pending/pinned state. A fresh real-model cancellation accepts
  the cancellation, completes its deterministic follow-up, and ends cleanly;
  four persistence/cancellation tests pass.
- Boundary: this is a correctness and retention-lifecycle admission only. It
  makes no throughput or memory-saving claim and does not alter the complete
  public cross-engine matrices. The source change remains uncommitted pending
  the I083 packaging review.
- Next decision: I083 performs the minimal diff review and a short repeated
  exact/strict-prefix/cancellation loop. Commit/push remains a separate
  explicitly requested action.

## 2026-07-31: Complete I083 lifecycle packaging without a commit

- Decision: retain the admitted exact-hit predicate and its explicit rollback
  switch. Complete I083 without changing the 8 GiB budget, eviction policy,
  snapshot representation, or repository history.
- Exact evidence: one cold 10,334-token request plus eight serial exact replays
  keeps stores/entries/trace events at one, advances exact hits `0..8`, uses
  zero replay prefill steps, and ends every request with zero active/pending/
  pinned state. All output token/text/finish identities match and swap delta is
  zero.
- Strict/cancel evidence: two 10,342-token derived requests each reuse the
  10,334-token base and need one prefill step; their outputs match while store,
  entry, trace, and pin counts stay bounded. A fresh cancellation matches I082
  follow-up identity and terminal cleanup.
- Compatibility: the new optional plan suffix is omitted from legacy payloads;
  I080 window 6 remains byte-identical. Fresh two-record Aster/direct-MLX-LM
  processes match source/model/generation/prompt/output tokens/text/finish and
  zero swap. Full verification is `559 passed, 9 skipped, 1 warning`.
- Reference decision: do not replace Aster's already-indexed bounded lookup
  with a Python token trie without a measured bottleneck. Current SGLang/vLLM
  shared-cache reference counting instead motivates I084's concurrent exact
  fanout ownership screen.
- Next decision: I084 measures fresh rotated B2/B4/B8 fanout before selecting
  or rejecting a shared-block/COW ownership candidate. No production change is
  pre-authorized.

## 2026-07-31: Advance exact-hit shared-state feasibility after I084

- Decision: advance a type-specific exact-hit shared-state or copy-on-write
  feasibility experiment. Do not change the production `copy.deepcopy` path,
  configured 8 GiB budget, reservation/eviction policy, persistence format, or
  rollback switch in I084.
- Observer gate: B2 untraced/sampled/sampled/untraced processes preserved exact
  output and zero swap. Sampled elapsed, replay TTFT, replay latency, and replay
  throughput median deltas were `+0.855%`, `-2.854%`, `-0.919%`, and `-0.183%`,
  each inside the absolute 3% no-op band.
- Fanout evidence: nine fresh Qwen3.5-9B processes in rotated B2/B4/B8 order
  passed all source, plan, output, cache, trace, and terminal-cleanup gates.
  Peak active estimates were exactly one/three/seven times `390,397,952` bytes;
  only one `390,103,040`-byte store entry remained. B8 raised reported MLX peak
  from 8.258 to 10.588 GB and pooled replay latency from B2's 0.464s median /
  0.473s p95 to 3.913s / 6.512s.
- Boundary: host-global B8 swap deltas are pressure context, not standalone
  ownership attribution. I084 supports a design experiment, not a physical-byte
  claim for every cache layer and not a cross-engine ranking.
- Next decision: I085 inventories concrete Qwen3.5 cache mutation semantics and
  requires base/sibling isolation before an opt-in shared-state implementation.
  Unknown or in-place-mutating layers must retain eager cloning.

## 2026-08-01: Reject I085 typed cache fork; attribute B8 growth to merge

- Decision: reject a type-specific replacement for production
  `copy.deepcopy`. Keep the configured 8 GiB budget, reservation and eviction
  policy, persistence representation, rollback switch, and native batch path
  unchanged.
- Mutation evidence: Qwen3.5-9B has 24 `ArraysCache` and 8 `KVCache` layers.
  Seven focused tests prove prefill admission, base/sibling, append, trim,
  merge/extract, first-write, unknown-type rejection, and retained-artifact
  contracts. MLX array assignment
  gives distinct deep-copied descriptors copy-on-write value isolation.
- Physical evidence: three left-rotated B2/B4/B8 repetitions on the locked
  10,334-token public QMSUM record report zero clone-construction active growth.
  Native merge grows active memory by 780,402,816 / 1,560,543,488 /
  3,120,824,832 bytes and releases exactly to baseline. The merged state is
  86.80% full-attention `BatchKVCache` and 13.20% linear-attention
  `ArraysCache`.
- Interpretation: I084's request estimate is valid as conservative logical
  ownership, but not as clone-time physical allocation. A typed wrapper cannot
  clear the 25% B8 gate because the existing clone already allocates zero
  physical bytes. Running a production A/B for that rejected mechanism would
  add risk without a plausible effect.
- Reference boundary: vLLM avoids prefix row copies by letting attention consume
  refcounted block tables. SGLang's local MLX prefix pool still materializes
  per-request contiguous K/V before batched SDPA. I086 tests a bounded
  full-attention shared-prefix consumption interface; all auxiliary state stays
  request-owned and native fallback remains mandatory.

## 2026-08-01: Reject I086 shared-pool SIMD kernel on latency

- Decision: stop before a locked 9B model-runner A/B and do not route the
  benchmark-only shared-pool kernel into production. Keep native batch merge,
  eager cloning, the configured 8 GiB budget, reservation/eviction policy,
  persistence schema, rollback switch, and all defaults unchanged.
- Correctness/lifetime evidence: B2/B4/B8 plus unequal-length attention matches
  row-wise native MLX within 6.10e-05. The candidate never invokes
  `materialize()` or native full-prefix merge, preserves partial-block CoW, and
  releases every table and pool reference. The affected suite passes 32/32.
- Memory evidence: locked 10,334/B8 metadata is 5,216 bytes over one shared
  pool, versus 338,624,512 dense bytes for one native layer. With all linear
  state left unchanged, conservative eight-layer extrapolation reduces total
  merge growth 86.7941% and full-attention construction 99.9985%, clearing
  both memory gates.
- Latency evidence: five fresh processes with 30 warmups and 200 interleaved
  samples per method give median-of-process p95 8.944 ms native versus 9.671 ms
  candidate (1.177x). Every process exceeds the 1.03 ceiling; process ratios
  span 1.078x to 1.394x. The hard latency gate fails despite the memory win.
- Interpretation: two-dimensional block tables are the correct ownership
  shape, but this `Hq=16/Hkv=4/D=256` SIMD-group kernel is not the correct
  execution shape. Retain it only as experimental benchmark scaffolding. A
  future proposal must use a genuinely different operator or a measured
  prefix-homogeneous scheduling policy, not launch-metadata micro-tuning.

## 2026-08-01: Advance active-cohort frontier after I087

- Decision: admit `max_active_requests=4` as a measured exact-prefix scheduling
  candidate, but do not change the global default or production scheduler.
  Advance an active-width frontier before selecting a conditional rule.
- Corrected variable: I084 B8 already had `max_decode_batch=4`; its 7 live
  exact replays, not a seven-row decode call, owned the pressure. I087 keeps the
  byte-identical B8 arrival plan and decode width while comparing configured
  active limits 16 and 4 in five rotated fresh-process pairs.
- Evidence: all 80 matrix requests pass output/cache/cleanup contracts. The
  candidate reduces peak MLX from 10.588 to 8.551 GB (-2.037 GB / -19.237%)
  while paired median throughput, p95 TTFT, and p95/max latency ratios are
  1.551x, 0.890x, and 0.645x. Both order strata pass. A post-pass cancellation
  accepts the target, completes the follow-up, and ends clean with zero swap.
- Boundary: this is one long exact-prefix QMSUM cell. It does not prove cap 4
  is optimal or safe for short, distinct-prefix, or mixed-prefix traffic, and
  Feather's heterogeneous-prefix locality gains are not assumed for Aster's
  native contiguous MLX merge.
- Next decision: I088 sweeps caps 2/3/4/5/6 and adds short plus distinct/mixed
  guards. Conditional engine logic is considered only after that frontier;
  I086's rejected shared-pool kernel remains outside production.

## 2026-08-01: Reject a lower active-cap policy after I088

- Decision: do not change the configured `max_active_requests`, production
  scheduler, cache ownership, or greedy sampler. The 18-cell frontier has no
  lower cap that clears exact-long, short-simultaneous, and mixed performance
  gates, and mixed caps 2/5 also fail the exact output gate.
- Performance evidence: all 144 requests pass their per-cell contracts.
  Exact-long caps 2/3/4 are eligible and cap 3 reaches 1.595x cap-16 throughput
  with 22.009% less peak MLX. Short traffic admits no lower cap because every
  candidate misses throughput or tail/TTFT gates. Mixed performance alone
  admits caps 2-6, but that cannot override the global or correctness gates.
- Correctness evidence: only `mixed-short-3` diverges. Fresh caps 2/5 share six
  output tokens before selecting token 364 in a single-row decode; caps 3/16
  select token 421 in a two-row decode. Candidate logits at that step are
  equal or separated by only 0.125, and both stable output groups reproduce.
- Boundary: the diagnostic establishes batch-shape-sensitive near-tie behavior,
  not whether Aster cache state or model-native BF16 batch arithmetic owns it.
  Its forced logit evaluation invalidates timing. Epsilon tie-breaking would
  change ordinary argmax semantics and is not admitted from one prompt.
- Next decision: I089 runs a same-cache single/batch control and the closest
  direct/model-native MLX-LM comparison. A determinism proposal requires a
  classified owner plus exact output/quality and serving-performance gates.

## 2026-08-20: Defer MTP until foundation parity

- Decision: record Multi-Token Prediction as a later research track, not an
  I089 candidate or production dependency. I089 remains the only active
  hypothesis and must first classify the current greedy batch-shape near tie.
- Reference evidence: llama.cpp `0a50d990` couples its next-n heads to a
  dedicated MTP context, target hidden-state transfer, target sampler
  verification, checkpoints, partial rollback, and per-position acceptance
  telemetry. vLLM-MLX `0dd11576` and OMLX `d0ee0e85` additionally demonstrate
  recurrent-state restore, membership reconciliation, stochastic acceptance,
  bypass paths, and batch/load-dependent value.
- Entry boundary: first close deterministic cache/decode ownership and
  source-bound Aster/direct-MLX-LM parity for prefill, decode, TTFT, throughput,
  tail latency, and memory. Then prove KV/recurrent rollback and complete
  sampler, logits-processor, stop, streaming, cancellation, and finish behavior.
- Admission boundary: a future candidate must use compatible model-side heads
  and clear independent-process B1/B4/B8 mixed-load acceptance, TTFT, TPOT,
  throughput, memory, and swap gates with exact semantics and a material
  effect. High acceptance alone is not an engine result.

## 2026-08-20: Classify I088 drift as reference-shared cohort arithmetic

- Decision: close I089 without a production change. Keep native cache merge,
  current scheduler/defaults, BF16 execution, and ordinary greedy argmax.
- Evidence: four independent balanced-order 9B processes pass all 22 gates.
  Aster and native MLX-LM paired-history target caches are byte-identical at
  `3d1f3322...c14bf`; serial state is `491b3b82...40c4b`. Merge/extract always
  matches direct single-row execution and all canonical caches stay immutable.
- Attribution: serial state selects 364 across single, duplicate, and native
  controls. The paired-history cache selects 8574 as one row, 364 when copied
  into two identical rows, and 421 with the original 57-token companion. Aster
  and MLX-LM agree at every boundary and the original cohort reproduces I088.
- Interpretation: the output drift is reference-shared BF16 batched-history and
  cohort-shape arithmetic on a three-token near tie, not Aster cache corruption.
  Cross-shape greedy identity cannot select a scheduler policy by itself.
- Rejected changes: epsilon tie-breaking, forced single-row routing, and a
  prompt-specific precision exception alter declared semantics or serving
  performance without broad quality evidence. I090 instead establishes current
  9B foundation parity before choosing another Aster-owned optimization.

## 2026-08-20: Establish the I090 foundation-parity performance baseline

- Decision: close the foundation screen without a production change and select
  `aster-manual-decode-driver` as the bounded I091 attribution profile.
- Baseline: direct/model-native MLX-LM on the locked Qwen3.5-9B public cohorts;
  Aster is the measured path. The matrix has 32 rows (four cells, two engines,
  four independent repetitions) with balanced first-engine order.
- Performance ledger: paired median Aster deficits for aggregate throughput,
  decode-driver TPS, TTFT p95, end-to-end p95, and peak MLX memory are
  respectively `+11.504%/+110.030%/-31.323%`,
  `-8.132%/+36.322%/+24.484%`,
  `+21.306%/+183.324%/-31.674%`,
  `+11.504%/+110.023%/-31.323%`, and
  `+1.049%/-0.921%/-32.763%` for B1-short/B4-short/B4-mixed. B1-long
  decode is `+5.978%` pooled but reverses by order (`+12.774%/-2.731%`) and
  is not selected.
- Correctness boundary: source/input/order/terminal/resource contracts pass;
  B4-short `short-3` and B4-mixed `short-0` have declared cross-engine output
  divergences, so this is a workload-scoped baseline, not an engine ranking.
- Next gate: I091 must remove forced-evaluation timing, align the B4 output
  contract, and report a fresh valid baseline/candidate delta before any
  runtime change. MTP remains foundation-gated.

## 2026-08-21: Reject tensorized decode normalization in I091

- Decision: reject `engine.decode_tensorized_logprobs_enabled` as a production
  optimization. Keep it default-off as an explicitly scoped experiment with a
  one-field rollback.
- Evidence: the balanced fresh-process ledger has 16 Aster rows (two
  baseline-first and two candidate-first repetitions per B4 cell), exact output
  and terminal identity, zero decode fallbacks, and candidate diagnostics that
  exercise 9/9 short and 8/8 mixed batch steps. B4-short decode changes
  `54.105929 -> 53.248829 tok/s` (`-1.584%`); B4-mixed changes
  `33.850635 -> 33.859096 tok/s` (`+0.025%`). Neither reaches the 3% gate.
- Contrary evidence: short order strata are `-1.496%/-0.685%`; mixed strata
  are `-0.610%/+0.157%`. Mixed aggregate throughput falls `0.898%`, TTFT p95
  and end-to-end p95 rise `1.259%/1.235%`, peak MLX rises `2.278%`, and one
  candidate row grows host swap by `317,587,456` bytes.
- Boundary: the random-logit micro-screen is numerically exact and `11.589%`
  faster, but it is not an end-to-end or lazy-graph result. The initial
  fixed-order real-model screen is superseded by the balanced recollection and
  is not used for admission.
- Next gate: I092 performs benchmark-only roofline/stage attribution inspired
  by LLMVisor. MTP and speculation remain behind foundation parity and full
  KV/recurrent rollback, sampler, stop, streaming, cancellation, and mixed-load
  gates.

## 2026-08-21: Reject per-step decode-stage observer in I092

- Decision: close I092 as a valid attribution measurement but reject the
  observer as production instrumentation. Keep
  `engine.decode_stage_observer_max_events=0` by default and retain the
  bounded observer only for benchmark experiments.
- Evidence: the final-source matrix has 16 fresh Aster processes across B4
  short/mixed and four repetitions per state. Source, input, execution
  contract, output/finish, terminal, and fallback gates pass; observer-off
  retains zero events and observer-on retains 11/18 timed events with zero
  drops.
- Performance delta: observer-on versus observer-off decode is `-1.063%` in
  B4-short and `-4.804%` in B4-mixed. TTFT p95 rises `+6.980%`/`+3.140%`,
  end-to-end p95 rises `+5.646%`/`+3.099%`, and B4-mixed peak MLX/RSS rises
  `+3.650%`/`+7.055%`.
- Attribution: the diagnostic window is dominated by existing lazy evaluation
  (`93.116%` short, `94.648%` mixed median share), but that share includes the
  MLX work released by the current materialization boundary and is not a
  private-kernel claim.
- Next gate: I093 must reduce instrumentation to under `1%` overhead before
  any stage share is used to choose a runtime change. TileMix (`2608.17336`)
  and CoRun (`2608.14376`) are watch-only frontier inputs; MTP remains
  foundation-gated.

## 2026-08-21: Keep sampled decode attribution benchmark-only in I093

- Decision: reject the periodic decode-stage observer as a production
  instrumentation change. Keep `engine.decode_stage_observer_max_events=0` by
  default; retain `decode_stage_observer_sample_interval` and the resettable
  window only for benchmark diagnostics.
- Implementation: disabled mode has no observer timer/event work. Enabled mode
  samples the first decode call and then every configured Nth call until the
  bounded event limit. The reset clears only diagnostic state after warmup;
  it does not touch model, KV, cache, scheduler, or sampler state.
- Evidence: the fresh adjacent matrix has 32 successful rows across Aster and
  MLX-LM, B4-short/B4-mixed, observer off/on, four repetitions, and balanced
  state order. Source/input, exact output/finish, terminal, zero fallback,
  zero swap, and observer bounds pass. B4-short samples two steps per timed
  repetition; B4-mixed samples three.
- Performance ledger: Aster paired median decode changes are `-0.275%` in
  B4-short and `-2.032%` in B4-mixed. Mixed Aster order strata are `-16.850%`
  off-first and `+3.183%` on-first; mixed peak-MLX strata include `+10.816%`.
  The MLX-LM control also varies by more than `5%` in both mixed decode order
  strata. Thus the strict `<1%` no-op gate is false and the timing is valid
  diagnostic evidence but confounded for attribution.
- Interpretation: sampled stage shares remain dominated by the existing lazy
  evaluation window (`91.476%` short, `94.678%` mixed), but that window
  includes work released by the materialization boundary and is not a private
  kernel claim. No runtime optimization is selected.
- Artifact: `docs/loop-engineering/artifacts/ITER-20260821-093-low-overhead-decode-stage-attribution/decode-stage-observer-sampled-matrix.json`,
  SHA-256 `f64adc494134c63b046a4ed4606bd7bc1fbe3efd0b43eb2ca0ca25d6620f31b5`.
- Next gate: I094 lengthens the B4 window and requires both Aster and control
  order strata to stabilize before any decode-stage owner is assigned. MTP,
  DFlash, EAGLE-family, tree speculation, and multi-token prediction remain
  foundation-gated.

## 2026-08-21: Reject longer-window observer attribution in I094

- Decision: reject the periodic sampled observer as production instrumentation
  and keep `engine.decode_stage_observer_max_events=0`. Add only a
  benchmark-harness output-cap parameter; the production inference path and
  defaults remain unchanged.
- Implementation: `--max-output-tokens` now propagates through the public
  cohort plan, Aster/direct-MLX-LM child commands, execution envelope, and
  completion-length contract. The default remains eight tokens, preserving all
  prior baselines. The I094 matrix uses 32 tokens and sample interval eight.
- Evidence: 32 fresh rows (B4-short/mixed x Aster/MLX-LM x observer off/on x
  four repetitions) pass source/input, exact output/finish, terminal, zero
  fallback, zero swap, and observer event/drop gates. Observer-on samples five
  steps per B4-short row and seven per B4-mixed row.
- Performance ledger: B4-short Aster decode is `81.406115 -> 81.652300`
  tok/s, paired median `+0.613%`, with order strata `+1.594%/+0.160%`.
  B4-mixed is `46.954306 -> 49.853883` tok/s, paired median `+6.393%`, but
  order strata are `-2.174%/+30.592%`; the four paired deltas are
  `-5.593%/+49.643%/+1.245%/+11.541%`. The MLX-LM control also varies beyond
  `1%` in mixed tails and decode strata, so this is control-confounded
  diagnostic evidence rather than a speedup claim.
- Boundary result: at the same 32-token observer-off window, Aster is
  `12.693%` faster than MLX-LM in B4-short and `36.373%` faster in B4-mixed.
  This reverses the eight-token relationship and proves the comparison is
  window/workload-scoped, not a global engine ranking.
- Attribution: evaluation remains the dominant sampled window (`95.501%`
  short, `96.565%` mixed median share). This includes work released by the
  existing materialization boundary and is not a private-kernel claim.
- Artifact: `docs/loop-engineering/artifacts/ITER-20260822-094-mixed-load-attribution-stability/mixed-load-attribution-stability.json`,
  SHA-256 `2cddae9bf06f6fb129a2b86893c76417b76b50bd7c307594ba713b118b2f7fb4`.
- Next gate: I095 uses a control-first decode-boundary design with explicit
  host-state classification. MTP, DFlash, EAGLE-family, tree speculation, and
  multi-token prediction remain foundation-gated.

## 2026-08-21: Reject I097 after retained quiescence timeout

- Decision: close I097 as an invalid formal performance attempt and retain its
  benchmark-only admission guard. Do not relax the preregistered CPU limits,
  replace the timed-out row, infer a speedup/slowdown, or change production.
- Evidence: the first planned direct-MLX-LM B4-short control row waited
  `120.084624s`, retained 1,156 raw samples and 1,137 rejected 20-sample
  windows, and launched no child. Whole-wait CPU median/p95 were
  `16.2%/33.3%`; minimum window p95 was `17.025%` versus the `12%` gate.
  Available memory stayed above `35.097%`, and swap never changed.
- Performance ledger: I096 remains the latest valid baseline. I097 candidate
  TPS, absolute delta, relative delta, and order strata are all `null`, with
  zero completed performance rows. Status is `invalid-quiescence-timeout` and
  decision is `reject-quiescence-timeout`.
- Retained change: deterministic rolling admission, complete window/timeout
  evidence, and sample-aligned child-normalized external CPU fields remain in
  the benchmark harness. Production model, scheduler, cache, sampler, and
  defaults are unchanged.
- Next gate: I098 uses two independently owned same-process decode branches in
  four-token AB/BA blocks. If the control cannot hold all medians/order strata
  within `1%`, a dedicated headless host becomes mandatory. MTP, MemSpec,
  Windowed-MTP, AngelSpec, and other speculative work remain reference-only.
