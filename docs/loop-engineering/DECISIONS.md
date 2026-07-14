# Decisions

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
- Next experiment: randomized sustained branch/cancel/recovery ordering, then
  model-native fixed-shape state isolation for batching.
