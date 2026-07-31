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
