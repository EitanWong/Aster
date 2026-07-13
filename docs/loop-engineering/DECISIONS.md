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
