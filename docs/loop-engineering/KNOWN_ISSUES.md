# Known Issues

- Historical workspace debt remains. Its immutable baseline is 1,164 changed
  paths / 1,114 artifact files at `2cb14052d4a3`; the current 1,172/1,121 state
  is within the explicit +25/+20 growth allowance, and the active iteration has
  its own 20-file / 5 MiB budget. Strict checking now passes with WARN rather
  than laundering or deleting owner-unknown work. Five mixed-index paths, 13
  reference updates, and foreign iteration artifacts remain visible; generated
  caches are zero.
- Long-context snapshot preflight is now implemented and the full suite is
  green (`498 passed, 9 skipped`). The threshold and both checkpoint call paths
  have automated coverage. The earlier controlled 128K comparison supports
  the 2 GiB cap, but a fresh default-8-GiB real-model run through the automatic
  policy has not yet been archived as formal evidence.
- Iteration 059 now retains exact EOS-membership reuse beside the one-entry mask
  snapshot. Formal short/long balanced interval lower bounds were 38.01% and
  21.09%; all 36 outputs matched, stop-aware B4 was 4/4 schema-valid, and fresh
  numeric memory confirmations passed. The initial omission of numeric memory
  ceilings is recorded as a protocol deviation; the first matrix is discovery
  evidence for memory, and only the later confirmation closes that gate.
- LMFE retains a fresh 246,884-entry `TokenList` for prefix states in the
  measured JSON workload. Residual profiles grew process RSS by about 3.69 GB
  at short B4 and 1.46 GB at long B2. This requires a separate ownership and
  lifetime investigation and is now the bounded Iteration 060 objective.

- Independent MLX streams for opt-in BatchGenerator lanes do not clear the 3%
  end-to-end gate. `BatchGenerator.next()` already performs stream-bound
  synchronous evaluation, and staggered cohort membership remains sensitive to
  arrival timing; the candidate is not enabled by default.

- Benchmark sampling is explicit; non-greedy paired runs reset the same seed
  before adjacent baseline/production calls. Widely spaced fresh-process A/B
  remains vulnerable to desktop load and frequency changes, so hot-path
  candidates need alternating adjacent pairs plus independent process repeats.
- Adjacent pairs have a visible second-call interaction on long B2. The strict
  256-step screen retained a production-first lower-bound miss, while the
  predeclared 1,024-step confirmation cleared balanced and both order strata.
  Final admission balances AB/BA counts and keeps the order strata as explicit
  gates for core cells; structured host-driven fallback uses its balanced
  no-regression interval and reports order interaction separately.
- Direct benchmark records include RSS, swap deltas, prompt token counts, prefix-store counters, admission rejections, overall MLX peak, and prefill active/peak memory for completed responses, but not energy or allocator peaks for failed requests. The paged-attention probe separately records randomized A/B timing and allocator peaks.
- `BatchGeneratorRuntimeKernel.available` is false; continuous batching is implemented by the manual scheduler and decode batch runner only.
- 9B hybrid memory accounting and bounded prefix snapshots now allow 30K prompts to complete at about 1.38 tok/s with 1.10 GiB swap growth in the best measured run; 32K mixed-agent, 35B, cancellation pressure, and 30-minute stability evidence remain incomplete.
- The admission-before-prefill scheduler experiment was rolled back: randomized mixed and staggered A/B did not meet a reliable performance gate.
- Paged KV, SSD cache, KV-cache quantization, structured-output black-box parity, and full tool parser parity remain open.
- Stop-aware structured B4 now validates 4/4 schema outputs under membership
  shrink, but a retained lane-0 prompt generated an unbounded valid-string
  interior until the 256-token length limit. Structured model behavior still
  needs broader schemas, prompts, tool calls, and maximum-string safeguards.
- Experimental KV quantization is not enabled: 4-bit KV failed fixed greedy token parity and 8-bit has no demonstrated material gain.
- The stronger OMLX/mlx-vlm 4-bit TurboQuant reproduction is also rejected.
  It reduces complete Qwen3.5 hybrid-cache bytes by `1.72x` at 2K and `2.67x`
  at 8K, but decode regresses `5.22%/5.72%`; only 3/5 greedy windows match at
  each context, teacher top-1 falls to `89.06%/93.75%`, and absolute PPL
  change reaches `7.49%/3.38%`.
  Isolated compressed attention beats Aster paged but not native MLX across
  the measured 2K/8K/32K/64K curve.
- Post-sample cache-tree evaluation and batch-size-proportional sample waits
  are removed after exact native/recurrent/paged RAW/WAW and heterogeneous
  B2/B4/B8 validation. One grouped sampled-token wait remains necessary before
  host stop/stream handling. Iteration 052 isolated greedy `logsumexp` and found
  only `13~53 us` of graph delta plus `+0.50%~+1.27%` paired real-model gains,
  below the 3% gate. Iteration 056 then measured host post-eval work below
  `0.3%`, active-penalty tensorization below `0.7%`, and processor-free batched
  normalization at `-1.16%~+1.78%`; all remain benchmark-only. Iteration 057
  removed the JSON processor's redundant 246k-entry allowed-token copy, but
  direct Python-token delivery remained below the standalone gate. Iteration
  058 then retained a request-local one-entry mask cache with exact list
  verification, mutable-input snapshotting, and shape invalidation; short B4
  and long B2 cleared all speed, parity, swap, and bounded-memory gates. The
  post-cache shares of LMFE `TokenList` construction/traversal, full-history
  MLX-to-Python token conversion, and model/sampling time remain unmeasured.
  Broader schemas, structured concurrency, energy, and sustained thermal
  behavior also remain open. Production keeps the normalized-logprob and
  per-row processor contracts.
- Iteration 055 bounds only Aster-created MLX-LM penalty processors. Structured,
  thinking, unknown custom, and legacy work-item processors still receive full
  history by design. The admitted result covers active penalties at B2 and
  24,601 prompt tokens; long B4/B8, custom processor scaling, energy, and
  sustained thermal behavior remain unmeasured. A failed 64-step formal matrix
  demonstrates that short measurement windows are too noisy for this roughly
  5% effect; the admitted matrix uses 256 steps and retains both order strata.
- Decode allocator cache is cleared after 512 generated tokens, not 512
  scheduler steps. The fixed-step candidate accumulated `481.42 MB` of
  free-cache in long batch-4 stress; token-normalized clearing held post-clear
  cache to `3.05 MB`. Cancellation/replacement churn beyond existing lifecycle
  tests still needs a sustained real-model memory trace.
- `kv_cache_step_tokens` reduces native KV growth copies but does not reduce retained KV memory; a true paged attention path remains unimplemented.
- The experimental `PagedKVCacheLayer` is lossless and COW-capable, and its block pool no longer repacks with `mx.stack` on every view. `PagedKVCacheBundle` reclaims full-attention pools after the last fork releases, but mixed recurrent/full-attention bundles are rejected and the MLX integration still materializes contiguous K/V on every update; batch merge falls back to native contiguous caches. It remains disabled in production paths.
- The opt-in hybrid list boundary is parity-clean on the Qwen3.5-0.8B greedy smoke. Contiguous-buffer reuse brings 8.4K randomized A/B to within `0.4%` median of native, but peak memory remains about `7.6%` higher (`2.471 GB` vs `2.297 GB`); the 2.2K single-run path remains about `10.6%` slower. Prefix snapshots are disabled and decode batch size is restricted to one; it is not a default path.
- The experimental block-indexed Metal kernel is numerically correct on Qwen3.5-shaped FP16 input after the threadgroup-grid fix, but the corrected median benchmark is `1.56x/3.42x/7.44x` slower than native at 512/2K/8K in the recorded run. Pool reclamation and hybrid bundle lifecycle are incomplete, so it is not a serving path.
