# Known Issues

- Historical workspace debt remains. Its immutable baseline is 1,164 changed
  paths / 1,114 artifact files at `2cb14052d4a3`; the active iteration keeps its
  own 20-file / 5 MiB budget. The latest strict audit has 83 incremental paths
  and 46 artifact files (0.54 MiB), with zero staged paths, mixed-index paths,
  and reference updates, plus 24 generated caches. It warns about retained
  closed-I060 through I071 evidence and generated caches rather than
  laundering or deleting owner-unknown work.
- Long-context snapshot preflight is now implemented and the full suite is
  green (`536 passed, 9 skipped, 1 warning`). The threshold and both checkpoint call paths
  have automated coverage. The earlier controlled 128K comparison supports
  the 2 GiB cap, but a fresh default-8-GiB real-model run through the automatic
  policy has not yet been archived as formal evidence.
- Iteration 059 now retains exact EOS-membership reuse beside the one-entry mask
  snapshot. Formal short/long balanced interval lower bounds were 38.01% and
  21.09%; all 36 outputs matched, stop-aware B4 was 4/4 schema-valid, and fresh
  numeric memory confirmations passed. The initial omission of numeric memory
  ceilings is recorded as a protocol deviation; the first matrix is discovery
  evidence for memory, and only the later confirmation closes that gate.
- Iteration 060 resolves the measured sequential JSON-freetext ownership
  growth: two source-bound pairs reduced short/long median RSS growth by
  98.68%/97.61%, retained one active state per lane, and released request
  `TokenList` objects after lanes finished. The implementation overrides LMFE
  0.11.3 private traversal behavior, so an LMFE upgrade needs source review and
  fresh parity/formal evidence. Broader schemas, tool calls, non-monotonic
  structured callers, concurrency/cancellation pressure, energy, and sustained
  thermal behavior remain unmeasured.
- Iteration 061 establishes a fair but narrow local Aster/direct-MLX-LM
  comparison: same 0.8B model files, locally constructed prompt content,
  greedy sampler, output cap, and exact completion IDs/text/finish. Short
  128-word/256-token decode is 5.743% slower at paired p50 (95% bootstrap
  `[-9.228%, -4.499%]`), while the 2,048-word/64-token case is 10.431% faster
  (`[+5.749%, +13.394%]`). It is historical scenario evidence only; public
  data and complete required-engine coverage are now mandatory before selecting
  another bottleneck or making any broader comparison.
- I062 rejects two plausible short-decode explanations. Matching production's
  empty no-processor token context changes the loop by only 0.052%/0.246%.
  MLX-LM lookahead graph scheduling preserves exact output but has a
  sign-reversing six-process order interaction: serial-first p50 is -27.123%
  while pipeline-first is +30.879%, with a bootstrap median interval spanning
  -27.195% to +38.942%. Short decode benchmark cold/warm, allocator/cache,
  stream, and thermal state classification is now a prerequisite for any
  asynchronous runtime change.
- I063 rejected three explanations for the short-decode order interaction:
  `mx.clear_cache()`, `gc.collect()`, and the fixed terminal prewarm graph. In
  eight exact crossed-prewarm processes, pipeline-first p50 was +16.746% versus
  serial-first -14.022%, while every matched contrast stayed positive. GC
  normalized active MLX memory but not timing; coarse host values and unavailable
  MLX stream counts did not identify the remaining state owner.
- I064 has classified a material same-process call-position confound: 7/8
  same-variant second calls were slower, with serial/pipeline medians
  -25.955%/-28.110%. The pipeline bootstrap interval still reaches +4.869%
  because of one positive record, so this is not a precise hardware mechanism
  estimate. I065 must ensure I061's isolated one-timed-call engine baseline
  remains separate from this diagnostic result. Thermal, power, and stream-level
  instrumentation remain unmeasured.
- I065 admits the I061 isolated-process boundary: 24 unique PIDs, one warmup
  plus one timed decode in each engine source branch, and 3/3 process-order
  balance per scenario. I066 has now completed its pinned MT-Bench/LongBench
  public-data foundation: 1,380 records x Aster/direct-MLX-LM passed source,
  model/Tokenizer, input, output-token, metric, and zero-swap gates. Its one
  order-alternated core matrix still has only one process per engine/task shard,
  so its workload/length-dependent timing directions cannot yet select a
  runtime candidate or support an engine ranking. Its one-pass limitation is
  superseded by I067's crossed result below.
- I067 completed the reversed public core matrix: both 1,380-record matrices
  pass all eight comparability gates, their nine-gate crossed join passes, each
  engine is first for 1,380 records, and swap stays flat. I068 resolves the
  QMSUM subcase with four independent ABBA blocks: 1,600 public engine-records
  pass all cross-block parity and zero-swap gates, and a stable Aster decode
  deficit (`-7.775%` / `-8.216%`) plus end-to-end increase (`+5.452%` /
  `+5.352%`) remains. I069 repeats the locked scope with component tracing and
  confirms a common decode-driver increase of `+8.791%` / `+8.655%`, with
  zero cache merges/rebuilds in B1 and negligible cache/processor/delivery
  shares. Its dominant Aster `sampling_completion` field includes the lazy MLX
  completion barrier and has no comparable direct private substep. I070's
  source-bound observer preserves exact output but fails its 3% no-op gate on
  both engines, so it does not align a production-valid lower-level boundary.
  QMSUM ABBA is intentionally not rerun from that observer.
- A 2026-07-29 external-source refresh could not complete in this environment:
  the configured Web search returned HTTP 404 and read-only GitHub API requests
  returned HTTP 403. The frontier radar retains local/pinned evidence only
  until a future read-only lookup succeeds.
- The installed comparison set currently contains only Aster and direct MLX-LM.
  Exo, Ollama, llama.cpp, vLLM, SGLang, vLLM-MLX, MLC-LLM, mistral.rs, LM Studio
  MLX Engine, and OMLX probes are unavailable. Their absence is recorded in the
  public inventory; a future install still needs a same-model/tokenizer adapter
  before it can participate.

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

- I071's public arrival/load baseline exposes process-level swap growth in B8,
  shared-prefix, and some staggered long-prefill runs. This is evidence rather
  than a scheduler fault attribution: the harness records it, but does not yet
  distinguish model loading, OS compression/swap, prefix snapshots, and policy
  effects.
- I072 admitted the 512-token decode-aware prefill cap after four fresh
  cache-on/cache-off lifecycle runs showed no positive candidate-specific swap
  stage and retained exact output parity. The earlier 1,043,791,872-byte
  candidate observation is non-repeatable as a policy effect, but it does not
  identify a global OS-memory root cause.
- I073 separated prefix-snapshot reuse, one-entry eviction, and cancellation
  ownership from output/cancellation correctness, but did not identify a
  cache-specific host-memory owner. Shared-prefix global swap grew in both
  cache states (+883,752,960 bytes off; +364,576,768 bytes on), while all four
  distinct/cancellation rows were zero. Since the meter is host-global and
  each state has one fresh row, it cannot select a cache policy.
- I074 completed the no-request lifecycle and opposite-order shared-prefix
  controls and rejected cache-policy selection: idle/off/on and one shared
  off/on/on group stayed at zero swap, while only the final cache-off row
  observed +78,577,664 bytes. The signal is neither cache-state-repeatable nor
  process-owned. The retained empty-plan control is now available for future
  lifecycle experiments.
- I075 rejected a 1 GiB snapshot budget despite a 1,025,114,112-byte explicit
  retention reduction: its first-record replay changed from an exact 0-step
  hit at 0.279007s to an 8-step miss at 19.219282s. Global swap remains context
  only; the configured 8 GiB default is unchanged.
- I076 rejected the 2 GiB budget despite final bytes under its cap: two existing
  clone-reserve evictions changed first replay from a 0-step `0.226819s` hit to
  an 8-step `27.859819s` miss. Aggregate final cache counters lack the
  configured/state/effective budget and per-call reserve target, so they cannot
  distinguish which reservation caused the loss.
- I077 must add a bounded prompt-free reservation decision trace and clear an
  output/TTFT no-op gate before it profiles another snapshot budget. Clone
  reserve, eviction policy, and cache defaults remain unchanged.
