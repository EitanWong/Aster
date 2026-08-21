# Known Issues

- I090 establishes a valid 32-row Qwen3.5-9B foundation baseline, but two B4
  cells have declared cross-engine output divergences (`b4-short/short-3` and
  `b4-mixed/short-0`). Aster's paired median decode-driver deficit is
  `+36.322%` and `+24.484%` in those cells; B1-long reverses by order. I091
  must align the output contract and remeasure a common decode-driver boundary
  before any production change. Every iteration now requires this explicit
  baseline/delta ledger and a pushed consolidation commit.
- Historical workspace debt remains. Its immutable baseline is 1,164 changed
  paths / 1,114 artifact files at `2cb14052d4a3`; the active iteration keeps its
  own 20-file / 5 MiB budget. The latest strict counts and warnings are tracked
  in `CURRENT.json`; generated Python caches remain workspace warnings rather
  than iteration artifacts.
- Long-context snapshot preflight is now implemented and the full suite is
  green (`568 passed, 9 skipped, 1 warning`). The threshold and both checkpoint call paths
  have automated coverage. The earlier controlled 128K comparison supports
  the 2 GiB cap, but a fresh default-8-GiB real-model run through the automatic
  policy has not yet been archived as formal evidence.
- Ruff passes for every Python file touched by I081-I085. The broader
  `aster tests scripts` tree still reports 227 historical lint errors, so
  full-tree Ruff remains an explicit repository-debt boundary rather than an
  iteration admission gate.
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
- The configured Web search still returns HTTP 404, but a 2026-07-31 read-only
  GitHub/ArXiv refresh succeeded through official endpoints. It pins current
  SGLang/vLLM/MLX-LM heads and six current cache/agent papers in I083's compact
  artifact. Network search quality remains unavailable, so conclusions use
  only upstream source, commit metadata, and paper records.
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
- The experimental `PagedKVCacheLayer` is lossless and COW-capable, and its block pool no longer repacks with `mx.stack` on every view. `PagedKVCacheBundle` preserves deep-copied `ArraysCache` layers and reclaims full-attention pools after the last fork releases. The default storage-only integration still maintains contiguous K/V, direct attention is B1-only, and batch merge falls back to native contiguous caches. It remains disabled in production paths.
- The opt-in hybrid list boundary is parity-clean on the Qwen3.5-0.8B greedy smoke. Contiguous-buffer reuse brings 8.4K randomized A/B to within `0.4%` median of native, but peak memory remains about `7.6%` higher (`2.471 GB` vs `2.297 GB`); the 2.2K single-run path remains about `10.6%` slower. Prefix snapshots are disabled and decode batch size is restricted to one; it is not a default path.
- The original tiled block-indexed Metal kernel was `1.56x/3.42x/7.44x`
  slower than native at 512/2K/8K. A later vector kernel beat native on the
  I046 `Hq=16/Hkv=8/D=128` 8K boundary, but I086's shared-pool
  `Hq=16/Hkv=4/D=256` confirmation regresses median-of-process p95 by 17.70%
  at 10,334/B8. Numerical error stays <=6.10e-05 and pool lifecycle is clean;
  execution performance, not correctness or reclamation, blocks serving.

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
- I077's bounded prompt-free reservation trace cleared its output/TTFT no-op
  gate. I079 excluded 3 GiB at a six-key reservation floor and advanced 4 GiB
  to a fresh balanced screen. I080 then rejected 4 GiB: only one of four
  disjoint windows retained all six keys without eviction; the other three
  evicted `2,110,849,024` bytes in total during the final exact-replay
  reservation. Replay still hit with zero prefill because lookup occurs first.
  I081's exact-hit predicate suppresses that duplicate work in all three
  previously failing 4 GiB windows while retaining exact output, LRU touch,
  pin/unpin, and terminal cleanup. I082 then passed four fresh configured
  8 GiB windows plus real-model cancellation and persistence/cancellation
  controls. The small predicate is admitted for a minimal production commit;
  clone reserve, general eviction policy, and the production default remain
  unchanged. I083 proves sequential exact/strict/cancellation stability. I084
  then measures repeatable exact-hit fanout ownership: active estimates scale
  as one/three/seven full request estimates at B2/B4/B8, B8 MLX peak rises by
  2.330 GB over B4, and replay latency reaches a 3.913s median / 6.512s p95.
  I085 then proves that `copy.deepcopy` construction has zero physical growth;
  native batch merge is the owner. At B8 it materializes 3,120,824,832 logical
  bytes, 86.80% in eight full-attention layers and 13.20% in 24 linear-attention
  layers. Typed forking is rejected and the current clone remains the default.
  Aster's experimental paged bundle cannot yet combine prefix caching with B>1
  decode, while SGLang's local MLX pool still materializes contiguous rows.
  I086 proves the direct singleton-pool/two-dimensional-table ownership shape:
  B8 total/full construction falls 86.7941%/99.9985%, numerical parity and
  release pass, and no native merge is invoked. It nevertheless fails latency
  in all five 10,334/B8 confirmation processes (p95 ratios 1.078x-1.394x), so
  the predeclared stop rule blocks model-runner membership/extraction/
  cancellation integration and the locked 9B A/B. B8 host-global swap remains
  pressure context only. I087 shows that an opt-in active cap of four removes
  2.037 GB / 19.237% peak MLX memory and improves throughput/tail latency on
  the exact-prefix B8 cell while preserving cancellation cleanup. The global
  default remains unchanged because caps 2/3/5/6 and short/distinct/mixed
  traffic have not yet established a conditional frontier.
- I088 closes that frontier without admitting a policy. Short-simultaneous has
  no eligible lower cap, so the global intersection is empty. In mixed traffic,
  `mixed-short-3` also changes greedy output under caps 2/5 versus 3/4/6/16.
  Fresh diagnostics locate the first difference at completion index 6 after a
  shared six-token prefix: single-row decode selects token 364 and two-row
  decode selects token 421 from candidates separated by at most 0.125 logits.
  I089 isolates the owner: Aster and native MLX-LM produce byte-identical
  paired-history caches and the same 364/8574/421 selections for serial,
  paired-history single, duplicate, and original-companion controls.
  Merge/extract is intact. The remaining issue is a reference-shared BF16
  near-tie whose ordinary argmax depends on decode history and cohort shape;
  exact output identity across different shapes is not a general invariant.
  Diagnostic timing is invalid, epsilon tie-breaking is not admitted, and all
  production scheduler/sampler/precision defaults remain unchanged.
- MTP remains deferred after I089. Adding speculative next-n verification now
  would multiply cohort-sensitive arithmetic, rollback, sampler, stop, and
  membership contracts before 9B foundation parity is measured. A usable
  model-side MTP head is also a separate compatibility requirement. The local
  llama.cpp, vLLM-MLX, and OMLX sources remain design references until the
  parity and full verification/rollback gates pass.
- I091's processor-free batch-wide logprob normalization is exact but rejected:
  balanced B4-short decode changes `-1.584%`, B4-mixed changes `+0.025%`, and
  one mixed candidate row grows host swap by `317,587,456` bytes. The opt-in
  `decode_tensorized_logprobs_enabled` switch remains false by default; it is
  not evidence for a production speedup. I092 must attribute the remaining
  decode-driver boundary without forced evaluation.
- I092's per-step decode-stage observer is valid for attribution data but fails
  its no-op gate as instrumentation: B4-short/B4-mixed decode changes
  `-1.063%/-4.804%`, TTFT p95 changes `+6.980%/+3.140%`, and mixed peak
  MLX/RSS changes `+3.650%/+7.055%`. It remains benchmark-only with a zero
  default. I093 must use aggregate or sampled counters and clear a stricter
  `<1%` overhead gate before stage data can select a runtime candidate.

- I093's periodic sampled observer preserves exact output/finish/terminal
  behavior and bounded events, but it still cannot be used as production
  instrumentation. The 32-row adjacent B4 matrix has Aster paired median
  decode changes of `-0.275%` (short) and `-2.032%` (mixed); mixed Aster order
  strata are `-16.850%/+3.183%`, and the MLX-LM control also exceeds `5%` in
  both mixed strata. The no-op gate therefore fails and the measurement is
  explicitly marked control-variance-confounded. Keep the observer disabled by
  default until a longer-window I094 matrix stabilizes both engines.
- Reference refreshes are source observations, not silent benchmark changes.
  The local llama.cpp pin is now `0e1d9185`; the prior read-only observation
  `6503355d` is historical, while MLX-LM remains pinned at `d06c5374` after
  its 2026-08-19 MLX 0.32.1 update. All configured reference branch refs were
  fetched in the I093 refresh and eleven gitlinks advanced. Reproducible Aster
  evidence must continue to name both the local pin and the refresh date.

- I094 lengthened the observer window to 32 generated tokens and preserved all
  correctness/resource contracts, but attribution is still not stable. Aster
  B4-mixed decode order strata are `-2.174%/+30.592%`; the MLX-LM control also
  violates the `1%` mixed stability gate. The apparent `+6.393%` paired median
  is therefore rejected as a performance claim. The same window makes Aster
  `12.693%/36.373%` faster than direct MLX-LM in B4-short/mixed, reversing the
  earlier short-window direction; all cross-engine conclusions remain scoped to
  the named token window and public cohort.
- The benchmark foundation now accepts a parameterized output cap, but this is
  measurement infrastructure only. Production requests, scheduler behavior,
  cache ownership, and inference defaults remain unchanged. I095 must classify
  host/control state before another observer or runtime candidate is considered.
