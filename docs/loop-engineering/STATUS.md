# Loop Engineering Status

Updated: 2026-08-01

## Current State

### Iteration Control

- Canonical current state: `docs/loop-engineering/CURRENT.json`.
- Operating contract: `docs/loop-engineering/ITERATION_PROTOCOL.md`.
- Last completed iteration: 085. Active iteration: 086, phase `planned`.
- I061 admitted a same-host Aster/direct-MLX-LM baseline with identical model
  files, locally constructed prompts, greedy sampling, fixed output caps,
  token/text/finish parity, and zero swap growth across 12 independent pairs.
  It remains historical isolated-process evidence only: its local input cannot
  select a future production optimization or support an engine-ranking claim.
- I062 rejected both short-decode screens without changing production code:
  no-processor work-item history was sub-0.25%, while the MLX-LM lookahead
  pipeline had exact output but sign-reversing order strata.
- I063 rejected allocator clearing, delayed Python collection, and fixed
  terminal prewarm order as sufficient explanations. Its 8-process crossed
  prewarm screen retained exact output and zero swap; pipeline-first p50 was
  +16.746% while serial-first was -14.022%, and all four matched contrasts
  stayed positive. No asynchronous runtime candidate was authorized.
- I064 rejected adjacent same-process comparison as performance evidence: fresh
  same-variant calls were 25.955% slower at serial p50 and 28.110% slower at
  pipeline p50 on the second call; 7/8 second calls regressed. This is a
  measurement-boundary result, not a pipeline performance claim.
- I065 admitted I061's isolated-process boundary without a new model run: all
  24 record PIDs are unique, each source branch has one warmup plus one timed
  second call, and each scenario has a 3/3 engine-order balance.
- I066 superseded the unstarted isolated Aster-only attribution plan before any
  collection. It established public source provenance and complete cross-engine
  coverage as a hard gate before another production candidate.
- The tracked public source lock now pins MT-Bench and LongBench v1. The local
  download verified MT-Bench's 80 questions and LongBench's 34 JSONL members /
  8,418 archive records, including the 21-task / 4,750-record primary corpus;
  LongBench's official 21 prompt templates and 21 output limits also match.
- I066 generated a 1,380-record scoped `cross-engine-core` workload and a
  4,830-record `full-public` workload. Both retain only public record IDs and
  prompt hashes; no local or copied prompt text becomes benchmark input. Only
  `full-public` may support a complete engine statement within its named
  MT-Bench plus LongBench-primary scope.
- The 12-runtime inventory makes Aster and direct MLX-LM available. Exo,
  Ollama, llama.cpp, vLLM, SGLang, vLLM-MLX, MLC-LLM, mistral.rs, LM Studio MLX
  Engine, and OMLX are explicitly unavailable by local module/command probes.
- I066 added source-bound Aster/direct-MLX-LM result adapters. They reconstruct
  MT-Bench/LongBench prompts from the locked source, retain only input/output
  hashes in results, and pin the direct MLX-LM 2,048-token prefill step in
  Aster as well. This removed an earlier long-prompt output drift from unequal
  prefill chunks.
- The complete I066 `cross-engine-core` run covers 1,380 public records and
  2,760 engine-records. All eight gates passed: source/public prompt identity,
  effective input tokens, execution contract, model/Tokenizer fingerprints,
  deterministic output tokens, coverage, metric completeness, and required
  engines. Both engines kept zero swap delta.
- The scoped first matrix is descriptive, not a ranking: paired median
  Aster/direct results are +15.762% decode throughput, -8.509% prefill
  throughput, -5.112% TTFT, +0.975% end-to-end time, and -3.304% peak RSS.
  Directions vary by workload and input length, and each engine/task shard ran
  once. I066 admits the adapter/matrix foundation but rejects a production
  performance selection.
- I067 completed the same 1,380-record public core matrix in fully reversed
  shard order (2,760 new engine-records). Both individual matrices pass every
  public comparability gate; their joined analysis also passed source, input,
  model, execution-contract, output-token, reverse-order, balanced-first-engine,
  and zero-swap gates. Each engine was first for 1,380 public records.
- The crossed result rejects a global ranking and production bottleneck claim.
  Aggregate Aster/direct prefill is consistently lower (`-10.504%` Aster-first,
  `-7.681%` MLX-first), but long-context decode reverses from `-10.515%` to
  `+27.532%`. `longbench-qmsum` reverses across decode, end-to-end, and
  prefill, so its large effects are measurement-state evidence rather than a
  runtime component attribution.
- I068 completed its public-source-only four-block QMSUM ABBA trace: all 1,600
  engine-records are comparable, token-identical across blocks, state-traced,
  and zero-swap. The reversal did not recur. Aster decode throughput is
  reproducibly lower by `7.775%` when Aster runs first and `8.216%` when
  MLX-LM runs first; end-to-end time is reproducibly higher by `5.452%` and
  `5.352%`. Prefill and TTFT remain inside the 3% no-op band, and peak RSS is
  not reproducible.
- I069 completed the same locked public QMSUM scope with opt-in source-bound
  component tracing: 1,600 engine-records / 800 paired records passed all
  state, component-contract, deterministic output, ABBA, and zero-swap gates.
  Aster's common decode-driver seconds per output token are reproducibly higher
  by `8.791%` (Aster first, 95% `[8.660%, 8.964%]`) and `8.655%` (MLX-LM first,
  `[8.522%, 8.773%]`); aggregate decode throughput is `-8.177%` / `-8.025%`
  and end-to-end time `+5.906%` / `+5.504%`. B1 has no batch-cache merge or
  rebuild, and cache resolution, processor dispatch, and result delivery are
  each immaterial. The dominant Aster-only sampling-completion field contains
  the lazy MLX completion barrier, so it is not a direct private-substep
  comparison. I070 will align a lower-level common boundary; no runtime
  candidate is admitted.
- I070 completed its fresh, source-frozen 80-record-per-engine MT-Bench
  traced/untraced smoke. Source/model/execution, deterministic token/text/finish
  parity, and zero swap all pass after correcting the comparison to match each
  engine's traced/untraced source fingerprints rather than requiring different
  engines to share package files. The observer itself fails every 3% no-op
  metric gate: Aster decode is `-3.635%`, end-to-end `+3.818%`, and TTFT
  `+7.432%`; direct MLX-LM decode is `-7.277%`, end-to-end `+7.717%`, and TTFT
  `+11.477%`. I070 rejects the observer as a QMSUM attribution tool and makes
  no production change or engine claim.
- I071 admitted a source-bound Aster arrival/load harness and baseline. B4/B8
  reached `51.247`/`53.410` aggregate tok/s without decode fallbacks; B8,
  shared-prefix, and a staggered long-prefill repeat exposed swap pressure.
  Exact shared-prefix replay reduced QMSUM TTFT `14.597 -> 0.169s`. Two
  staggered QMSUM-plus-MT-Bench controls showed a short request spending
  `9.346s`/`9.614s` in decode while long prefill continued.
- I072 admitted the decode-aware 512-token continuing-prefill cap as the
  production default. Four balanced public-source pairs retained exact
  token/text/finish parity and improved median short decode/end-to-end by
  `55.204%`/`48.276%`; cancellation left no active state. Its predeclared
  prefix-on/off lifecycle screen retained exact outputs across four fresh
  processes: candidate swap was zero in both cache states except for one
  `-8MiB` prefix-off workload delta, while controls were zero. The cache-on
  runs each retained one `390,103,040`-byte snapshot without swap growth, so
  the earlier `+1.044GB` candidate sample is not policy-repeatable.
- I072 also ran a current-source two-record 9B Aster/direct-MLX-LM
  compatibility smoke. Model/tokenizer, source lock, generation, and shared
  harness sources match; token IDs, text hashes, finishes, and zero swap match
  for both records. This validates output compatibility only, not a timing
  ranking or a replacement for the complete I066/I067 public matrices.
- I073 rejected a cache-policy selection after six fresh locked-source rows.
  Exact cache reuse (10,333 tokens), one-entry distinct-key eviction, and
  cancellation checkpoint ownership all behaved as expected with exact terminal
  output/cancellation parity and zero active state. Shared-prefix workload
  global swap changed in both cache states (+883,752,960 off, +364,576,768 on),
  while all distinct/cancellation rows were zero; `psutil.swap_memory().used`
  is host-global, so this is not a cache-specific owner.
- I074 rejected cache-policy selection after completing that control. Both idle
  rows were zero-swap and submitted no requests; shared-prefix values in
  `off,on,on,off` order were `0,0,0,+78,577,664` bytes. Cache-on exact reuse
  repeated (one hit / 10,333 tokens in each row), while the only positive swap
  value was cache-off. `idle-lifecycle` is retained; no cache default changed.
- I075 rejected the temporary 1 GiB snapshot budget. It lowered explicit
  retained bytes by 1,025,114,112 but changed the predeclared first-record
  replay from an exact 0-prefill-step hit at `0.279007s` TTFT to an 8-step miss
  at `19.219282s`; all terminal output identities still matched. The 8 GiB
  default remains unchanged.
- I076 rejected the temporary 2 GiB budget. It ended under its final cap but
  incurred two clone-reserve evictions and changed first replay from an exact
  0-prefill-step hit at `0.226819s` TTFT to an 8-step miss at `27.859819s`.
  The 8 GiB default remains unchanged.
- I077 admitted a bounded prompt-free reservation observer. Four focused tests
  plus the affected 77-test suite pass; the default-64/max-256 FIFO can be
  disabled with value 0. Its source-bound 8 GiB traced/untraced replay retained
  exact five-request terminal identity, four snapshots / 1,988,067,328 bytes,
  zero eviction, and zero active state. Replay TTFT moved `0.169433 ->
  0.166854s` (`-1.522%`), inside the absolute 3% no-op gate. The five events
  contain no prompt or token-ID payload, and no cache behavior changed.
- I078 then tested the trace-predicted temporary 3 GiB boundary. Every
  pre-reservation store remained below its target; the row retained all four
  snapshots, zero evictions/preflight skips, exact output identity, and a
  zero-prefill replay at `0.165729s`. This admits 3 GiB only for wider
  retention testing; the production 8 GiB default remains unchanged.
- I079 completed a six-distinct-key plus first-replay chain. Its 8 GiB control
  exposed a `3,626,565,632`-byte maximum reservation floor, excluding 3 GiB
  and selecting 4 GiB. The fresh 4 GiB row matched all seven terminal token/
  text identities, retained six snapshots / `2,846,359,552` bytes, recorded
  zero evictions/preflight skips, and preserved exact zero-prefill replay at
  `0.167437s`. The production 8 GiB default remains unchanged.
- I080 rejected the temporary 4 GiB budget after only one of four fresh,
  disjoint, order-balanced windows retained all six snapshots with zero
  eviction. All eight rows preserved exact terminal identity, zero active
  state, and exact zero-prefill replay, but candidate windows 6, 18, and 24
  each evicted another retained snapshot during the final replay reservation
  and ended with five entries. Total candidate eviction was three entries /
  `2,110,849,024` bytes. The production 8 GiB default remains unchanged.
- I081 completed the measured lifecycle fix. Exact-hit TDD reproduced three
  duplicate-store failures before implementation, then the one-line predicate
  change passed the focused, affected (`97`), and full (`554 passed, 9
  skipped, 1 warning`) suites. The default exact path preserves lookup LRU
  touch and pin ownership while avoiding a second reservation/clone/store;
  `snapshot_skip_full_prompt_on_prefix_hit=false` retains the old refresh path.
  Fresh 4 GiB offsets 6, 18, and 24 all matched I080 source/plan/terminal
  identity, retained six entries, and had zero eviction/preflight/replay
  prefill/terminal state. The candidate is admitted only to I082's configured
  8 GiB validation.
- I081 also reran the current-source two-record 9B MT-Bench smoke through fresh
  Aster and direct MLX-LM processes. Source/model/generation, token IDs, text,
  length finishes, and zero swap match; this remains compatibility evidence,
  not a heterogeneous timing ranking or a replacement for I066/I067.
- I082 validated the predicate at the configured 8 GiB budget. Fresh offsets
  6, 12, 18, and 24 all match their I080 source/plan/execution/terminal
  controls, retain six entries with six stores and one exact hit, record zero
  evictions/preflight skips/replay prefill, and clean up all active/pinned
  state. A fresh real-model cancellation plus four persistence/cancellation
  tests also pass. The change is admitted for a minimal production commit;
  the budget and eviction policy remain unchanged.
- I083 completed the packaging and repeated lifecycle gate without creating a
  commit. A fresh 9B cold-plus-eight-exact chain advanced exact hits from zero
  through eight while retaining one store/entry/trace event, zero replay
  prefill, exact token/text/finish identity, zero terminal state, and zero
  swap delta. A real 10,334/10,342-token strict-prefix append repeated with
  one prefill step, identical derived outputs, one retained store, and zero
  terminal pins. Fresh cancellation matches I082, and the two-record
  Aster/direct-MLX-LM compatibility smoke remains exact and zero-swap.
- I083 also added opt-in serial lifecycle plans and compact per-request terminal
  snapshots to the public arrival harness. The existing I080 plan remains byte
  identical. The full suite is `559 passed, 9 skipped, 1 warning`; the arrival
  harness has 22 passing tests and touched Ruff passes.
- The read-only frontier refresh now succeeds. Current heads are SGLang
  `5f9b0db1`, vLLM `82ae4164`, and MLX-LM `e5baded8`; MLX-LM's cache source is
  byte-identical to the local example. Aster already has O(1) exact, bounded
  distinct-length prefix, and sorted-neighbor LCP lookup, so an unmeasured
  Python token trie is not selected. I084 instead measured concurrent exact
  fanout ownership before considering shared-block or copy-on-write state.
- I084 added an opt-in prompt-free lifecycle sampler; its harness has 24
  passing tests. The fresh B2 ABBA no-op
  screen retained exact output/zero swap and kept elapsed, replay TTFT,
  latency, and throughput movement inside 3%. Nine fresh rotated B2/B4/B8 9B
  processes then completed all 42 requests with exact output, expected hits,
  one store/entry, zero eviction/preflight/error, bounded traces, and clean
  terminal state.
- Concurrent active estimates scale exactly as one/three/seven times
  `390,397,952` bytes while retained storage stays at one `390,103,040`-byte
  snapshot. B8 reports 10.588 GB peak MLX memory versus 8.258 GB at B2/B4;
  replay latency grows from B2's 0.464s median / 0.473s p95 to B8's 3.913s /
  6.512s. This passes the gate for I085's type-specific shared-state/COW
  feasibility work, but does not authorize generic shallow copies or a default
  change. Positive B8 host-global swap remains pressure context only.
- I085 rejects the typed shared-state fork before production implementation.
  Qwen3.5-9B uses 24 `ArraysCache` and 8 `KVCache` layers; focused tests prove
  retained-base, sibling, append, trim, merge/extract, and first-write
  isolation. MLX `deepcopy` construction produced zero active-memory growth in
  all nine B2/B4/B8 rows and took only 0.231/0.367/0.741 ms at the median.
- The actual owner is native batch merge. Median B2/B4/B8 active deltas are
  780,402,816 / 1,560,543,488 / 3,120,824,832 bytes and match materialized
  state; every release returns exactly to baseline. Per request, full-attention
  `BatchKVCache` is 338,591,744 bytes (86.80%) and linear-attention
  `ArraysCache` is 51,511,296 bytes (13.20%). Production cloning, the 8 GiB
  budget, eviction, persistence, and rollback behavior remain unchanged.
- Local reference review confirms the next boundary. vLLM lets paged attention
  consume shared refcounted block tables directly. SGLang's MLX slot pool still
  converts the prefix to per-request contiguous caches and concatenates batch
  rows before SDPA. I086 therefore tests only a benchmark-gated shared-prefix
  full-attention path while keeping linear-attention state request-owned.
- I086 adds a benchmark-only singleton-pool/two-dimensional-block-table
  boundary. Ten new tests prove B2/B4/B8 and unequal-length numerical parity,
  absence of batch-prefix materialization/native merge, partial-block CoW,
  independent-pool rejection, and zero-reference release. No model-runner,
  Qwen bridge, configuration, or production-default call site changed.
- The locked 10,334/B8 probe uses 161 common blocks plus eight private tail
  blocks. Candidate metadata is 5,216 bytes versus 338,624,512 bytes for one
  native dense layer. Conservatively retaining all `ArraysCache` bytes and
  multiplying metadata across eight full-attention layers estimates 86.7941%
  less total merge growth and 99.9985% less full-attention construction.
- The current SIMD-group kernel fails the hard latency gate. Across five fresh
  30-warmup/200-measurement processes, median-of-process median latency is
  4.271 -> 5.434 ms (1.272x) and median-of-process p95 is 8.944 -> 9.671 ms
  (1.177x). Every process p95 ratio is 1.078x or worse; max absolute error is
  6.10e-05 and all block/pool releases are clean. The predeclared stop rule
  therefore prevents a 9B model-runner A/B or production integration.
- The 2026-08-01 source refresh pins MLX main `2ad0d4d3`, vllm-metal main
  `b6e35b6c`, and vllm-metal's `32cc5fd7` once-per-forward metadata change.
  PackInfer, Feather, and RadixMLP reinforce that kernel packing,
  prefix-homogeneous scheduling, and position-wise prefill deduplication are
  distinct mechanisms; none rescues this rejected kernel shape.
- The current engine-gap assessment is recorded in
  `CORE_REFERENCE_MATRIX.md`. It distinguishes confirmed Aster capabilities
  from unmeasured reference-engine differences: the public QMSUM result now
  points to a stable decode driver rather than prefill, but not yet to one
  comparable low-level operation. Paged KV, compressed cache, SIMD/Metal
  kernels, native-runtime work, and CPU tokenization remain conditional on the
  public arrival/load evidence.
- Gigatoken `34a1599` is pinned as a remote P1 CPU-ingress reference, not an
  installed dependency. It may help host prompt tokenization only after I071
  and only if Qwen3.5 public/chat/special-token parity plus queue-aware TTFT
  and end-to-end gates pass. It does not stand in for GPU/Metal SIMD inference
  work, which remains a separate measured kernel class.
- The pre-consolidation I086 verification is green: the hash-bound attention
  summary records all six scratch-result hashes, recomputes the selection
  decision, and verifies all five source hashes; the affected suite passes
  33/33, and the full suite is
  `578 passed, 9 skipped, 1 warning`.
  Touched-file Ruff and formatting pass; full-tree Ruff still reports 227
  historical issues outside this iteration boundary. Final strict workspace
  counts are recorded in `CURRENT.json`.
- Iteration 059 retained only 13 artifact files / 0.47 MiB, compacting 50
  logical evidence files into one 237,686-byte archive. Its repeated scratch
  output was removed after archive-only recomputation passed.
- Iteration 060 retains 9 artifact files and one 218,232-byte archive. Its 49
  logical evidence records recompute two formal matrices and two repeated
  source-bound memory comparisons exactly.
- Iteration 061 retains 7 artifact files and one 31,783-byte archive. Its 38
  logical evidence records recompute a 12-pair, two-scenario local
  cross-engine matrix without scratch.
- Iteration 062 retains 7 artifact files and one 5,996-byte rejected-screen
  archive. Its 13 logical records recompute both rejected profile branches
  without scratch.
- Iteration 063 retains 7 artifact files and one 19,498-byte rejected-screen
  archive. Its 14 logical records preserve allocator/GC and crossed-prewarm
  evidence; the archive-only recomputation passes without scratch.
- Iteration 064 retains 6 artifact files and one 23,097-byte rejected-screen
  archive. Its 8 records recompute the same-variant call-position result
  without scratch.
- Iteration 065 retains 3 small audit artifacts and reuses I061's 31,783-byte
  archive; no duplicate model evidence or scratch was created.

- Current measured baseline commit: `f3bfb01a8710` (I060-admitted manual
  runtime; production attention remains native MLX).
- Working tree: an uncommitted opt-in independent-MLX-stream candidate is
  present; it remains experimental and is not part of the default path.
- Previous dependency commit: `86ed15c` (refresh compatible dependency lock).
- Orthogonal baseline repair: `25067b8` (`fix: report continuous batching compatibility warning`).
- Dependency refresh: `1a0b993` (latest compatible MLX and serving package set).
- Manual runtime is the production path. `BatchGeneratorRuntimeKernel` remains an unavailable adapter boundary.
- The admission-before-prefill scheduler experiment was rolled back after randomized mixed and staggered A/B did not show a reliable short-request benefit.
- Core reference matrix: `docs/loop-engineering/CORE_REFERENCE_MATRIX.md`.
- Iteration 041 selected transient-aware prefill memory admission as the next
  bounded core candidate, based on OMLX's route-aware prefill guard.
- Iteration 042 retains a production manual-runtime guard: the runner thread
  extracts immutable full-attention dimensions, the scheduler caps each
  prefill chunk against transient score/output memory, and it rejects an
  unaffordable chunk before model execution.
- Iteration 043 makes that guard adaptive: each request tracks observed chunk
  peak growth over the prior active MLX baseline and combines it with the
  static estimate before selecting the next chunk.
- Iteration 044 ruled out snapshot clone as the next primary hot path for a
  single 9B exact hit: 8,372 reused tokens had only 9.545 ms admission work;
  decode dominated the 2.521 s exact-hot request.
- Iteration 045 bounds high-cardinality prefix lookup by probing only the
  distinct retained snapshot lengths and using direct token-key lookups. It
  keeps Aster's bounded flat index instead of importing a trie/radix owner.
- Iteration 046 starts the frontier intake and reproduction track. Uzu and
  vllm-metal are pinned as open Apple Silicon references; the first M5
  reproduction rejects vllm-metal's current split-KV occupancy gate while
  retaining its fused-scatter/lazy-Primitive integration as a candidate.
- Iteration 047 rejects a native attention Primitive after a guarded,
  same-math reproduction. Against a GPU-work-equivalent guarded `mx.fast`
  control, two main cells had nominal >=3% medians but neither established a
  >=3% interval. In a five-process confirmation one fell to 2.13% and the
  other reversed to a 3.51% regression; every 32K/64K stress interval crossed
  zero.
- Iteration 048 validates vllm-metal's fused K/V scatter in its own
  token-contiguous slot-mapping layout, then rejects the transfer to Aster.
  The pinned reference cleared the 3% gate at 1/4/8/16/64/128 tokens, but no
  Aster-layout cell repeated a >=3% gain across both five-process groups;
  confirmation instead found >=3% regressions at 64-token batch 4/8. No runtime
  or private ABI was retained.
- Iteration 049 profiles the complete real-model paged graph and rejects both
  the direct path and 4-bit TurboQuant for production. Ten fresh controls per
  variant measured direct paged elapsed `+0.37%/+0.20%` at 2K/8K. A
  five-process compressed-domain kernel matrix beat Aster paged by
  `18%~60%`, but did not beat default MLX across 2K/8K/32K/64K. The 20-process
  Qwen3.5 model matrix then found `5.22%/5.72%` decode regressions; only 3/5
  greedy windows matched at either context. No runtime code
  was retained.
- Iteration 050 retains token-budgeted decode allocator-cache clearing. Decode
  now relies on sampled-token/logit synchronization instead of re-evaluating
  every cache leaf, and clears after 512 generated tokens rather than every
  scheduler step. Across 18 production bridge processes, native/direct batch
  1/2/4 improved `9.51%~17.90%` over the archived baseline with exact
  token/text/cache parity. Native/direct 10,000-token stress improved
  `5.58%/5.06%`; adaptive batch-4 clearing kept post-clear allocator cache at
  or below `3.05 MB` and swap stayed zero.
- Iteration 051 retains grouped asynchronous batch sampling. The manual runner
  builds each row's existing processor/sampler graph in request order,
  async-submits MLX sampled scalars, waits once, and then materializes ordered
  results. Host-driven structured processors retain eager row evaluation after
  the shared model forward. The strict v7 admission uses 18 fresh processes / 9
  runner-balanced replicates per cell. Short core balanced intervals span
  `[+7.46,+7.85]` through `[+16.78,+17.43]`; a 1,024-step 6,169-token greedy B2
  confirmation spans `[+6.65,+7.44]`, and mixed B8 stress spans
  `[+14.52,+15.41]`. Token/text/cache hashes matched and swap did not grow.
- Iteration 052 profiled a benchmark-only raw-logit path for `temperature=0`
  samplers. Ten fresh paired records across greedy B2/B4/B8, penalties B4,
  and mixed B4 retained exact token/text/cache hashes and zero swap growth, but
  observed speed changes were only `+0.50%~+1.27%`. A same-logits MLX profile
  measured raw-vs-normalized argmax deltas of `13~53 us` at B1/B2/B4/B8;
  this is below the 3% end-to-end gate, so no production change was admitted.
- Iteration 053 expanded raw-logit-safe sampling to shift-invariant built-in
  samplers, but mixed B4/B8 medians were only `+1.22%/+0.05%`; it remains
  benchmark-only.
- Iteration 054 isolated the neutral MLX-LM repetition processor. Greedy
  B2/B4/B8 medians were `+1.47%/+1.57%/+1.50%`, below the standalone gate.
- Iteration 055 retains a bounded processor-context contract. Built-in active
  penalties receive only their required 20-token window; structured/thinking
  processors retain full history and no-processor requests carry an empty
  window. The 24,601-token B2 strict matrix passed with a balanced interval of
  `[+4.75%,+6.72%]`, both order strata above `+4.09%`, 9/9 stable replicates,
  exact output, and zero swap growth. Short B2/B4 lower bounds remained above
  `-1%` and are recorded only as no-regression evidence.
- Iteration 056 profiles residual grouped-decode sampling costs without a
  production change. Host token materialization plus result construction was
  only `0.225%~0.294%` of batch time. Tensorized active penalties measured
  `+0.01%~+0.66%`; processor-free batched normalization measured
  `-1.16%~+1.78%`. All nine payloads retained exact output and zero swap growth,
  but every candidate failed the 3% early gate.
- Iteration 057 retains LMFE allowed-token list reuse in Aster's JSON schema
  processor. The strict 18-process/9-replicate matrices passed at short B4 and
  24,601-token B2 with balanced intervals `[+31.33%,+33.40%]` and
  `[+24.96%,+27.63%]`; both order strata cleared 20%, all 36 outputs matched
  exactly, and swap did not grow. Stop-aware B4 produced 4/4 schema-valid JSON
  results while membership shrank `4 -> 3 -> 1`.
- Iteration 058 retains a request-local one-entry structured mask cache. The
  strict short B4 and 24,601-token B2 matrices passed with balanced intervals
  `[+114.79%,+119.26%]` and `[+64.81%,+68.91%]`; throughput medians improved
  `+112.39%` and `+65.94%`. All 36 outputs matched exactly, swap stayed flat,
  and conservative dual-runner MLX peak deltas stayed below their 16/8 MiB
  bounds. Stop-aware B4 again produced 4/4 schema-valid results under
  membership shrink `4 -> 3 -> 1`.
- Iteration 059 retains exact EOS-membership reuse beside that mask snapshot.
  Strict short B4 and 24,601-token B2 matrices passed with balanced intervals
  `[+38.01%,+47.33%]` and `[+21.09%,+24.47%]`; both order strata cleared 10%,
  all 36 outputs matched, and swap stayed flat. Stop-aware B4 was 4/4 valid.
  A missing numeric memory predeclaration was recorded as a protocol deviation;
  after explicit 4/2 GiB RSS and 16/8 MiB MLX limits were written, four fresh
  confirmations passed.
- Iteration 060 retains request-local LMFE freetext-list reuse and active
  prefix-state retention. The short B4 and 24,601-token B2 formal balanced
  lower bounds were `+16.01%` and `+12.52%`; all 36 outputs matched exactly,
  both order strata cleared 3%, and swap stayed flat. Two independent
  ownership pairs per cell reduced median RSS growth by `98.68%` and `97.61%`.
  Stop-aware B4 was 4/4 schema-valid with membership `4 -> 3 -> 1`; no request
  TokenList remained after lane release.

## Evidence

- Current full worktree suite: `525 passed, 9 skipped, 1 warning`.
- Long-context snapshot preflight: `15 passed, 39 deselected`. Requests below
  65,536 prompt tokens preserve their available-memory budget; requests at or
  above that threshold cap it at 2 GiB before the configured snapshot limit is
  applied. Both checkpoint paths use the request-aware budget.
- Iteration 055 affected runner/runtime/structured suite: `127 passed,
  1 deselected`.
- `compileall` and `git diff --check`: passed.
- Iteration 052 artifact tests: `4 passed`; its operator profile and ten paired
  payloads are hash-bound under the iteration artifact directory.
- Iteration 053/054/055 artifact tests pass (`3/2/6`). Iteration 055 binds 54
  formal long/short payloads, strict manifests, aggregates, and descriptive
  summaries to current source and model hashes; its nine-gate composite
  admission passes.
- Iteration 056 artifact tests pass (`4 passed`). Its aggregate binds nine
  paired real-model payloads and records all three candidates as rejected.
- Iteration 057 affected tests pass (`66 passed`) and artifact assertions pass
  (`3 passed`). Its 36 formal payloads, two strict aggregates, stop-aware
  validation, descriptive summary, and 11-gate composite admission are bound
  to current production/model hashes.
- Iteration 058 affected tests pass (`84 passed`) and artifact assertions pass
  (`3 passed`). Its 36 formal payloads, two 11-gate strict aggregates,
  stop-aware validation, memory comparison, and 13-gate composite admission
  are bound to current production/model hashes.
- Iteration 059 affected plus artifact tests pass (`142 passed`), standalone
  compact-evidence assertions pass (`2 passed`), and its source/model-bound
  composite admission recomputes both strict aggregates and passes 12/12 gates.
- Iteration 060 focused constrained-decoding, runner, and archive tests pass
  (`61 passed`). Its 49-file source/model-bound archive recomputes both strict
  aggregates and passes all 12 composite gates.
- The initial grouped 0.8B mixed A/B suggested `-13.6%` elapsed time, but randomized interleaving invalidated that as a global claim: current was `+2.86%` slower in elapsed median and `-2.78%` lower in completion throughput, with bootstrap intervals containing zero.
- The benchmark now defaults to explicit greedy sampling (`temperature=0.0`); seven validation trials all produced 288 completion tokens and 4/4 successful requests.
- Resource-aware validation now records platform, Python, MLX-LM, total memory, RSS peak, and swap before/after values; seven trials showed zero swap growth.
- The benchmark harness now includes staggered arrival and request-level latency diagnostics; the scheduler candidate is not retained.
- Exact prefix reuse is now measured separately from divergent LCP reuse; 9B produced one exact hit for the repeated workload and safely skipped divergent LCP matches for `ArraysCache`.
- The 9B long-context probe completed at 8,181 prompt tokens with 1.61 GiB swap growth, while 12,181, 16,181, and 30,181 token probes were rejected by memory pressure.
- Hybrid-attention accounting removed the false 9B admission rejects: fresh 12K, 16K, and 30K prompts all completed.
- Bounding automatic prefix snapshots reduced 30K stores from 53 to 8 at cap 4K and to 1 at default cap 0; the cap-0 run took 92.6s, reached 1.38 completion tok/s, and added 1.10 GiB swap.
- Prefill memory is now separately observable: default cap-0 12K measured 9.121 GB peak / 6.866 GB active; 30K measured 12.124 GB peak / 6.149 GB active. Overall request peaks matched prefill peaks.
- 4-bit/8-bit MLX KV prototypes were evaluated and not adopted: 4-bit changed fixed greedy output, while 8-bit showed no material gain in the measured 12K trial.
- Native KV growth step 2048 reduced single-trial 12K latency to 33.2s and 30K latency to 79.8s with exact greedy smoke parity; peak memory was unchanged, so this is an allocation-copy optimization rather than a paged-KV solution.
- The experimental paged KV adapter now writes full-attention K/V into fixed blocks with reference-counted table forks and COW; Qwen3.5-0.8B 2K chunked prefill matched native logits exactly (`max_abs_logit_difference=0.0`).
- The adapter's contiguous materialization fallback did not clear the 3% performance gate: 2K median was `1.29%` slower and 8K median was statistically flat (`0.03%` slower); it is not enabled by default.
- A block-indexed `mx.fast.metal_kernel` now consumes persistent physical block pools and logical block indices with GQA and causal offsets. A tiled SIMD path reduces duplicate softmax work and reaches Qwen3.5-shaped FP16 parity at or below `3.1e-05` max absolute difference for 512/2K/8K probes. The corrected dispatch benchmark is still slower than native attention: median ratios were `1.56x`, `3.42x`, and `7.44x` in the recorded run, so it remains disabled.
- The persistent pool removes per-call `mx.stack` packing and preserves per-layer COW data when a shared block table forks. It is an experimental storage boundary; pool capacity and release lifecycle are not yet integrated with serving.
- Package refresh on 2026-07-14: `uv lock --upgrade` resolved 72 compatible packages, including `mlx 0.32.0`, `mlx-lm 0.31.3`, `mlx-audio 0.4.5`, `fastapi 0.139.0`, `numpy 2.5.1`, `uvicorn 0.51.0`, and `transformers 5.12.1`. `transformers 5.13.1` remains excluded by the current `mlx-audio` and project bound `<5.13.0`. `pydub` and `python-multipart` were added to `pyproject.toml` after a locked sync exposed that they were only present in `requirements.txt`; `pip check` and `uv lock --check` pass.
- BatchGenerator audit: with the installed `mlx-lm 0.31.3` API, the experimental `BatchedEngine` completed a 4-request 0.8B smoke and cleaned up cancellation. Prefix reuse is not correct yet: a second identical 196-token request recomputed its prompt and reported `prefill_cache_hit=false` because the current adapter does not pass stored caches into `BatchGenerator.insert()`; response cache flags are also hardcoded false. No serving change was made.
- BatchGenerator prefix restore is now implemented in the experimental engine: prompt-boundary cache extraction, cloned `caches=` insertion, correct cached-token history, terminal pin release, and response cache flags. The 0.8B 4-request repeated-prompt matrix improved hot median throughput from `204.0` to `261.6 tok/s` (`+28.2%`) and reduced elapsed from `1.255s` to `0.979s` (`-22.0%`) with identical hashes and unchanged `1.486 GB` peak. Exact 12,295-token reuse measured `5.725s -> 0.484s` on 0.8B and `35.755s -> 3.375s` on 9B, both with zero swap delta and exact greedy parity. Cancellation and streaming probes left zero pinned entries.
- Iteration 028 hardened the experimental BatchGenerator path after finding hybrid-cache batch invariance failures: active requests are now grouped by prompt length and cache profile. The corrected 0.8B 30-record on/off matrix had exact response-hash parity, zero errors, and zero swap delta. Warm cache-on elapsed improved `9%~34%` across reuse, mixed, divergent-reuse, staggered, and long workloads at concurrency 2/4/8. Structured JSON output passed at concurrency 2/4, and the 8-request cancellation probe left zero running/pinned entries. This is a conservative profile guard, not unrestricted continuous batching.
- Iteration 029 added opt-in bounded per-profile BatchGenerator lanes with `engine.batch_generator_max_lanes`, default `1`. Lane limit `2` preserved exact hashes for simultaneous mixed requests and improved their elapsed time by `2.90%~5.78%`, with unchanged `1.495 GB` peak and zero swap delta. Real staggered arrival still changed batch membership and produced hash drift in all four records, so lane `2` remains experimental and is not a default performance claim. Structured output, prefix reuse, cancellation, follow-up, and lane cleanup passed their probes.
- Iteration 030 added an opt-in cohort window and lane sealing. With lane `2` and a `160ms` window applied only to isolated secondary lanes, the 8-record mixed/staggered matrix restored exact hash parity, zero errors/swap, and `1.495 GB` peak; elapsed improved `0.19%~4.99%`. Staggered p95 increased `9%~12%`, so the default remains one lane. Multi-lane configurations without a positive window are now rejected.
- Iteration 031 added explicit cohort target sizing and longest-lane step quanta. With lane `2`, a `160ms` window, target size `3`, and quantum `2`, all 8 mixed/staggered records retained exact hashes against both lane `1` and quantum `1`, with zero errors/swap and `1.495 GB` peak. Staggered p95 improved `3.82%~6.42%` versus quantum `1`, but elapsed remained `18.13%~18.77%` slower than lane `1`; the controls remain opt-in and defaults are unchanged.
- A production-shaped 0.8B manual-runtime baseline completed without swap growth: 2,229 prompt tokens took `2.638s` at `48.52` completion tok/s with `1.677 GB` MLX peak / `0.999 GB` active; 8,373 prompt tokens took `5.279s` at `24.25` completion tok/s with `2.297 GB` peak / `1.277 GB` active. These are baselines, not an optimization claim.
- Paged KV lifecycle probing showed `2,097,152` pool bytes retained after a child fork was released, then `0` pool bytes and `0` manager allocated blocks after the source bundle was released. After `mx.clear_cache()`, active MLX memory fell to `16` bytes in the isolated probe.
- An opt-in hybrid prompt-cache boundary now preserves Qwen3.5's `ArraysCache + KVCache` list shape, deep-copies recurrent state on fork, and releases full-attention pools during request cleanup. Native and opt-in greedy parity matched exactly for a 10-token prompt and 32-token completion.
- The same-model opt-in A/B did not pass the performance gate: at 2,229/8,373 prompt tokens elapsed time regressed by `19.9%/39.0%`, peak MLX memory increased from `1.677/2.297 GB` to `2.285/10.681 GB`, and both paths completed successfully without swap growth.
- Reusing a geometrically grown contiguous fallback removed the repeated full-table concatenation: the 8,373-token opt-in path fell to `5.420s` and `2.471 GB` peak in a single run. Randomized 3×3 A/B measured native median `5.448s` versus paged median `5.425s` (`-0.4%`, below the 3% gate), with paged peak memory still `2.471 GB` versus native `2.297 GB`.
- Lazy pool promotion now keeps the opt-in serving path storage-only until a block-indexed consumer requests `block_pool()`. The 8,373-token randomized 3×3 A/B measured native median `5.4541s` versus paged median `5.4526s` (`-0.03%`, below the 3% gate); peak MLX memory was `2.297 GB` versus `2.374 GB` (`+3.38%`). Greedy output parity and zero swap delta held across the probe.
- Step-bounded fallback growth removes the final 8K chunk's geometric overshoot: native KV ended at capacity `10240`, while paged materialized capacity ended at `8373`. Randomized 8K 3×3 A/B now measures native `5.4353s` versus paged `5.4259s` (`-0.17%`), with peak memory `2.297 GB` versus `2.286 GB` (`-0.46%`). This clears the memory regression but remains below the 3% speed gate.
- The paged Metal boundary now partitions long KV scans across 32 simdgroups and reduces partial online-softmax states. Qwen3.5-shaped kernel medians are `0.880x`, `0.976x`, and `0.733x` of native at 512/2K/8K tokens, with max absolute differences `0`, `0`, and `3.05e-05`. This is a kernel-level result only; serving still uses contiguous MLX-LM SDPA.
- An explicitly disabled Qwen3.5 direct-attention bridge now uses the pool kernel for decode (`Q<=8`) and native SDPA for long prefill. Randomized 8K direct/native A/B measured `5.4561s/5.4423s` (`+0.25%`) and `2.286/2.297 GB` peak memory (`-0.46%`), with exact greedy parity and zero swap delta. It is functional and memory-neutral, but not a speed win.
- The direct bridge now reuses a cached `uint32` block-index tensor until block-table or COW topology changes. A fresh randomized 8K 3x3 A/B measured native/direct elapsed medians `5.4306s/5.4597s` (`+0.54%`) and completion throughput `23.570/23.445 tok/s` (`-0.53%`), with unchanged peak memory `2.297/2.286 GB`; all six requests completed and the result does not clear the 3% speed gate.
- Native manual decode now keeps a persistent merged cache across stable multi-request steps, returning lightweight per-request references and remerging only after batch membership changes. In a 0.8B no-prefix 4-request A/B, batch=4 improved from `19.460` to `29.476 tok/s` (`+51.5%`) and reduced elapsed median from `26.310s` to `17.371s` (`-34.0%`) with unchanged `1.829 GB` peak and zero failures/swap growth. Randomized 9B 2x2 A/B measured batch=2 at `13.576 tok/s / 37.715s` versus batch=4 at `23.247 tok/s / 22.025s` (`+71.2%` throughput, `-41.6%` elapsed); peak memory was `6.256/6.220 GB`, with zero failures and zero swap delta. Greedy response hashes, token counts, and finish reasons matched batch=1 exactly; mixed and staggered membership-change probes completed 4/4.
- A controlled native prefill-batching experiment was rolled back: 0.8B 4x8K baseline elapsed was `17.390s` at `29.442 tok/s`, while batch=4 prefill was `23.423s` at `21.859 tok/s` with `12.886 GB` peak and `0.93 GiB` swap growth; batch=2 was `20.892s` at `24.507 tok/s` with `3.282 GB` peak. The root cause is activation and merged-cache memory scaling with batch times chunk size; simple prefill batching does not pass the speed or memory gates.
- A corrected cache-only microbatch probe found a narrow performance region: batch=4 with 128/256-token chunks had lower isolated model time than four serial calls, while 512/1024-token chunks had no stable gain. End-to-end 0.8B prefill256 A/B improved elapsed `22.818s -> 20.182s` (`-11.5%`) and throughput `22.439 -> 25.369 tok/s` (`+13.1%`), with peak memory `1.662 -> 1.931 GB`; however, greedy parity failed for one of four prompts (different text SHA). A 9B one-shot prefill256 probe improved `25.355s -> 24.358s` but raised peak `5.997 -> 6.622 GB`; no swap growth. The implementation was rolled back because deterministic correctness is a hard gate.
- Direct benchmark records now include MLX allocator peak memory; the 9B single smoke measured 5.169 GB and the 12K run measured 12.187 GB.
- `powermetrics` is unavailable without superuser privileges. `memory_pressure` reported 58% system-wide free memory and no thermal/performance warning was recorded by `pmset`.
- Iteration 032 evaluated a separate MLX stream per opt-in BatchGenerator
  lane. The matched 8-record A/B had zero errors, exact hashes, zero swap
  growth, and unchanged `1.495 GB` peak MLX memory, but elapsed improvement
  was only `0.84%~2.53%`. Interleaved reruns did not clear the 3% gate and
  staggered p95 once regressed `4.79%`; the candidate is not promoted.
- Iteration 033 evaluated event-driven cohort closure and rolled it back:
  mixed elapsed improved `0.89%~2.77%`, but staggered elapsed regressed
  `24.50%~27.73%`, p95 `7.98%~11.22%`, and completion throughput about 20%.
  The cause was repeated single-request lanes after staggered arrivals.
- Iteration 034 evaluated a greedy batch-wide argmax path and rolled it back:
  six candidate runs were `1.07%` slower and `1.02%` lower throughput than
  three baselines at manual decode batch 4.
- Iteration 035 keeps the disabled-prefix chat reuse guard: 40-turn Agent
  encoding improved `73.136ms -> 1.787ms`, and five end-to-end runs improved
  median elapsed `2.7199s -> 1.8380s` with identical output hash and finish
  reason.
- Iteration 036 keeps a bounded chat prompt token/reuse-point LRU. On a
  prefix-enabled 40-turn Agent workload, repeated encode time improved
  `74.082ms -> 0.028ms`; fresh-process e2e medians improved `29.8%` for exact
  hot reuse and `6.1%` for append-only turns, with cold latency `1.6%` lower.
- Iteration 037 keeps only the most recent eight chat snapshot reuse points by
  default. In a fresh-process 40-turn Agent A/B, cold latency improved
  `2.5375s -> 1.8549s` (`-26.9%`), while exact/append/branch hashes and cache
  hits remained identical. Snapshot memory fell from `1.192 GB` / 39 entries
  to `0.326 GB` / 9 entries (`-72.7%`), with zero swap growth.
- Iteration 038 adds a sparse older tier only for prompts at least 2048 tokens.
  In an 80-turn Agent A/B, cold latency improved `3.6749s -> 2.2592s`
  (`-38.5%`), initial snapshot memory fell `3.092 GB -> 0.628 GB`
  (`-79.7%`), and exact/recent/mid/old branches all retained cache hits and
  output hashes. A 40-turn probe stayed on the recent-only path, avoiding the
  short-chat memory and latency regression seen in the first sparse trial.
- Iteration 039 skips full-prompt snapshots for non-exact prefix-hit branches.
  In a sustained 80-turn / 12-branch A/B, post-recovery snapshot memory fell
  `1.511 GB -> 0.739 GB` (`-51.1%`) with identical hashes, hits, and saved
  tokens. Randomized 4-seed pairing showed branch deltas of `-0.17%~+0.49%`,
  append `+0.63%`, and recovery `+0.51%`; the initial grouped branch variance
  did not reproduce. I081/I082 supersede only its exact-hit refresh detail:
  exact hits now reuse the lookup-owned clone and pin without a second store.
- Iteration 040 resets the active priority to core manual-runtime work. A
  prefix-off Qwen3.5-0.8B baseline at concurrency 4 measured `5.307s` and
  `31.16` average generation tok/s for the 4,820-token long workload, with
  `1.626GB` peak MLX memory and zero swap growth. DFlash remains reference-only.
- Iteration 042 adds an OMLX-inspired transient full-attention prefill guard.
  Unit and fake-runtime lifecycle coverage verify chunk clamping, preflight
  rejection before model execution, output completion under pressure, and
  single-runner-thread ownership. Prefix-off 0.8B smoke probes for long and
  mixed workloads at concurrency 1/4 completed with zero failures, zero
  admission rejections, and zero swap growth. Their single-run timings are
  not treated as a throughput claim because no interleaved guard-off control
  was run.
- Iteration 043 found and fixed an idle-fast-path bypass that could expand a
  safe 1024-token chunk into an unsafe long tail. In two controlled Qwen3.5-9B
  30,181-token greedy pairs against `9e64a98`, output SHA, token count, and
  finish reason matched exactly; median peak MLX memory fell `11.378 -> 10.977
  GB` (`-3.53%`), with median total latency `89.753 -> 89.241s` (`-0.57%`).
  Candidate runs used 28 prefill steps versus 27 and added no swap; this is a
  long-context resource optimization, not a general throughput claim.
- Iteration 045 reduced the 256-entry / 8,192-token divergent-branch lookup
  median from `4.984 ms` to `0.228 ms` (`-95.4%`, `21.8x`) while retaining
  longest-prefix behavior. Exact and ordinary prefix medians changed only by
  microseconds. A same-shape Rapid-MLX trie probe measured `0.247 ms` for the
  divergent branch, but ownership semantics differ, so this is an index-level
  cross-check rather than an engine-level performance claim.
- Iteration 046 reproduced vllm-metal commit `4c18ee0` on this Apple M5 with
  MLX 0.32.0. Its full split-KV correctness matrix passed `19/19`, including
  FP16/BF16/FP32, mixed lengths, high-occupancy gate-off, sliding windows,
  fully masked partitions, and TurboQuant. Same-binary runtime A/B found the
  adaptive split slower at 8K by `7.27%`, `30.93%`, and `69.63%` for batch
  1/2/4; gate-off batch 5/8 controls were within `-1.83%/+0.22%`.
- In the same Qwen3-shaped 2K/8K kernel matrix, Aster's existing block-indexed
  kernel retained max absolute error at or below `6.10e-05`. At 8K it was
  `0.67%~25.33%` faster than MLX native SDPA and `6%~32%` faster than the
  vllm-metal single-pass kernel across batch 1/2/4/5/8. Layouts differ, so this
  is a kernel boundary result, not an end-to-end engine claim.
- A standalone MLX C++ Primitive around the same Aster attention math passed
  18 boundary corners and every timed 2K/8K batch shape against native MLX;
  maximum absolute errors were `6.10e-05` and `7.63e-06`, respectively. The
  Metal guard prevented invalid physical-pool access and the harness rejected
  its NaN sentinel. Five 30-warmup/200-measurement processes compared the
  Primitive with public, direct, and GPU-work-equivalent guarded `mx.fast`
  paths. No interval established a >=3% gain. A five-process confirmation
  reduced the 2K/batch-1 nominal gain to 2.13% and reversed 8K/batch-2 to a
  3.51% regression. Five-process 32K and 64K stress intervals also crossed
  zero, exposing the guard as a confounder in the earlier unguarded comparison.
  Probe peak MLX memory stayed at or below `268,813,526 B`, post-clear active
  memory was `16 B`, and swap delta was zero across 22 archived process
  records. No native extension was retained in the runtime.
- Fused-scatter cross-validation separated three mechanisms. Pure MLX combined
  storage established no gain and regressed directionally at 64-token batch 2.
  The exact vllm-metal `reshape_and_cache` Primitive was byte-identical for
  FP16/BF16/FP32 and cleared the gate at 1/4/8/16/64/128 tokens; 64/128 improved
  by `8.35%/11.65%`. An Aster-layout Primitive preserved exact complete-pool
  parity across repeated start/end/rotated writes, alias lifetimes, and two
  lazy chained calls, while rejecting real/spoofed invalid Python types and
  overlapping buffers before dispatch. Its 64-token single-request confirmation
  was only `0.85%` faster and crossed zero; batch 4/8 instead confirmed
  `7.10%/8.22%` regressions. A 1,000-iteration matrix peaked at `52,428,824 B`,
  archived zero post-loop error and swap growth, and recorded no thermal
  warning. No scatter change was admitted.
- Real-model paged profiling captured 40 primary records plus 20 confirmation
  controls for Qwen3.5-0.8B. Native/direct token IDs and text hashes matched,
  and swap stayed flat. At 2,229 tokens, elapsed changed `1.1005 -> 1.1044s`
  and generation `118.83 -> 118.64 tok/s`; at 8,373 tokens, elapsed changed
  `2.5783 -> 2.5838s` and generation `109.25 -> 111.00 tok/s`. All intervals
  crossed zero. Decode profile medians put sampled-token synchronization at
  about `6.97 ms` per token, model enqueue at `0.67 ms`, and post-step cache
  evaluation at `0.364 ms`.
- The pinned OMLX/mlx-vlm TurboQuant path passed `51/51` reference tests and
  compressed isolated FP16 K/V by `3.94x`. Five-process 2K/8K/32K/64K
  testing retained fused/dequant error <=`1.53e-5` and zero swap, but only
  beat Aster's slower experimental paged kernel. Against default MLX it was
  `3.47%` slower at 2K, inconclusive at 8K, `34.13%` slower at 32K, and
  `25.08%` slower at 64K.
- Full-model 4-bit TurboQuant reduced total hybrid cache by `1.72x` at 2K and
  `2.67x` at 8K. It failed admission across five distinct corpus windows:
  only 3/5 greedy windows matched at each context, minimum teacher top-1 was
  `89.06%/93.75%`, absolute PPL change reached `7.49%/3.38%`, and decode
  regressed `5.22%/5.72%`. Model-weight-dominated allocator peak did not
  improve materially; all swap deltas were zero, while the strict RSS interval
  no-regression gate did not pass.
- Decode cache synchronization screening used 36 fresh processes and showed
  that skipping cache-tree evaluation while retaining per-token
  `mx.clear_cache()` was flat. Periodic clearing improved paired decode
  throughput by `6.98%/8.70%` at 409/6,169 prompt tokens. A 60-process
  confirmation then established `5.10%~15.13%` medians across native/direct
  batch 1/2/4, with every 95% speed interval above 3% and all RSS/MLX/swap
  gates passing.
- A fixed 512-scheduler-step policy was rejected after batch-4 long stress
  accumulated `481.42 MB` of allocator free-cache within one interval. The
  retained 512-generated-token budget clears batch 1/2/4 every 512/256/128
  steps. Twenty fresh token-budget confirmation processes measured
  `+11.94%/+15.27%` for batch 2/4; 4,096-token-per-lane batch-4 stress improved
  `14.87%` and held post-first-clear allocator cache to `3.05 MB`.
- Synthetic native KV WAW, recurrent sibling-state RAW, and direct paged-pool
  WAW probes completed 10,000 steps per policy with exact sampled and final
  state digests. The complete Iteration 050 archive contains 142 fresh process
  records and passes 16/16 strict artifact recomputation tests. Final
  integration approval is hash-bound to both the production bridge and the
  token-budget long-stress aggregate.
- Iteration 051's current composite admission binds 288 strict timing payloads
  plus one stop-aware structured run to current measurement and model hashes.
  The 72-record long screen retained a greedy B2 production-first lower-bound
  miss (`+1.75%`); its stronger 18-process, 1,024-step confirmation clears all
  three 3% intervals with lower bounds `+6.57%` or better. The earlier n=3 and
  failed compatibility matrices remain historical/negative evidence.
- Stop-aware structured B4 exercised active membership `4 -> 3 -> 1`; all
  four lanes produced schema-valid JSON and stopped in 17 to 58 tokens. The
  failed lane-0/unbounded-string prompt is retained as negative evidence.
- Independent review added all-Python model barriers, non-replaying
  post-sample failures, symmetric benchmark instrumentation, balanced AB/BA
  diagnostics, public-path cache reorder coverage, and a single
  `final-admission.json` gate.
- The first final-source fresh-process matrix is preserved as a measurement
  warning: all short-cell medians were positive, but unrelated desktop load
  produced isolated `-25%` and `+65%` paired excursions. The accepted gate
  uses independent KV states in one process with adjacent AB/BA calls and the
  same per-step random seed; it does not delete the noisy records or weaken
  the 3% lower-bound requirement.

## Active Risks

- The worktree still combines evidence from Iterations 051-059,
  reference-project gitlinks, mixed index paths, and unrelated existing
  changes. Strict checking has no blocker because growth is bounded against
  the immutable debt baseline, but the underlying 1,172-path inventory remains
  an ownership and reviewability risk rather than a clean Git boundary.
- The full suite is green (`561 passed, 9 skipped, 1 warning`). Long-context snapshot
  budgeting is implemented and covered, but the automatic default-config
  128K real-model reproduction remains unarchived.
- The new paged-attention benchmark randomizes A/B order and records allocator peak memory, but it is a synthetic kernel probe rather than a full model serving benchmark; failed-request allocator data and energy remain unavailable.
- Batch-size-proportional sampled-token synchronization is removed. Iteration
  056 found host post-eval work below `0.3%`, active-penalty tensorization below
  `0.7%`, and batched normalization unstable and below `1.8%`; none enters
  production. Iterations 057-059 removed the dominant JSON allowed-list copy,
  retained a one-entry exact mask cache, and reused exact EOS membership.
  LMFE prefix states still retain large `TokenList` objects and grew RSS by
  roughly 3.69/1.46 GB in short/long profiles. Ownership and lifetime are the
  active Iteration 060 question. Broader schemas, structured concurrency,
  energy, and sustained thermal behavior remain open.
- Built-in penalty history is now bounded to 20 tokens, but structured,
  thinking, and unknown custom processors intentionally retain full history.
  The 24,601-token result is scenario-specific; broader custom-processor and
  long-context B4/B8 evidence remains incomplete.
- 4-bit TurboQuant is rejected for the measured Qwen3.5-0.8B workload. A
  future 6/8-bit or capacity-only proposal must start from the archived
  token/PPL curve and independently clear default-path speed and quality gates.
- The 9B/32K mixed-agent matrix and sustained-run matrix are not yet complete; long-context prefill still incurs substantial transient memory and swap costs.
- The bounded chat snapshot policy has been measured at 40 and 80 turns, but
  sustained longer runs and more branch-diverse traces may need a different
  recent/sparse budget or retention policy.
- The tiered policy trades old-branch reuse depth for memory: in the 80-turn
  probe mid/old branches saved fewer tokens and were slower than unlimited
  snapshots, although they still hit and preserved output parity.
- Skipping branch-only full snapshots reduces sustained memory; randomized
  sustained ordering found no material branch-latency regression. Cold/exact
  deltas remain about `+2.13%/+1.34%` and should be monitored.
- Paged KV ownership, a persistent GPU block pool, and a block-indexed Metal contract are experimental boundaries. BatchGenerator exact/strict-prefix cache restore now works in `BatchedEngine`, while per-profile lanes, cohort windows, and lane priority remain opt-in because the safe multi-lane path still carries a staggered elapsed cost. Broader model/mask/batch coverage, lower-cost deterministic cohort closure, SSD tiering, KV quantization, and the separate `BatchGeneratorRuntimeKernel` serving adapter remain incomplete.
- MLX C++ extensions bind to a private ABI and a matching nanobind type-registry
  version. The attention and Aster-layout scatter reproductions did not justify
  that packaging cost; do not add it unless a future operator proves both a
  stable local gain and a real-model benefit that cannot be achieved through
  public MLX APIs.

## Next Priority

1. Execute I086's benchmark-only shared-prefix full-attention feasibility work.
   Prove block/refcount lifetime, private suffix writes, request-owned
   `ArraysCache`, native fallback, and absence of B-by-prefix materialization
   before any 9B B2/B4/B8 A/B or default-path proposal.
2. Reduce the immutable workspace debt only through owner-attributed review
   boundaries. Do not grow generated caches or exceed the recorded +25/+20
   allowances.
3. Archive a fresh automatic default-config 128K snapshot-cap reproduction.
4. Close evidence gaps in this order: broader schemas and tool calls, 32K
   mixed-agent and cancellation pressure, 30-minute stability, then energy and
   thermal behavior. Keep native MLX attention as production and DFlash
   deferred until a measured candidate clears the same gates.
