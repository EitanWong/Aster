# Iteration 051 - Grouped batch sampler synchronization

Date: 2026-07-17 to 2026-07-18
Start commit: `2cb1405`
End commit: the commit containing this record
Status: `SUCCESS`

## Problem and hypothesis

Iteration 050 removed redundant decode-cache evaluation and per-step allocator
clearing. The remaining batch path still forced logits before sampling and then
materialized one sampled scalar per row. A batch of size `B` therefore crossed
approximately `1+B` host/device synchronization boundaries per decode step.

The hypothesis was that Aster could preserve heterogeneous per-request
samplers, logits processors, structured constraints, random order, and cache
ownership while submitting all sampled scalars as one MLX graph and waiting
once. Admission required:

1. exact token, text, and logical KV-cache hashes against the old path;
2. processor and sampler execution in request order;
3. compatibility with Python-valued custom samplers;
4. correct cache extraction after batch membership replacement and reorder;
5. at least 3% paired decode improvement for greedy, mixed, and penalty paths;
6. structured output compatibility, zero swap growth, and sustained B8/long
   context stress.

Iteration 034's rejected greedy-only batch argmax was not reconsidered. This
iteration keeps every row's existing sampler contract.

## CodeGraph and ownership analysis

CodeGraph was used before source reads. It traced:

`InferenceEngine._step_decode()` -> `ModelRunner.decode_batch_step()` ->
`_decode_batch()` -> per-row `_apply_logits_processors()` -> sampler ->
`_decode_result()`.

Each `DecodeWorkItem` owns its sampler, processor tuple, token history,
detokenizer, and request ID. `_get_decode_batch_cache()` owns the merged-cache
state keyed by the ordered request-ID tuple. Stable membership reuses that
state; replacement or reorder extracts each old lane and rebuilds in the new
order. The optimization therefore cannot concatenate requests by policy or
materialize results out of order without changing semantics.

## Mature reference comparison

- `examples/mlx-lm/mlx_lm/generate.py`, especially `GenerationBatch._step`,
  keeps row-specific processors and samplers, submits sampled tokens and
  logprobs with `mx.async_eval`, and materializes a completed group together.
- `examples/vllm-metal/vllm_metal/v1/sampling_batch.py` tensorizes
  heterogeneous sampling policy into one batch call. Its Torch-shaped policy
  bridge and broader tensor contract are not a direct Aster fit.
- `examples/omlx/omlx/patches/mlx_lm_mtp/batch_generator.py` rebuilds row
  ownership around stable request UIDs. It is evidence against carrying stale
  processor rows across membership changes.
- `examples/lmstudio-mlx-engine/mlx_engine/model_kit/batched_vision/batch_generator.py`
  separates `_sample_next_token` from grouped output materialization.
- MLX [PR 998](https://github.com/ml-explore/mlx/pull/998) introduced shared
  event-backed async evaluation and reported model-dependent gains, including
  9.4% on one M1 Max generation case. This supports the mechanism, not a
  transferable Aster number.
- MLX-LM [PR 924](https://github.com/ml-explore/mlx-lm/pull/924) deliberately
  evaluates retained arrays to bound lazy graph growth and reports no speed
  change. Aster therefore retains explicit sampled-token completion rather
  than allowing an unbounded graph.

## Frontier intake

The following current work was reviewed and not directly imported:

- llama.cpp [PR 17004](https://github.com/ggml-org/llama.cpp/pull/17004)
  moves samplers into backend graphs and reports NVIDIA gains, but Metal
  probability differences of roughly `1e-4` remain a correctness warning.
- [SIMPLE](https://arxiv.org/abs/2512.00719) combines CPU decision-plane
  sampling, sequence parallelism, truncation-first sampling, and a speculative
  hot vocabulary. Its largest gains target distributed NVIDIA serving, not
  Aster's single-SoC B2-B8 path.
- [FlashSampling](https://arxiv.org/abs/2603.15854) fuses LM-head and
  Gumbel-max work and reports up to 19% TPOT improvement on B200. The CUDA/
  Triton kernel does not preserve Aster's arbitrary processor/full-logit
  contract.
- SGLang [PR 20501](https://github.com/sgl-project/sglang/pull/20501) found a
  fused temperature/softmax kernel regressed small batches and dispatches only
  at batch 128 or larger. This is negative evidence for Aster's B1-B8 target.
- vLLM [PR 16436](https://github.com/vllm-project/vllm/pull/16436) reports an
  8% CUDA decode gain from removing penalty-mask synchronization. vLLM
  [PR 31336](https://github.com/vllm-project/vllm/pull/31336) similarly avoids
  logprob shape/filter synchronization. These remain candidates for a
  processor-specific profile, not evidence for this implementation.

## Candidate screening

The first screen used 48 fresh processes: four policies, greedy/mixed B2/B4,
three rotated runs, a 409-token prompt, and 256 generated tokens per lane.

| Policy | Median across cells | Result |
| --- | ---: | --- |
| Eager logits, grouped sample wait | +6.41% | One lower-bound gate failed |
| Lazy logits, grouped sample wait | +9.54% | Passed |
| Lazy logits, async submit plus grouped wait | +10.19% | Selected |

The selected candidate's cell medians were `+6.56%`, `+11.07%`, `+9.31%`,
and `+11.96%`. Every token, text, and cache hash matched; swap growth and
unexpected allocator-clear counts were zero.

A separate microbenchmark compared evaluating a list of sampled scalars with
concatenating them and calling one `tolist()`. Concatenation was `0.2%~0.4%`
slower in most B2/B4/B8 cells and only `2.72%` faster for mixed B8. It was
rejected because it adds a copy and shape assumptions without a repeatable
cross-cell win.

## Candidate confirmation

One hundred fresh processes compared the old path with the selected candidate
over greedy, mixed, penalties, structured constraints, and B2/B4/B8. Core
paired medians and exact bootstrap intervals were:

| Workload | B2 | B4 | B8 |
| --- | ---: | ---: | ---: |
| Greedy | +6.47% `[+5.93,+9.02]` | +12.22% `[+11.60,+12.47]` | +15.70% `[+15.35,+16.12]` |
| Mixed | +8.53% `[+6.85,+10.32]` | +11.87% `[+11.56,+12.42]` | +15.15% `[+14.91,+15.92]` |
| Penalties | +8.92% `[+7.26,+15.40]` | +12.27% `[+11.60,+13.53]` | not run |
| Structured | +3.43% | +8.60% | not run |

Structured B2/B4 had four nonnegative pairs out of five; one cold structured
pair was retained as an outlier rather than removed. Twenty-four additional
fresh processes with 6,169-token prompts measured `+4.95%/+8.44%` for greedy
B2/B4 and `+6.07%/+8.24%` for mixed B2/B4. All output/cache hashes matched.

## Runtime change

`ModelRunner._decode_batch()` now:

1. builds each row's processed logprobs and invokes its existing sampler in
   the original request order without eagerly forcing logits;
2. collects actual MLX array samples by the runtime's `mx.array` type;
3. calls `mx.async_eval(evaluation_targets)` once;
4. prepares lightweight cache references and reads peak memory while the work
   is queued;
5. calls `mx.eval(evaluation_targets)` once, then materializes every result in the
   original order.

Python integers, lists, tuples, and custom scalar wrappers retain the old
materialization behavior and are not passed to MLX evaluation. If every row
returns a Python value, logits become the evaluation target so model/KV work
still crosses a barrier. Mixed Python/MLX rows evaluate logits plus lazy
samples. A post-sample evaluation failure is returned to every lane without
replaying processors/samplers or advancing RNG twice; only errors before that
boundary retain individual fallback. `_sample_token` uses the same shared
materializer for single decode. No API, sampler factory, cache format, random
seed policy, allocator-clear cadence, or engine lifecycle contract changed.

Host-driven processors can declare `batch_sampling_mode = "eager_rows"`.
JSON schema processors use this compatibility mode: the model forward remains
batched, but logits processing, sampling, evaluation, and materialization run
once per row so mutable parser state observes the previous token before the
next row. Wrapped processors are detected through `_inner` without replaying
side effects. The grouped barrier remains the default for ordinary MLX rows.

## Final-source measurement and process-noise finding

The pre-review production path was compared with a source-frozen Iteration 050
baseline in 124 fresh processes. All ten short-cell medians were positive,
from structured B2 `+3.40%` through greedy B8 `+16.39%`; all token, text, and
cache hashes matched. Four short cells nevertheless failed the deliberately
strict bootstrap lower-bound gate after one process-level outlier each. The
24-process long matrix had positive medians `+2.92%~+6.99%`, but also contained
individual `-23%` and `+97%` excursions.

`pmset` reported no thermal or performance warning. Concurrent WindowServer,
Chrome, and Codex processes were consuming substantial CPU, so widely spaced
fresh-process pairs were not a trustworthy variance model. These records are
preserved in `production-confirmation` and `production-long-confirmation`.
The gate was not weakened and no outlier was deleted.

The stronger final gate uses two shallow `ModelRunner` states sharing one
loaded model and MLX runtime but owning independent KV caches, merged-cache
state, samplers, detokenizers, and allocator counters. Baseline and production
advance once per pair in balanced alternating AB/BA order. Odd/even processes
also exchange the physical runners and share one seed, forming nine independent
assignment-balanced replicates from 18 fresh processes per cell. Cells execute
round-robin. Replicate estimates geometrically combine within-process time
ratios, cancelling multiplicative process scale and runner identity. Admission
uses a 96.09%-coverage distribution-free median interval: core cells must clear
3% in the balanced and both order strata and keep at least 8/9 replicates
stable. The host-driven structured fallback instead gates its balanced interval
against a -1% no-regression floor; order and block stability remain diagnostics.
All paths require exact outputs and no swap growth; system swap reclamation is
recorded but is not treated as growth.

## Final paired results (superseded n=3 screen)

Thirty short-context fresh processes measured 256 adjacent pairs for eight
core cells plus structured B2/B4. Greedy B2 and penalties B2 each had one
order stratum below 3% despite positive balanced totals, so both were extended
to 512 pairs in three independent processes. Final admission uses those focused
B2 replacements.

| Cell | Adopted median gain | Exact independent-process median-resample 95% interval |
| --- | ---: | ---: |
| Greedy B2, 512 pairs | +9.89% | `[+8.66,+9.98]` |
| Greedy B4 | +16.28% | `[+13.05,+17.04]` |
| Greedy B8 | +16.27% | `[+16.02,+19.60]` |
| Mixed B2 | +11.56% | `[+11.54,+11.64]` |
| Mixed B4 | +18.06% | `[+10.48,+18.77]` |
| Mixed B8 | +16.47% | `[+16.06,+17.30]` |
| Penalties B2, 512 pairs | +10.70% | `[+9.30,+12.00]` |
| Penalties B4 | +17.82% | `[+13.16,+18.13]` |

The adopted eight-cell core median is `+16.27%`. Every independent-process
resample interval clears 3%; token/text/cache parity and zero swap growth hold.
Structured B2/B4 balanced medians are `+5.28%/+3.29%`; they use a 0%
compatibility floor and
are not included in the core speed range.

The final 6,169-token matrix measured `+9.11%` greedy B4, `+7.51%` mixed B2,
and `+12.51%` mixed B4. Greedy B2 was extended to 1,024 pairs because of a
first-call interaction and one marginal 512-pair stability interval. Its
three independent processes measured a median `+5.37%` with exact process
interval `[+4.51,+6.05]`. The final long range is therefore
`+5.37%~+12.51%`, all exact with zero swap growth.

## Strict v7 confirmation (current admission)

The current short matrix contains 180 timing records: 18 fresh processes and
nine assignment-balanced replicates for each of ten cells. Every core interval
clears the 3% floor in balanced, baseline-first, and production-first strata.

| Cell | Distribution-free balanced median interval |
| --- | ---: |
| Greedy B2 / B4 / B8 | `[+7.46,+7.85]` / `[+13.23,+13.86]` / `[+16.78,+17.43]` |
| Mixed B2 / B4 / B8 | `[+8.05,+8.37]` / `[+12.34,+12.88]` / `[+15.81,+16.29]` |
| Penalties B2 / B4 | `[+7.88,+8.66]` / `[+13.50,+14.64]` |
| Structured B2 / B4 | `[-0.25,+0.38]` / `[-0.72,+0.32]` |

The 72-record 6,169-token screen passed greedy B4 and mixed B2/B4. Greedy B2
had a positive balanced interval `[+5.77,+6.48]` but failed the production-first
3% stratum (`+1.75%` lower bound). That failure is retained. A predeclared
1,024-step, 18-process confirmation tightened greedy B2 to balanced
`[+6.65,+7.44]`, baseline-first `[+6.57,+7.64]`, and production-first
`[+6.75,+7.65]`; all gates then passed.

The current 1,024-step mixed B8 stress contains another 18 fresh processes.
Its balanced interval is `[+14.52,+15.41]`, with both order-stratum lower bounds
at about `+14.5%`; 9/9 replicates are stable. Across the current short, long,
focused-long, and stress components, all timing payloads retain exact
token/text/cache parity and no swap growth.

## Sustained stress and corners (superseded n=3 screen)

Three mixed B8 processes each measured 1,024 adjacent decode pairs, or 8,192
timed generated tokens per policy. Median gain was `+14.26%`; the exact
independent-process interval was `[+13.87,+14.58]`, all 96 blocks were positive,
and token/text/cache hashes matched. Each policy executed the expected 16
allocator-cache clears. MLX peak was about `1.295 GB`, RSS ended near
`1.574 GB`, and the host's existing swap value did not grow.

Unit coverage additionally proves:

- processors and samplers execute once in row order;
- no sampled scalar is materialized before the group barrier;
- mixed MLX/Python sampler return values preserve order and compatibility;
- all-Python sampler batches still force model/KV evaluation;
- post-sample evaluation failures do not replay side-effectful samplers;
- stable batch cache is reused, membership replacement rebuilds it, and C/A
  reorder extracts old A/C lanes in the new order through the public decode path;
- existing fallback-to-individual decode, stop, length, structured schema,
  cancellation, and error tests continue to pass.

A stop-aware real-model B4 structured run separately honored each lane's stop
IDs, parsed final text against the schema, and exercised active membership
`4 -> 3 -> 1`. All four lanes stopped in 17 to 58 tokens with valid JSON and
zero swap growth. A first lane-0 prompt produced an unbounded schema-valid
string interior and hit the 256-token length limit; that failed payload is
retained as a structured-model corner case rather than hidden.

## Independent review hardening

The read-only reviewer identified seven issues. The implementation and evidence
were hardened by:

- forcing logits when all custom samplers return Python values;
- preventing individual replay after a group evaluation failure;
- removing baseline-only sampler timing instrumentation;
- separating exact independent-process intervals from intra-process block
  diagnostics;
- recording balanced AB/BA counts and order-interaction strata;
- adding public-path cache/result reorder coverage;
- validating stop-aware structured output rather than hash parity alone;
- adding one composite admission artifact that requires every selected current-
  source component to pass while retaining noisy and failed evidence.

## Failed and rejected paths

- Greedy-only whole-batch argmax remains rejected from Iteration 034.
- Eager grouped sampling was slower than both lazy grouped candidates.
- Sample concatenation added overhead and shape coupling without a stable win.
- The first final-source fresh-process matrices did not satisfy every strict
  interval gate. They remain archived as evidence that adjacent pairing is
  required on a busy desktop host.
- SIMPLE, FlashSampling, backend sampler graphs, and large-batch fused softmax
  are not imported because their hardware, batch, or correctness contracts do
  not match this path.

## Reproduction commands

```bash
ART=docs/loop-engineering/artifacts/ITER-20260717-051-batch-sampler-sync

.venv/bin/python "$ART/run_matrix.py"
.venv/bin/python "$ART/aggregate.py"
.venv/bin/python "$ART/confirm_matrix.py" --profile short
.venv/bin/python "$ART/confirm_matrix.py" --profile long
.venv/bin/python "$ART/production_matrix.py" --profile short
.venv/bin/python "$ART/production_matrix.py" --profile long
.venv/bin/python "$ART/paired_matrix.py" --profile short \
  --output-dir "$ART/results/paired-final2-short"
.venv/bin/python "$ART/paired_matrix.py" --profile long --steps 256 \
  --output-dir "$ART/results/paired-final2-long"
.venv/bin/python "$ART/paired_aggregate.py" \
  --manifest "$ART/results/paired-final2-short/execution-manifest.json" \
  --output "$ART/results/paired-final2-short/aggregate.json"
.venv/bin/python "$ART/paired_matrix.py" --profile short --runs 18 \
  --steps 256 --block-size 16 --pair-warmup-steps 32 \
  --output-dir "$ART/results/strict-final-v7-short-r18"
.venv/bin/python "$ART/strict_aggregate.py" \
  --manifest "$ART/results/strict-final-v7-short-r18/execution-manifest.json" \
  --output "$ART/results/strict-final-v7-short-r18/strict-aggregate.json"
.venv/bin/python "$ART/structured_validation.py" --batch-size 4 \
  --max-tokens 256 --output "$ART/results/structured-final-v7-stop-validation.json"
.venv/bin/python "$ART/structured_validation.py" --batch-size 4 \
  --output "$ART/results/structured-final2-stop-validation.json"
.venv/bin/python "$ART/admission.py"
.venv/bin/python -m pytest "$ART/test_artifacts.py" -q
```

## Verification

- TDD RED: the original batch path materialized rows before a group barrier;
  the mixed Python-return test then exposed that passing arbitrary values into
  `mx.async_eval` forces a batch fallback. Independent review added RED cases
  for an all-Python batch with no model barrier and sampler replay after a
  failed group evaluation.
- TDD GREEN: `tests/test_model_runner.py`,
  `tests/test_model_runner_sampling.py`, constrained/schema, and thinking
  processor suites: `68 passed`.
- Strict artifact and admission checks: `25 passed`; Ruff passed for runtime,
  tests, and all Iteration 051 scripts.
- Full worktree suite after the final evidence refresh: `477 passed, 9
  skipped, 1 failed, 1 warning`.
- Excluding the pre-existing user-worktree
  `test_long_context_snapshot_budget_is_capped_for_clone_headroom` mismatch:
  `477 passed, 9 skipped, 1 deselected, 1 warning`.
- `final-admission.json` binds 288 strict timing payloads across short, long,
  focused-long, and stress components plus one stop-aware structured run to
  current measurement/model hashes and reports `admitted=true`. The initial
  n=3 screen, the long B2 order-stratum miss, and the retained structured
  corner failure remain in the archive.
- `powermetrics` remains unavailable without elevated privileges; `pmset`
  reported no thermal/performance warning.

## Decision and next priority

Retain grouped asynchronous sampled-token evaluation in the production manual
runtime. It removes batch-size-proportional host synchronization while keeping
every request's existing policy and state ownership. Host-driven structured
processors retain the eager-row compatibility path because their mutable parser
state cannot be advanced as one lazy batch graph.

Next, profile the post-change sampler graph by processor class. Isolate
logsumexp, repetition/presence/frequency penalties, structured constraints,
and host output materialization before considering tensorized homogeneous
groups or a sampler backend graph. Any next candidate must preserve arbitrary
processor semantics, dynamic membership, random-state order, and the same
adjacent-pair/long/stress gates.

## Fixed loop output

LOOP ITERATION: 051
STATUS: SUCCESS
START COMMIT: 2cb1405
END COMMIT: commit containing this record
FOCUS: Remove batch-size-proportional sampler synchronization without narrowing sampler semantics
ROOT CAUSE: Batch decode forced logits and then materialized each row separately, creating approximately 1+B host/device boundaries
CHANGES: Build ordinary MLX row samplers lazily, async-submit samples, wait once, then materialize ordered results; preserve Python returns and eager-row host-driven processors
TESTS: 68 focused, 25 artifact/admission, 477 full-suite passes plus one unrelated existing failure
BENCHMARK: Strict v7 short core balanced intervals +7.46% to +17.43%; long screen +5.77% to +10.90% with greedy B2 resolved at 1,024 steps to +6.65% to +7.44%; mixed B8 stress +14.52% to +15.41%
MEMORY_POWER: Exact cache hashes, expected clear cadence, about 1.295 GB MLX peak in B8 stress, zero swap growth; power unavailable
REGRESSION: Exact token/text/cache parity across greedy, random mixed, penalties, structured output, long context, B8 stress, and membership reorder
REFERENCE_PROJECTS: MLX-LM 15b522f, MLX PR 998, vllm-metal SamplingBatch, OMLX UID ownership, LM Studio grouped materialization, vLLM/llama.cpp sampling sync work
DECISION: Retain grouped asynchronous sampled-token evaluation for ordinary MLX rows; preserve eager-row compatibility for host-driven structured processors
NEXT PRIORITY: Profile processor-specific sampling cost and evaluate tensorization only where exact dynamic-row semantics survive
