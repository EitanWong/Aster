# Iteration 050 - Amortized decode cache synchronization

Date: 2026-07-17
Start commit: `9c84e7b`
End commit: the commit containing this record
Status: `SUCCESS`

## Problem and hypothesis

Iteration 049 measured a median `8.082 ms` Qwen3.5 decode step. Sampled-token
synchronization consumed about `6.974 ms`, while the cache evaluation that
followed it consumed about `0.364 ms`. The single-request path called
`_sample_token(...).item()` and then `_eval_cache()`, which evaluated every
hybrid cache leaf and called `mx.clear_cache()` after every token. The batch
path explicitly evaluated both logits and merged cache state, then also
cleared the allocator cache after every scheduler step.

The hypotheses were:

1. scalar `item()` or the batch `mx.eval(logits)` already evaluates the graph
   required to produce the sampled token;
2. retained KV and recurrent state can remain lazy until the next RAW use,
   matching MLX-LM's generation loop, without stale state or an unbounded
   dependency chain;
3. most measured overhead comes from per-token allocator-cache clearing, not
   from re-evaluating already materialized arrays;
4. periodic clearing must be normalized by generated tokens, not scheduler
   steps, or a large batch can accumulate excessive free-cache within one
   interval.

Admission required exact token, text, and final logical-cache byte parity;
zero swap growth; at least 3% median decode improvement with a paired bootstrap
lower bound of at least 3% in confirmation; no more than 1% RSS, active MLX,
or peak MLX regression; and 10,000-step recurrent, native KV, paged-pool, and
real-model stress.

## CodeGraph and execution semantics

CodeGraph was used before source reads. It traced the production path as:

`InferenceEngine._step_decode()` -> `ModelRunner.decode_batch_step()` ->
`_decode_single()` or `_decode_batch()` -> model/cache update -> sampling.

The previous single path sampled through `array.item()` and then traversed all
cache states through `_eval_cache()`. The previous batch path called
`mx.eval(logits)` followed by `mx.eval([layer.state ...])`. Prefill separately
calls `_eval_cache()` after a chunk and must keep doing so because no sampled
token consumes its graph.

The blast radius is limited to the manual `ModelRunner`; the engine and runtime
kernel call the same `decode_batch_step()` contract. Direct paged attention,
native `KVCache`, hybrid `ArraysCache`, merged `BatchKVCache`, logits
processors, detokenization, and fallback-to-individual decode all pass through
this boundary.

## Reference comparison

Primary and local references were fixed before the experiment:

- official [MLX lazy evaluation documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
  states that scalar `array.item()` implicitly evaluates its graph and that
  evaluating an already evaluated array is a no-op;
- MLX 0.32.0 `mx.eval`, `mx.async_eval`, and `mx.clear_cache` documentation was
  checked through Agent Reach's Exa backend;
- current MLX-LM main and local `examples/mlx-lm` both resolve to commit
  `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` and match installed
  `mlx-lm 0.31.3` cache code;
- `mlx_lm.generate_step()` evaluates sampled tokens/logprobs, does not evaluate
  cache state during decode, and clears periodically at 256 steps;
- `mlx_lm.BatchGenerator` evaluates current tokens/logprobs, does not evaluate
  cache state on each decode step, and clears every 512 scheduler steps;
- MLX-LM PR 926 reports that periodic batch clearing reduced allocator cache
  from 46.64 GB to 5.14 GB in its 4,096-token batch benchmark;
- vLLM-MLX separately evaluates cache state at prompt/snapshot boundaries but
  uses sampled/logprob synchronization during decode.

The official warning that `item()` can perform only a partial graph evaluation
was treated as a correctness risk. Qwen3.5 recurrent state can be a sibling
output rather than a direct dependency of the current logits. The experiment
therefore compared final bytes after the next-step RAW chain and a final
explicit evaluation, rather than assuming token parity alone proved state
materialization.

## Candidate screening

The first matrix used 36 fresh processes: six policies, two prompt sizes, and
three rotated runs. Every process loaded local Qwen3.5-0.8B-4bit, used greedy
sampling, disabled prefix reuse, warmed eight decode tokens, and generated 256
tokens. The prompts encoded to 409 and 6,169 tokens.

| Policy | 409-token paired TPS change | 6,169-token paired TPS change | Result |
| --- | ---: | ---: | --- |
| Skip cache eval, clear every step | -0.42% `[-0.71,-0.25]` | +1.40% `[-2.46,+1.54]` | Reject |
| Skip eval, clear every 256 steps | +7.13% `[+7.13,+8.31]` | +8.14% `[+2.76,+10.75]` | Pass screen |
| Skip eval, clear every 512 steps | +6.98% `[+6.75,+7.53]` | +8.70% `[+7.50,+9.54]` | Pass screen |
| Skip eval, clear every 2,048 steps | +7.57% `[+6.73,+8.37]` | +7.41% `[+7.39,+9.76]` | Pass screen |
| Skip eval, never clear | +5.05% `[+3.93,+7.90]` | +7.87% `[+6.92,+9.77]` | Reject memory policy |

All 36 candidate/baseline pairs had exact token IDs, text hashes, and final
cache digests, and zero swap growth. Skipping only cache evaluation was flat;
the material win came from amortizing `mx.clear_cache()`.

## Synthetic 10,000-step stress

One isolated MLX process ran three 10,000-step cases under eager baseline,
periodic-512, and no-clear policies:

- native `KVCache` repeated append/WAW updates;
- an `ArraysCache` recurrence whose new state was deliberately not a direct
  dependency of the current sampled scalar, forcing next-step RAW provenance;
- direct `PagedKVCacheLayer` pool writes across 157 physical blocks.

All nine runs produced identical sampled-token and final state digests. The
periodic native cache retained about 3.81 MB of allocator cache versus 51.14 MB
with no clearing; recurrent and paged periodic caches retained only kilobytes.
Active logical state matched baseline, RSS stayed bounded, and swap remained
zero. This validates lazy next-step consumption for the targeted RAW/WAW
surface.

## Five-pair real-model confirmation

The held-out confirmation used 60 fresh processes: baseline and periodic-512,
five adjacent A/B pairs, native/direct caches, batch 1/2/4, 512 generated
tokens, and 409/6,169-token prompts. Each process hash-bound runtime source,
configuration, safetensors, tokenizer, and chat template.

| Cell | Decode speedup median | 95% paired bootstrap | RSS upper | Step-p95 median |
| --- | ---: | ---: | ---: | ---: |
| Native b1, 409 prompt | +6.05% | `[+5.23,+7.37]` | +0.12% | -3.60% |
| Native b1, 6,169 prompt | +6.95% | `[+6.49,+7.97]` | +0.05% | -1.91% |
| Direct b1, 409 prompt | +5.10% | `[+4.17,+6.16]` | +0.13% | -5.89% |
| Direct b1, 6,169 prompt | +7.03% | `[+5.22,+8.75]` | +0.05% | -2.43% |
| Native b2, 409 prompt/lane | +11.91% | `[+11.27,+12.87]` | +0.28% | -8.78% |
| Native b4, 409 prompt/lane | +15.13% | `[+11.94,+15.67]` | +0.08% | -10.92% |

Every cell passed exact token/text/cache parity, policy-count, swap, RSS,
active-memory, peak-memory, and p95 gates.

## Long stress and failed fixed-step policy

Six fresh processes tested native and direct batch-1 generation for 10,000
tokens and native batch 4 for 4,096 tokens per lane.

| Cell | Fixed-512 speedup | RSS change | Active change | Allocator-cache maximum |
| --- | ---: | ---: | ---: | ---: |
| Native b1, 10,000 | +5.58% | +0.82% | 0.00% | 166.17 MB |
| Direct b1, 10,000 | +5.06% | +0.85% | +0.0003% | 139.80 MB |
| Native b4, 4,096/lane | +13.92% | +0.09% | +0.12% | 481.42 MB |

All outputs and cache bytes matched and swap stayed zero, but fixed 512
**scheduler steps** failed the 256 MiB transient allocator-cache gate for
batch 4. This negative result changed the implementation: the budget is 512
generated tokens globally, so batch 1/2/4 clear every 512/256/128 successful
steps.

Twenty additional fresh processes confirmed the token-budget candidate:
batch 2 improved `+11.94%` with bootstrap `[+10.88,+13.29]`; batch 4 improved
`+15.27%` with `[+15.14,+15.62]`. RSS upper bounds were `+0.085%/+0.042%`.
The batch-4 long candidate improved `+14.87%`; after its first step-128 clear,
allocator cache stayed below 3.05 MB, with RSS `+0.12%`, active memory
`+0.13%`, peak memory `+0.34%`, exact cache parity, and zero swap.

## Production bridge

After implementation, 18 fresh processes called the production runner without
the policy proxy. Three runs covered the same six confirmation cells.

| Cell | Production median | Improvement over archived baseline | Delta from experimental candidate |
| --- | ---: | ---: | ---: |
| Native b1, 409 prompt | 132.19 tok/s | +10.62% | +4.32% |
| Native b1, 6,169 prompt | 119.33 tok/s | +10.82% | +3.31% |
| Direct b1, 409 prompt | 129.96 tok/s | +9.51% | +4.61% |
| Direct b1, 6,169 prompt | 115.81 tok/s | +10.07% | +3.18% |
| Native b2 | 224.12 tok/s | +15.98% | +3.49% |
| Native b4 | 346.40 tok/s | +17.90% | +2.22% |

These records were rerun after review so their manifest binds the final runtime
source. Every record matched the reference token/text/cache digests, exceeded
the experimental candidate, had zero swap and cache-clear failures, and
reported exactly 1/2/4 clears for batch 1/2/4 over 512 steps. The bridge gate
is intentionally one-sided: production may outperform the experimental
candidate but may not trail it by more than 3%.

## Runtime change

`ModelRunner` now:

1. removes post-sample cache-tree evaluation from `_decode_single()`;
2. retains one `mx.eval(logits)` in `_decode_batch()` but removes the second
   merged-cache evaluation;
3. removes per-step decode `mx.clear_cache()`;
4. counts successful generated tokens and clears after each 512-token budget;
5. resets the budget after prefill or explicit runtime cache clearing;
6. preserves a failed clear's budget so the next generated token retries it;
7. reports token budget, attempts, successes, and failures in decode
   diagnostics.

No API, model format, cache ownership, prefill behavior, or fallback contract
changed. Reverting the iteration commit restores the old eager behavior.

## Reproduction commands

```bash
ART=docs/loop-engineering/artifacts/ITER-20260717-050-decode-cache-sync

.venv/bin/python "$ART/run_matrix.py"
.venv/bin/python "$ART/aggregate.py"
.venv/bin/python "$ART/synthetic_stress.py"
.venv/bin/python "$ART/confirm_matrix.py"
.venv/bin/python "$ART/confirm_aggregate.py"
.venv/bin/python "$ART/long_stress_matrix.py"
.venv/bin/python "$ART/long_stress_aggregate.py"
.venv/bin/python "$ART/token_budget_matrix.py"
.venv/bin/python "$ART/confirm_aggregate.py" \
  --manifest "$ART/results/token-budget-confirmation/execution-manifest.json" \
  --output "$ART/results/token-budget-confirmation/aggregate.json" \
  --candidate-policy periodic-token-512
.venv/bin/python "$ART/token_budget_long_manifest.py"
.venv/bin/python "$ART/token_budget_long_aggregate.py"
.venv/bin/python "$ART/production_matrix.py"
.venv/bin/python "$ART/production_aggregate.py"
```

## Verification

- TDD RED: four decode-policy tests failed for the expected old behavior; two
  reset tests then failed for the expected counter-lifecycle gap.
- TDD GREEN: `tests/test_model_runner.py`: `33 passed`.
- Runtime/paged boundary tests: `27 passed`.
- Artifact recomputation and manifest tests: `16 passed`.
- Full suite: `466 passed, 9 skipped, 1 failed, 1 warning`.
- Excluding the pre-existing user-worktree
  `test_long_context_snapshot_budget_is_capped_for_clone_headroom` failure:
  `466 passed, 9 skipped, 1 deselected, 1 warning`.
- Ruff, `py_compile`, JSON parsing, manifest hashes, and `git diff --check` are
  required again before commit.

The archive contains 142 fresh process records across screening,
confirmation, synthetic stress, long stress, token-budget confirmation, and
production bridging. Power remains unavailable because `powermetrics`
requires elevated privileges; `pmset` reported no thermal/performance warning.

## Decision and next priority

Retain the 512-generated-token allocator-cache budget and lazy decode-cache
provenance. This is a default manual-runtime improvement with measured gains
from batch 1 through 4 and both native and direct cache paths.

Next, re-profile the post-change decode graph. In batch mode, determine whether
the remaining explicit `mx.eval(logits)` plus per-row sampled-token `.item()`
creates redundant synchronization for heterogeneous samplers and logits
processors. Do not repeat Iteration 034's rejected greedy-only batch argmax;
first measure the general sampler path and require exact structured/logits-
processor behavior under membership churn.

## Fixed loop output

LOOP ITERATION: 050
STATUS: SUCCESS
START COMMIT: 9c84e7b
END COMMIT: commit containing this record
FOCUS: Remove redundant decode cache synchronization and amortize MLX allocator-cache clearing
ROOT CAUSE: Per-token `mx.clear_cache()`, not repeated cache-tree `mx.eval`, consumed the stable multi-percent decode budget; fixed scheduler-step cadence over-retained free-cache for large batches
CHANGES: Decode relies on sampled/logit synchronization, clears every 512 generated tokens, resets on prefill/explicit clear, and exposes diagnostics
TESTS: 33 model-runner, 27 runtime/paged, 16 artifact, 466 full-suite passes plus one unrelated existing failure
BENCHMARK: Production +9.51% to +17.90% across native/direct batch 1/2/4; 10,000-token native/direct +5.58%/+5.06%
MEMORY_POWER: Exact cache bytes, RSS/active/peak gates pass, batch-4 post-clear allocator cache <=3.05 MB, zero swap; power unavailable
REGRESSION: Exact token/text/cache parity across 142 archived processes; fixed 512-step policy rejected and replaced by token budget
REFERENCE_PROJECTS: MLX 0.32.0 docs, MLX-LM 15b522f/PR 926, vLLM-MLX decode/prompt boundaries, Aster Iteration 049 profile
DECISION: Retain token-budgeted decode clearing in the production manual runtime
NEXT PRIORITY: Profile general batch sampler synchronization without repeating the rejected greedy-only argmax experiment
