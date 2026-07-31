# Iteration 069: Public QMSUM Decode Component Trace

- **Date:** 2026-07-29
- **Phase:** completed
- **Scope:** identify the source-bound portion of I068's stable Aster/direct
  MLX-LM long-QMSUM decode gap. No production inference behavior changes are
  in scope.

## Objective

Measure the decode component boundary that explains Aster's repeatable
`7.8%~8.2%` throughput deficit and `5.3%~5.5%` end-to-end increase, without
changing public inputs, token outputs, or the timing contract used by I068.

## Hypothesis

The deficit is in a recurring decode-path component rather than prefill or
ingress: Aster's cache merge/rebuild, model forward ownership, row sampling,
host materialization, or stream delivery will hold a material, repeatable share
of a source-bound decode request. The trace must distinguish this before a
production candidate is selected.

## Predeclared Method

1. Add opt-in, source-bound timing fields to the public adapters. The existing
   aggregate request metrics remain unchanged; component timestamps are
   diagnostic metadata only.
2. Preserve the locked LongBench QMSUM records, model/tokenizer fingerprint,
   greedy output contract, 2,048-token prefill step, fresh process isolation,
   and output-token parity gates.
3. Trace only semantically comparable boundaries: prefill model time, decode
   model/cache work, sampling/processor work, host materialization/delivery,
   cache merge/rebuild counts, and batch size. Do not compare an implementation
   private sub-step to an absent counterpart as a speed claim.
4. Run balanced public repeats. Admit no production candidate until the same
   component direction is stable across both order strata and all correctness,
   resource, and lifecycle gates pass.

## Candidate Rules

- If Aster cache merge/rebuild or model/cache work dominates the stable gap,
  profile that exact path before evaluating ownership or layout changes.
- If sampling/materialization dominates, profile the relevant processor class
  before tensorizing or moving work into a backend graph.
- If no component is stable, retain the result as a measurement boundary and
  do not introduce batch prefill, Gigatoken, Metal simdgroup, paged KV, or
  speculative decoding.

## Implementation

- Added the opt-in `--component-trace` path to the public engine matrix. It
  retains the existing request timing and generation path while recording a
  source-bound `decode_driver_seconds` boundary, the Aster internal decode
  subcomponents, cache counters, and batch shape.
- The direct MLX-LM adapter advances its existing `stream_generate` generator
  and removes the already-reported prompt prefill component from the common
  driver boundary. The Aster observer wraps existing decode call sites without
  adding an extra MLX evaluation.
- Focused component-trace tests cover ABBA analysis, missing-trace rejection,
  and observer restoration. Public MT-Bench smoke runs showed exact token/text
  parity and trace overhead within the 3% no-op band for both engines.

## Public Result

The full locked LongBench QMSUM matrix completed four independent ABBA blocks:
200 records x 2 engines x 4 blocks = 1,600 engine records (800 paired
comparisons). Source lock, model/tokenizer, execution contract, deterministic
cross-block output tokens, state trace, component-trace metadata, and zero-swap
gates all pass.

- Common `decode_driver_seconds` per output token is slower for Aster by
  `+8.791%` when Aster runs first (95% bootstrap `[+8.660%, +8.964%]`) and
  `+8.655%` when MLX-LM runs first (`[+8.522%, +8.773%]`).
- The unchanged aggregate result agrees: decode throughput is `-8.177%` /
  `-8.025%`, and end-to-end time is `+5.906%` / `+5.504%` by first-engine
  stratum. Prefill (`-2.438%` / `-2.021%`) and TTFT (`-0.936%` / `-1.291%`)
  remain inside the 3% no-op band.
- In Aster's internal diagnostic accounting, cache resolution (`0.004%` of
  decode driver), processor dispatch (`0.004%`), and result delivery (`0.082%`)
  are immaterial. The B1 workload has zero batch-cache merges and rebuilds.
  Model graph dispatch accounts for `7.173%`; sampling completion accounts for
  `92.550%`, but it contains the lazy MLX completion barrier and is not a
  comparable direct-MLX-LM private substep.

## Decision

No production optimization is admitted. The stable common decode-driver gap is
real, while the current source-bound trace rules out cache merge/rebuild,
processor dispatch, and result delivery as useful targets. It does not separate
the lazy forward/sample graph from its mandatory completion barrier on both
engines. I070 is therefore a measurement-only lower-level boundary alignment;
it must retain the same public inputs and outputs and may not add a new MLX
synchronization merely to make a timing label convenient.

## Bounded Files

- `scripts/dev/public_engine_matrix.py`
- `tests/test_public_engine_matrix.py`
- `docs/loop-engineering/iterations/ITER-20260729-069-public-qmsum-decode-component-trace.md`
- `docs/loop-engineering/CURRENT.json`, `STATUS.md`, `DECISIONS.md`, and
  `KNOWN_ISSUES.md`, and `CORE_REFERENCE_MATRIX.md`
- One compact tracked summary below 5 MiB; raw results remain ignored under
  `run/loop-engineering/`.
