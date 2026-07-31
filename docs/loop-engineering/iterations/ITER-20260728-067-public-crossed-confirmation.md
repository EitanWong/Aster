# Iteration 067: Public Crossed Confirmation

- **Date:** 2026-07-28
- **Phase:** completed
- **Scope:** repeat the pinned public `cross-engine-core` matrix with each
  shard's engine order reversed. No production inference source changes are in
  scope.

## Objective

Determine whether I066's observed input-length-dependent prefill and decode
differences survive a fully reversed engine-order repetition on the same public
source records.

## Primary Metric

For each workload and input-length bin, the paired Aster/direct-MLX-LM median
ratio for prefill throughput, decode throughput, TTFT, and end-to-end time,
reported separately for the two engine-order strata and together only when
their directions agree.

## Hypothesis

If the I066 direction is a runtime property rather than a process-order or
host-state artifact, the reversed matrix will retain the same direction within
each predeclared workload/length bin while preserving exact input and output
token parity.

## Predeclared Method

1. Add a tested matrix-order selector to
   `scripts/dev/public_engine_matrix.py`; it may only change shard execution
   order, never public source selection, rendering, token IDs, generation, or
   metrics.
2. Execute all 1,380 existing `cross-engine-core` public records on Aster and
   direct MLX-LM in fresh processes with every I066 shard order reversed.
3. Validate each matrix independently, then join the two matrices by workload
   ID and compare predeclared workload and input-length bins.
4. Report both order strata, bootstrap intervals, output-token parity, peak
   RSS, and swap. A directional disagreement rejects the bottleneck claim.

## Gates

1. Both matrices pass every I066 comparability gate with the same source lock,
   workload hash, model/Tokenizer hashes, 2,048-token prefill step, and zero
   output-token drift.
2. Every public record occurs once per engine in each matrix, and each shard is
   executed first by each engine exactly once across the two matrices.
3. No production candidate is selected unless a predeclared workload/length
   effect has the same direction in both order strata and its 95% bootstrap
   interval excludes a 3% no-op band in the relevant metric.
4. Any swap growth, metric incompleteness, or parity drift rejects the result.

## Bounded Files

- `scripts/dev/public_engine_matrix.py`
- `tests/test_public_engine_matrix.py`
- `docs/loop-engineering/iterations/ITER-20260728-067-public-crossed-confirmation.md`
- `docs/loop-engineering/CURRENT.json`, `STATUS.md`, `DECISIONS.md`, and
  `KNOWN_ISSUES.md`
- One compact tracked summary below 5 MiB; raw matrices remain ignored under
  `run/loop-engineering/`.

## Result

The reversed matrix completed all 1,380 locked public records on both engines
for 2,760 new engine-records. Together with I066, each public record was
observed in an Aster-first and a direct-MLX-LM-first stratum. The two matrices
have the same source lock and workload hash, model/Tokenizer hashes, 2,048-token
prefill step, generation contract, effective inputs, and deterministic output
token hashes. Both independent matrix validations passed every eight public
comparability gate, and neither engine grew swap.

The crossed analyzer also passed all nine join gates: both matrices comparable,
same source/workload/model/execution contract, deterministic cross-matrix
parity, every shard reversed, balanced first-engine records (1,380 for each
engine), and zero swap growth.

## Findings

- The aggregate prefill difference is directionally stable but not sufficient
  for a production selection: Aster/direct-MLX-LM is `-10.504%` when Aster
  runs first (95% bootstrap `[-10.951%, -9.868%]`) and `-7.681%` when MLX-LM
  runs first (`[-8.455%, -7.037%]`).
- In the three bins below 8,192 input tokens, prefill is consistently lower
  for Aster: `-74.005%/-71.417%` for `[0,512)`,
  `-21.017%/-18.311%` for `[512,2048)`, and
  `-11.416%/-8.882%` for `[2048,8192)` across the Aster-first/MLX-first
  strata. This is a measurement target, not a component attribution.
- Decode is stable within several shorter bins but reverses at long context.
  In `[8192,32769)`, Aster/direct decode is `-10.515%`
  (`[-11.215%, -9.381%]`) when Aster runs first and `+27.532%`
  (`[+23.074%, +31.090%]`) when MLX-LM runs first.
- `longbench-qmsum` has the clearest block-order conflict: decode is
  `-11.423%` versus `+35.203%`, end-to-end time `+10.308%` versus
  `-36.111%`, and prefill `-6.087%` versus `+81.979%` across the two order
  strata. The result is evidence of a material measurement-state interaction,
  not evidence that either engine changed behavior intrinsically.

## Decision

Reject the I066/I067 public-core data as a global engine ranking or a
production bottleneck attribution (`reject-directional-disagreement`). Keep
the order-confirmed prefill gap as a hypothesis only. No production inference
source changed in this iteration.

I068 will diagnose the repeated public `qmsum` block-state interaction using a
four-block, order-balanced ABBA design and outside-timing host/process state
trace before selecting a prefill, scheduler, SIMD, cache, or native-kernel
candidate.

## Retained Evidence

- Compact summary:
  `docs/loop-engineering/artifacts/ITER-20260728-067-public-crossed-confirmation/crossed-order-summary.json`
- Ignored raw crossed analysis:
  `run/loop-engineering/ITER-20260728-067-public-crossed-confirmation/crossed-comparison.json`
- Ignored reversed matrix:
  `run/loop-engineering/ITER-20260728-067-public-crossed-confirmation/core-matrix-reversed/`
