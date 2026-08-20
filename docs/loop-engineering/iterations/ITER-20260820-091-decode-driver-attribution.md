# ITER-20260820-091: Decode-Driver Attribution

## Baseline

I090 is the frozen baseline: the current Aster and direct/model-native MLX-LM
Qwen3.5-9B B4-short/B4-mixed public cells, four independent repetitions per
engine and balanced first-engine order. Its paired median Aster
`decode_driver_tps` deficits are `+36.322%` and `+24.484%`, respectively.
The I090 B4 output divergences remain an explicit red gate, not an ignored
outlier.

## Objective

Identify one Aster-owned cost inside the common manual decode-driver boundary,
then measure its baseline/candidate delta without changing cache ownership,
greedy argmax semantics, or scheduler defaults.

## Primary Metric And Gates

- Primary metric: `decode_driver_tps`; report absolute baseline/candidate
  medians and signed percentage change.
- Secondary metrics: TTFT p95, end-to-end p95, aggregate throughput, peak MLX,
  RSS, and swap delta.
- Minimum evidence: fresh isolated processes, balanced AB/BA order, same source
  and model locks, no forced logit evaluation, exact token/text/finish identity,
  zero unexplained swap growth, and clean terminal state.
- Admission: a reproducible absolute 3% primary improvement with no correctness,
  memory, latency-tail, cancellation, or rollback regression. Otherwise reject
  the candidate and retain the baseline.

## Bounded File Set

`aster/inference/`, `scripts/dev/benchmark_foundation_parity.py`,
`tests/test_foundation_parity_benchmark.py`, the I091 artifact directory, and
the four loop-engineering state documents. No MTP or reference-shared
tie-breaking implementation enters this iteration.

## Required Closeout

The result must contain a recomputable performance ledger, an explicit
measurement-validity field, the order-stratum deltas, failed/contrary evidence,
and the pushed consolidation commit. A timing-invalidation result keeps I091
open or rejects the candidate; it cannot be presented as a speedup.
