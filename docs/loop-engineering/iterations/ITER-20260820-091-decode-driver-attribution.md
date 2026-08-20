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

## Candidate

The experiment adds the opt-in `engine.decode_tensorized_logprobs_enabled`
switch. When enabled, processor-free multi-row decode batches compute one
batch-wide `logsumexp` graph and still invoke each row's sampler separately.
Rows with logits processors stay on the existing per-row path. The switch is
`false` by default and is a complete rollback to the baseline path.

## Evidence

The random-logit screening shape `(4, 151936)` was numerically exact and
measured a `-11.589%` median normalization time delta, but it is a kernel
screen only. The first real-model screen collected baseline before candidate in
every pair, so it is retained only as a superseded diagnostic. Repetitions 2
and 4 were recollected candidate-first; the selected 16 rows now have two
baseline-first and two candidate-first processes per cell, all with the same
source/model/workload locks.

| Cell | Baseline decode TPS | Candidate decode TPS | Median delta | Baseline-first stratum | Candidate-first stratum | Aggregate TPS delta | TTFT p95 delta | E2E p95 delta | Peak MLX delta | Peak RSS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4-short | 54.105929 | 53.248829 | -1.584% | -1.496% | -0.685% | -0.348% | -1.186% | +0.405% | 0.000% | -6.623% |
| B4-mixed | 33.850635 | 33.859096 | +0.025% | -0.610% | +0.157% | -0.898% | +1.259% | +1.235% | +2.278% | -5.626% |

The four paired decode deltas are `+1.444%/-1.713%/-4.436%/+0.344%` for
B4-short and `-0.388%/+1.387%/-0.831%/-1.073%` for B4-mixed. Every pair has
exact request keys, token IDs, text hashes, finish reasons, and clean terminal
state. The candidate path executed 9/9 tensorized steps in B4-short and 8/8
in B4-mixed with zero decode fallbacks. One B4-mixed candidate process grew
host swap by `317,587,456` bytes; the median swap was zero.

## Decision

`decode-tensorized-logprobs` is rejected. Neither cell reaches the required
3% primary improvement, the mixed order strata straddle zero, and the mixed
cell has contrary latency/resource evidence. No production default or
scheduler/cache behavior changes. The opt-in switch remains available for a
future profile only. The complete ledger, raw selected rows, source hashes,
and gate recomputation are stored in
`docs/loop-engineering/artifacts/ITER-20260820-091-decode-driver-attribution/decode-tensorized-logprobs-rejection.json`.

## Frontier Intake

The reference set now includes `mlx-swift-lm` at
`7871b09b2eda7500bc2acad51125ebd772cbaffe` (MIT), with a current Swift MTP
iterator, staged KV rounds, sticky passthrough, and acceptance telemetry, and
SpecForge at `2590f48e3a93f69a1e9e63caa23e9f2f9e07c84a` (MIT), with current
EAGLE3/P-EAGLE/DFlash/Domino/DSpark training and acceptance paths. These are
research references only: MTP/speculation remains blocked until foundation
parity and rollback gates close. LLMVisor's stage attribution model and the
resource-fair scheduling paper are carried into I092's benchmark-only
roofline attribution plan.

## Verification

The focused foundation, sampling, configuration, and artifact tests pass
(`34 passed`); the full suite passes `608 passed, 9 skipped, 1 warning`.
Touched Ruff/format and JSON parsing pass. Strict workspace audit exits zero
with no blockers; it reports expected warnings for the just-closed I091
artifact, two new reference gitlinks, and 24 generated caches. The commit hash
is recorded in the delivery update.

## Delivery

Commit `e492b44f60dc90ce271213571fb5a20bb9acd011` contains the implementation,
tests, 16-row rejection artifact, two reference gitlinks, frontier intake, and
I092 plan. It was pushed to `origin/main`. The artifact SHA-256 is
`979443e7549ded3b1d465ac8c2e93c33e6fee8e4f09da4167461c8bb2d14c5c9`.
The post-push strict audit is clean except for the existing 24 generated-cache
warning.
