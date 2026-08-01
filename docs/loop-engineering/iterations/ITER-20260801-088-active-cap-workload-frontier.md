# Iteration 088: Active-Cap Workload Frontier

- Date: 2026-08-01
- Phase: experiment
- Baseline commit: `03c222ac687d8a5ffe9ef08eedbd943209285391`
- Evidence baseline: completed, source-bound I087 worktree artifact
- Scope: benchmark/selection only; no production scheduler or default change

## Objective

Map the active-request cap frontier across three materially different B8
workloads before designing a conditional policy. I087 proves cap 4 is better
than configured 16 for one exact-prefix long-context cell; I088 tests whether
that result is a local optimum or a workload-specific exception.

## Workloads

1. `exact-long`: the byte-identical I087 plan, one 10,334-token QMSUM cold
   request followed by seven simultaneous exact replays.
2. `short-simultaneous`: eight public interactive prompts submitted together.
3. `mixed`: one cold QMSUM request followed by two exact replays, one distinct
   QMSUM request, and four public interactive prompts submitted together.

All use Qwen3.5-9B, greedy eight-token output, thinking disabled, 8 GiB
snapshots, 256 entries, 512-token decode-active prefill, trace depth 64, and
`max_decode_batch=4`. The only Stage A variable is configured
`max_active_requests` in `2/3/4/5/6/16`.

## Reference Boundary

- Feather (`arXiv:2605.06046v1`) motivates workload-dependent batch width but
  does not establish an Apple-Silicon cap or transfer its heterogeneous-prefix
  gains to Aster's contiguous MLX cache merge.
- vllm-metal main `b6e35b6c` and SGLang main `e1964da4` retain separate
  scheduler/running-batch and cache-layout assumptions. They are source/design
  references; this matrix does not claim runtime ranking across incompatible
  backends.
- I087 is the exact-long confirmation baseline. Its raw records remain intact
  and its source hashes must continue to pass while I088 uses a new tool.

## Stage A Pilot

Run 18 independent processes in a thermally mixed order. For each workload,
every cap must preserve per-key token/text/finish identity, complete all
requests, retain clean terminal state, and record zero error, eviction,
preflight skip, or dropped trace event.

Report peak MLX/RSS, aggregate throughput over measured requests, p95 TTFT,
p95/max latency, completion spread, cache statistics, and active-memory
observations. A cap is eligible for confirmation only when, relative to cap 16,
it retains at least 97% throughput, stays within 1.03 for TTFT/p95/max latency,
and has no material memory regression on every workload where it is proposed.
An exact-long candidate must additionally reduce peak MLX at least 10%.

Stage A cannot admit a policy. It only selects non-dominated caps for rotated
Stage B repetition.

## Stage B Diagnostic Freeze

Stage A completed all 18 cells, but the strict cross-cap output gate rejected
the matrix: `mixed-short-3` has one token/text fingerprint under caps 2/5 and a
different fingerprint under caps 3/4/6/16. Therefore Stage B is a correctness
diagnostic, not a performance confirmation.

Run fresh cap 2/3/5/16 mixed processes. In the target request only, retain the
decode cohort for every step, the selected token, top-two logits, and logits
for candidate token IDs 364/421/8574. The instrumentation forces evaluation,
so every diagnostic record must carry `performance_measurement_valid=false`.
Admission remains closed regardless of diagnostic latency or throughput.

The diagnostic passes when it reproduces both output groups, identifies the
first divergent step after a shared token prefix, and preserves all ordinary
request/cache/lifecycle contracts. It does not justify tolerance-based token
selection or a production scheduler change.

## TDD Contract

Initial red covered the missing mixed-plan builder and frontier summarizer.
The green boundary builds the exact/distinct/short mixed plan, validates the
complete 18-cell grid, computes 3% Pareto/eligibility results, rejects an
incomplete grid, and retains output drift as explicit no-go evidence instead
of losing the valid cells to an exception.

## Results

Stage A completed 18/18 fresh processes and 144/144 requests with every
per-cell request, cache, configured-cap, and terminal-clean contract passing.
Exact-long performance-eligible caps are 2/3/4; short-simultaneous has no
eligible lower cap; mixed performance-eligible caps are 2/3/4/5/6. Thus no
global lower cap exists even before applying the output gate.

Cross-cap token/text identity passes for exact-long and short-simultaneous. It
fails only for mixed `mixed-short-3`: caps 2/5 differ from cap 16 while caps
3/4/6 match it.

Four fresh Stage B diagnostics reproduce two stable groups. Caps 2/5 select
token 364 and share output hash `98e114d3...`; caps 3/16 select token 421 and
share `5827dd24...`. All four share the first six selected tokens
`271,12646,25,357,2526,2923`. At completion index 6, caps 2/5 run the target
alone and report candidate logits within 0.125 (cap 5 is a three-way tie),
while caps 3/16 run it in a two-row batch and raise token 421 by 0.125. Every
diagnostic retains the ordinary request contract and is excluded from timing.

The retained 73,393-byte evidence artifact contains 18 compact pilot rows,
18 raw-result hashes, four diagnostic hashes, four source hashes, the complete
Pareto/eligibility calculation, and the recomputed diagnosis. Its SHA-256 is
`01160945ba026afca289a166e31d9387ea9a444de369ce97cfb0164cf152a056`.

## Decision

Final decision: `reject-output-drift`. No global lower cap clears all workload
performance gates, and mixed caps 2/5 also fail exact cross-cap output parity.
Do not change `max_active_requests`, scheduler routing, cache ownership, or
greedy sampling. In particular, do not add epsilon/tolerance tie-breaking from
one near-tie prompt because that changes ordinary argmax semantics.

I089 must compare the same single/batched boundary with direct/model-native
MLX-LM and isolate cache-merge shape from model batch arithmetic before any
determinism proposal. The production engine remains unchanged.

## Verification

- Focused I087/I088/arrival suite: 34 passed.
- Full suite: 588 passed, 9 skipped, 1 dependency deprecation warning.
- Touched-file Ruff, new-file Ruff formatting, tracked JSON parsing, retained
  source-hash verification, and `git diff --check` pass.
- Strict workspace audit exits successfully with no blockers: 18 changed paths,
  two artifacts / 0.09 MiB, zero mixed-index/reference paths, and zero debt
  growth. It warns about the completed I087/I088 artifacts relative to planned
  I089 and 24 generated cache directories.
