# Iteration 065: Short Decode Isolated Baseline

- **Date:** 2026-07-28
- **Phase:** complete
- **Scope:** benchmark and evidence audit only; no production runtime change is
  authorized in this iteration.

## Objective

Make the I061 128-word / 256-token Aster versus direct-MLX-LM comparison
explicitly robust to I064's same-process call-position confound.

## Primary Metric

For every retained or newly collected engine observation, prove that exactly
one declared decode is timed in its isolated process after its fixed prewarm;
then report the existing balanced Aster/direct paired statistic without mixing
in same-process adjacent-comparison data.

## Hypothesis

I061's process-isolated matrix remains a valid scenario-scoped baseline because
each engine has a single timed decode after prewarm. The required work is to
verify and bind that boundary, not to reuse I062/I064 same-process throughput
numbers as an engine ranking.

## Predeclared Gates

1. Preserve I061 model hashes, raw prompt IDs, greedy sampler, output cap,
   exact completion IDs/text/finish, and non-growing swap.
2. Inventory timed-call cardinality, warmup sequence, process PID, and source
   hash for every retained I061 record.
3. If confirmation is necessary, run isolated one-timed-call processes with
   balanced Aster/direct order and retain every record.
4. Keep I064 same-process measurements as diagnostics only; they may not enter
   engine-ranking statistics or authorize a runtime candidate.

## Planned Bounded Artifact Set

- `isolated_baseline_audit.py`: record and source-boundary verifier.
- An optional compact confirmation archive only if the existing I061 evidence
  does not already prove the boundary.

## Audit Result

The existing I061 38-member archive already proves the boundary, so no new model
run was needed. The audit source parsed both `run_engine` branches and found two
generation calls per process: one discarded warmup and the second assigned to
the sole timed result.

| Boundary | Result |
| --- | --- |
| Archived engine records | 24 / 24 unique PIDs |
| Aster and MLX-LM call plan | One warmup plus timed call #2 |
| Per-scenario process order | 3 Aster-first / 3 MLX-LM-first |
| Archive and current source hashes | Match I061 admission |
| Archived pair parity | All comparable |

The I061 short result remains a scenario-scoped isolated-process comparison.
I064's same-process call-position observations are not part of its statistics.

## Evidence And Verification

- `isolated-baseline-audit.json`: source-bound audit of I061's retained
  `formal-evidence.tar.gz` (38 members, SHA-256
  `e554a596ff923b524ae567b038b46e042dfb9cbf8fda67105a0751a311613d0a`).
- `test_i065_artifacts.py`: current source/hash validation and archive-boundary
  recomputation passed (`2 passed`).
- No model process or production source was changed in I065.

## Conclusion And Next Priority

Decision: **admit** I061's process-isolated baseline boundary. I066 will profile
only Aster's one-timed-call path under that boundary to select a measured
production bottleneck; it will not retry the rejected adjacent pipeline screen.
