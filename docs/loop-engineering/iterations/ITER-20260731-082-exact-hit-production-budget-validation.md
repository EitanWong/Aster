# Iteration 082: Exact-Hit Production-Budget Validation

- Date: 2026-07-31
- Phase: admitted-for-production-commit
- Baseline: I081 working-tree candidate, engine SHA
  `0e699c09da3c5188b591432734d584b57733f60341faa838176ecdb548e55f9f`;
  no I081 production commit exists yet
- Scope: validate the exact-hit checkpoint lifecycle at the configured 8 GiB
  snapshot budget; no budget, eviction, representation, or persistence-policy
  change is planned.

## Problem

I081's three fresh 4 GiB windows all retained six snapshots after suppressing
the duplicate exact-hit checkpoint. That screen demonstrates the measured
capacity boundary, but the configured production budget is 8 GiB and the
candidate has not yet been exercised across its normal retention range.

## Hypothesis

The exact-hit predicate is budget-independent: lookup already owns the LRU
touch, cloned request state, and pin; skipping the same full-prefix store will
preserve terminal output and cleanup while eliminating redundant reservation
work at the production budget as well.

## Predeclared Work

1. Rerun fresh, source-locked six-key replay windows at 8 GiB in the same
   order-balanced process sequence used by I080, without replacing a failed
   row.
2. Compare source, plan, execution, terminal, cache, trace, and cleanup
   identity against the I080 8 GiB controls. Require exact replay with zero
   prefill and no duplicate store growth.
3. Add or run wider cancellation, persistence, strict-prefix, and repeated
   exact-hit lifecycle controls. Require zero pinned/active state after every
   terminal path.
4. Keep `snapshot_skip_full_prompt_on_prefix_hit=false` as the explicit
   rollback path and keep the configured 8 GiB budget unchanged.
5. Repeat the two-record current-source Aster/direct-MLX-LM smoke only if the
   adapter or model source changes; it is not a performance-ranking gate.

## Success Gates

- Every fresh 8 GiB row matches its locked source/plan/execution and terminal
  identities.
- Exact replay remains a one-hit, zero-prefill path with no extra full-prefix
  reservation, clone, or store beyond the cold six-key chain.
- Entries, bytes, eviction/preflight counters, trace bounds, and cleanup state
  stay within the I080 control contract; all pins and active estimates reach
  zero at terminal cleanup.
- Cancellation, persistence, strict-prefix append, and rollback-switch tests
  remain green.
- A failed gate restores the refresh behavior and keeps the 8 GiB production
  configuration untouched. A pass can authorize a small production commit;
  it still does not imply a global throughput, memory, or engine-ranking claim.

## Rollback

Set `engine.snapshot_skip_full_prompt_on_prefix_hit=false`, or revert the
I081 predicate and tests. Preserve all raw rows and the I081 artifact as
bounded lifecycle evidence.

## Results

Four fresh 8 GiB rows ran in order-balanced source windows at offsets 6, 12,
18, and 24. Each row matches its I080 8 GiB control's source, plan,
execution contract, workload count, terminal output-token hashes, text hashes,
and length finishes. Every row retains six snapshots with six stores, one
exact hit, zero evictions, zero preflight skips, six prompt-free trace events,
zero dropped events, zero replay prefill steps, and zero active/pending/pinned
state after cleanup. The raw rows and SHA-256 bindings are recorded in
`docs/loop-engineering/artifacts/ITER-20260731-082-exact-hit-production-budget-validation/exact-hit-production-budget-validation.json`.

A fresh 9B cancellation control accepted the primary cancellation, reclaimed
the cancelled prefill checkpoint, completed its deterministic follow-up, and
ended with zero active/pending state. The persistence and cancellation focused
suite passed `4 passed in 0.76s`.

## Verification

- Full suite: `554 passed, 9 skipped, 1 warning`.
- Touched-source Ruff and `git diff --check`: passed.
- The configured snapshot budget remains `8589934592` bytes; no eviction,
  representation, persistence, or cache-index policy changed.
- The current-source Aster/direct-MLX-LM two-record compatibility smoke from
  I081 remains exact on token IDs, text hashes, finishes, and swap delta. It
  is not a timing-ranking gate.

## Decision

Admit the I081 exact-hit lifecycle change for a small production commit. The
production-budget gate passed across all four fresh windows and the wider
cancellation/persistence controls remain green. This is a correctness and
capacity-retention admission only; it makes no throughput, memory-saving, or
cross-engine ranking claim.

## Next

I083 should package the admitted predicate and focused tests into a reviewable
commit, then run a short sustained exact/strict-prefix/cancellation loop after
the commit. The 8 GiB budget and existing eviction policy remain the baseline.
