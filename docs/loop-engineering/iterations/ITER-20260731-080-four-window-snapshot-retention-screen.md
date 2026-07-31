# Iteration 080: Four-Window Snapshot Retention Screen

- Date: 2026-07-31
- Phase: rejected
- Baseline commit: `d69557e1b1801cf47619b2bbe2978d36e356e661` plus the
  SHA-bound I077 observer and I079 arrival-plan surface
- Scope: public source-diversity screen for temporary 4 GiB retention; no
  production cache default or policy change.

## Problem

I079 proves one ordered six-key chain fits 4 GiB, but it selected that budget
from the same chain's trace. Reusing the favorable pair or extrapolating to
other prompt-size mixtures would be selection bias. A new source-diverse and
process-order-balanced screen is required.

## Hypothesis

Across four new disjoint six-key QMSUM windows, temporary 4 GiB rows will match
their 8 GiB controls, retain all six snapshots, and preserve exact zero-prefill
first-record replay without eviction or preflight skip.

## Predeclared Work

1. Add a non-negative opt-in QMSUM start offset to
   `capacity-replay-six`. Default zero must reproduce I079's plan exactly; an
   out-of-range six-record window must fail before execution.
2. Freeze four disjoint windows outside I079 before running the model. Run each
   budget in a fresh process with budget order `8/4, 4/8, 8/4, 4/8`.
3. Reuse the locked I079 contract: manual runtime, concurrency 2, greedy
   8-token output, decode-aware prefill 512, trace capacity 64, disabled
   persistence, and 120-second per-request timeout.
4. Require exact paired source, plan, workload ID, completion count, finish,
   output-token hash, and text hash. Every row must end with zero active state.
5. Each 4 GiB row must retain six snapshots at or below budget, record zero
   store/reservation evictions and preflight skips, accept every bounded
   prompt-free trace event, and preserve exact zero-prefill replay.

## Decision Rule

Any candidate gate failure rejects 4 GiB. A 4/4 pass admits it only to a deeper
reuse-distance formal gate; I080 cannot change the production default. Timing,
RSS, MLX peak, and host-global swap are retained as context and cannot produce
a speed or memory claim from this screen.

## TDD Contract

Three focused offset tests failed before implementation because the plan
builder did not accept `qmsum_start_index`. The minimum harness-only change now
provides:

- default zero, preserving I079's first six QMSUM identities;
- deterministic six-record slicing for a non-negative start index;
- explicit negative, insufficient-tail, and non-six-key-scenario errors;
- dependency-executor evidence that emitted event workload IDs stay inside the
  selected source window.

The complete arrival-load test file passes `17` tests and Ruff passes for the
harness and its tests. No engine or cache-policy source changed in I080.

## Frozen Matrix

Public-source verification passed before model execution. Offsets `6`, `12`,
`18`, and `24` select 24 globally unique QMSUM records; each plan contains six
distinct records followed by exact replay of its first, and all dependency
chains are complete.

| Process order | Window offset | Budget | Plan SHA-256 |
|---:|---:|---:|---|
| 1 | 6 | 8 GiB | `9612ccb4d1da60c0a3474795bbd22f012f1a7093ad305b59da6ebcc300e6ee1e` |
| 2 | 6 | 4 GiB | same plan |
| 3 | 12 | 4 GiB | `bcf1309fba51b4702f3de4d7ce0090fa37688946056be1f0df4d25dcd0fc8e15` |
| 4 | 12 | 8 GiB | same plan |
| 5 | 18 | 8 GiB | `a47a0e88993c7d0fe7937c06e3669d22baf9cb900d80b09d1f73afb0197a8644` |
| 6 | 18 | 4 GiB | same plan |
| 7 | 24 | 4 GiB | `9908df5d2d63185e8bbb8237da32e5c130035841656ecbd0d84e685dac9f7a81` |
| 8 | 24 | 8 GiB | same plan |

The source-verification record SHA-256 is
`a788475a8f3a5b4051eb9a0cb4db9487af8479da8f2e4a7d0396a39d0184f71a`.
No I079 record appears in these four windows. The matrix order and gates are
frozen before process 1 starts; no failed or inconvenient row may be replaced.

## Results

All eight fresh processes completed in the frozen order. Every pair matched
source, plan, execution settings other than budget, and all seven terminal
workload IDs, completion counts, finishes, output-token hashes, and text
hashes. Every row ended with zero running, waiting, pending, or failed state.
All replay requests were exact hits with zero prefill steps.

The 4 GiB retention gate nevertheless passed only one of four windows:

| Window | Order | 8 GiB entries / evictions | 4 GiB entries / evictions | 4 GiB evicted bytes | Candidate gate |
|---:|---:|---:|---:|---:|---|
| 6 | 8 / 4 | 6 / 0 | 5 / 1 | 673,939,456 | fail |
| 12 | 4 / 8 | 6 / 0 | 6 / 0 | 0 | pass |
| 18 | 8 / 4 | 6 / 0 | 5 / 1 | 933,396,480 | fail |
| 24 | 4 / 8 | 6 / 0 | 5 / 1 | 503,513,088 | fail |

The three candidate failures total three evictions / `2,110,849,024` bytes.
All occurred during `public-arrival:capacity-six-replay-0`: exact lookup first
hit and cloned the requested snapshot, then the full-prompt checkpoint path
reserved another two-clone slot and evicted a different unpinned entry before
replacing the same logical key. This leaves five retained entries even though
the replay itself remains an exact zero-prefill hit. The source trace is a
bounded next-iteration hypothesis, not an I080 policy change.

Control reservation floors were `4,339,564,544`, `4,226,711,552`,
`4,883,644,416`, and `4,435,443,712` bytes for windows 6, 12, 18, and 24.
Three exceed 4 GiB, matching the observed rejection. All candidate events were
accepted and prompt-free, none were dropped, and preflight skips stayed zero.
Timing, RSS, MLX peak, and host-global swap remain context only.

## Verification

- Public source verification, four plan hashes, and all eight raw results are
  bound by SHA-256 in
  `artifacts/ITER-20260731-080-four-window-snapshot-retention-screen/four-window-snapshot-retention-rejection.json`.
- The focused red run failed three new offset boundaries before implementation;
  the complete arrival-load file then passed 17 tests.
- No engine or cache-policy source changed in I080. The tracked and local
  runtime configuration remain at 8 GiB.
- The affected config/engine/arrival suite passed 83 tests; the full suite
  passed `552 passed, 9 skipped, 1 warning`; touched-file Ruff, JSON
  validation, and `git diff --check` passed.
- Strict workspace checking is warning-only with 22 changed paths, four compact
  artifact files / 0.04 MiB, zero blockers, and the recorded foreign-iteration
  artifact / generated-cache warnings.

## Decision

Reject 4 GiB as a production snapshot budget. Only `1/4` fresh disjoint
windows retained all six snapshots with zero eviction, so the predeclared
any-eviction rule fails. Preserve the configured 8 GiB default, two-clone
reserve, eviction policy, entry limit, persistence, and snapshot
representation. Exact output and replay correctness do not override the
retention failure, and this matrix supports no performance or cross-engine
claim.

## Rollback

No runtime rollback is required. The 4 GiB setting existed only as a harness
override in four experiment processes. Revert the QMSUM offset argument and
its tests only if the source-diversity measurement surface is no longer useful.

## Next

I081 tests the narrower lifecycle hypothesis exposed here: an exact snapshot
lookup already touches and pins the retained entry, but the request-local
`checkpoints_created` set does not record it and decode activation stores the
same full prefix again. Start with focused exact-hit, strict-prefix, miss,
cancellation, pin/LRU, and clone-count tests. Only then rerun the three failing
4 GiB windows; no snapshot-budget default changes in that iteration.

## Non-Goals

- Do not reuse the I079 pair as one of the four screen pairs.
- Do not change clone reserve, eviction policy, snapshot representation, entry
  limit, persistence, or the tracked 8 GiB default.
- Do not make a cross-engine ranking from Aster-specific snapshot telemetry.
