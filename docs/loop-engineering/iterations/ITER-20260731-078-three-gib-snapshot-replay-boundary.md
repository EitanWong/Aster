# Iteration 078: Three-GiB Snapshot Replay Boundary

- Date: 2026-07-31
- Phase: admitted for wider retention testing
- Baseline commit: `d69557e1b1801cf47619b2bbe2978d36e356e661` plus the
  SHA-bound I077 observer candidate
- Scope: source-bound candidate selection from I077's recorded reserve targets;
  no production cache default or policy change.

## Problem

I075 and I076 reject 1 GiB and 2 GiB because reserve-time eviction loses the
first replay. I077 now exposes the actual decision boundary. For its four-key
8 GiB row, every pre-reservation store is below the target that would result
from a 3 GiB configured budget. The smallest measured safe candidate remains
unverified.

## Hypothesis

A temporary 3 GiB snapshot budget will perform zero reservation evictions,
retain all four snapshots, and preserve the first record as an exact
zero-prefill-step replay. It reduces the configured ceiling by 62.5% from the
8 GiB control without changing output or the existing clone-reserve policy.

## Predeclared Matrix

1. Reuse the exact I077 `capacity-replay-depth` source order: four distinct
   locked QMSUM records followed by replay of the first.
2. Run one fresh manual-runtime process with prefix cache on, 3 GiB configured
   snapshot budget, trace capacity 64, concurrency 2, greedy 8-token output,
   512-token decode-aware prefill, and disabled persistence.
3. Compare source, plan, terminal identities, cache utility, and trace fields
   against I077's immediately preceding traced 8 GiB row. Timing is context;
   this iteration selects retention capacity only.

## Gates

1. All five terminal workload IDs, token hashes, text hashes, completion counts,
   and finishes match the I077 traced row.
2. The 3 GiB row ends with four entries, bytes at or below 3 GiB, zero
   reservation/store evictions, zero preflight skips, and an exact first-record
   replay with zero prefill steps.
3. Every trace event has effective budget at or below 3 GiB, accepted status,
   no forbidden prompt/token payload, and pre-reservation bytes at or below its
   target.
4. A pass admits only a wider public retention matrix. It does not change the
   production 8 GiB default or support a global cache-policy claim.

## Results

The fresh 3 GiB process passed every gate. All five workload IDs, completion
counts, output-token hashes, text hashes, and length finishes match I077's
traced 8 GiB row. The engine ended with zero running, waiting, and pending
requests.

| Reservation | Live estimate | Two-clone reserve | Target store | Store before | Evictions |
|---|---:|---:|---:|---:|---:|
| first | 390,103,040 | 780,206,080 | 2,441,019,392 | 0 | 0 |
| second | 396,525,568 | 793,051,136 | 2,428,174,336 | 390,103,040 | 0 |
| third | 628,588,544 | 1,257,177,088 | 1,964,048,384 | 786,628,608 | 0 |
| fourth | 572,850,176 | 1,145,700,352 | 2,075,525,120 | 1,415,217,152 | 0 |
| replay | 390,103,040 | 780,206,080 | 2,441,019,392 | 1,988,067,328 | 0 |

All five effective budgets are exactly 3,221,225,472 bytes. Every store value
before and after reservation is below its target. The final store retains four
entries / 1,988,067,328 bytes, with zero store evictions, zero preflight skips,
five accepted trace events, and no dropped or forbidden-payload event.

Replay remains an exact hit with zero prefill steps at `0.165729s` TTFT. The
I077 traced 8 GiB row was `0.166854s` (`-0.674%` context-only movement). Whole
plan elapsed is `79.647944s`, peak RSS is 5,321,654,272 bytes, and workload
swap delta is zero. Timing and resources are context, not selection metrics in
this retention-only iteration.

The compact evidence is
`artifacts/ITER-20260731-078-three-gib-snapshot-replay-boundary/three-gib-snapshot-replay-admission.json`;
the ignored raw row is bound by SHA-256.

## Decision

Admit 3 GiB only as the candidate for a wider retention matrix. It preserves
the measured four-key utility while lowering the experiment ceiling by 62.5%,
but one ordered QMSUM sequence is insufficient evidence for a production
default. The tracked 8 GiB value and every cache policy remain unchanged.

## Rollback

No runtime rollback is required: 3 GiB was passed only as a process-local
harness override.
