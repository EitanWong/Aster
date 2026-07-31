# Iteration 075: Snapshot Budget Utility Frontier

- Date: 2026-07-29
- Phase: rejected
- Scope: a source-bound cache-budget measurement only; no default is selected
  before retention and replay utility are quantified.

## Problem

I073 proves that a one-entry capacity limit evicts a 390,103,040-byte QMSUM
snapshot, while I074 proves host-global swap cannot decide a cache policy. The
remaining actionable ownership signal is explicit cache bytes versus retained
exact-reuse utility. The configured 8 GiB budget has not been compared with a
bounded multi-key condition on locked source records.

## Hypothesis

A 1 GiB temporary snapshot budget will bound retained multi-QMSUM cache bytes,
but may evict the first snapshot before its replay. Measuring that loss against
the existing 8 GiB budget will establish whether a lower budget is viable or
must be rejected before any default change.

## Predeclared Matrix

1. Add a source-only `capacity-replay` plan: three distinct locked QMSUM
   records run sequentially, then the first record replays after the third
   completes. Prompts remain resolver-owned and are not embedded in plans.
2. Run one fresh control at the configured 8 GiB budget and one fresh candidate
   at 1,073,741,824 bytes. Both use manual runtime, concurrency 2, greedy
   8-token completions, `decode_active_prefill_token_budget=512`, and disabled
   persistence.
3. Record output identity, exact hit/miss status, retained/evicted bytes and
   entries, replay TTFT/prefill steps, lifecycle resource stages, and cleanup.

The frozen source order is QMSUM
`fdd371de2668a6f1e7914fe9a67aef33927ecc392fdc2606`, then
`ff770653bb3cc4fd9b0921c77d356e3769d01ed91292299e`, then
`fb7891826fb3f47e3750d45365e8c0c97e23613752335cc0`, followed by replay of
the first ID. The control passes `--snapshot-budget-bytes 8589934592`; the
candidate passes `--snapshot-budget-bytes 1073741824`.

## Gates

1. Every control/candidate request must have exact workload identity,
   completion-token count, output-token hash, text hash, and finish reason.
2. The 8 GiB control must preserve the first-record replay as an exact hit;
   the 1 GiB candidate must expose its actual retention or eviction outcome
   without relying on an assumed LRU sequence.
3. A lower budget is selectable only if it keeps a predeclared useful replay
   and materially bounds explicit cache bytes or evictions without any output,
   cleanup, or latency regression outside the documented tradeoff.
4. Host-global swap remains context only. Explicit snapshot bytes, cache
   counters, and terminal identity are the decision fields.

## Results

Both rows retained exact workload IDs, completion counts, output-token hashes,
text hashes, and finishes for all three distinct records and the first-record
replay. Both ended with zero running, waiting, and pending requests.

- The configured 8 GiB control retained three snapshots totaling
  1,415,217,152 bytes. Its first-record replay was an exact hit with zero
  prefill steps and `0.279007s` TTFT.
- The temporary 1 GiB candidate retained one 390,103,040-byte snapshot after
  two evictions totaling 786,628,608 bytes. It reduced explicit retained bytes
  by 1,025,114,112, but the first-record replay missed, used eight prefill
  steps, and had `19.219282s` TTFT.

The compact evidence is
`artifacts/ITER-20260729-075-snapshot-budget-utility-frontier/snapshot-budget-utility-frontier-rejection.json`;
it binds both ignored raw results by SHA-256. Host-global swap changed in
opposite directions between the two rows and is not a decision field.

## Decision

Reject a 1 GiB snapshot-budget default. Its explicit-byte reduction is real,
but it violates the predeclared useful-replay gate. The configured 8 GiB value
remains unchanged. I076 must test a wider bounded budget with a deeper
distinct-key replay plan before a lower default is considered.

## Non-Goals

- Do not change the default 8 GiB budget, entry count, eviction algorithm, or
  snapshot representation during this measurement iteration.
- Do not make a general host-memory or cross-engine performance claim.
