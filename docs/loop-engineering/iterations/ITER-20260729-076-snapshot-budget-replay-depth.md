# Iteration 076: Snapshot Budget Replay Depth

- Date: 2026-07-29
- Phase: rejected
- Scope: source-bound budget utility measurement; no cache default change is
  selected before a deeper replay result passes.

## Problem

I075 shows the configured 8 GiB cache retains three selected QMSUM snapshots
(1,415,217,152 bytes) and replays the first exactly, while 1 GiB evicts that
first snapshot and fails the useful-replay gate. The smallest viable bound is
still unknown because I075 tests only three distinct keys.

## Hypothesis

A temporary 2 GiB snapshot budget may retain the first-record exact replay
across four distinct locked QMSUM records while reducing the configured ceiling
by 75%. A deeper source-bound replay screen will confirm or reject that
specific utility claim.

## Predeclared Matrix

1. Add `capacity-replay-depth` using four distinct locked QMSUM records in
   sequence, followed by replay of the first record after the fourth completes.
2. Run one fresh cache-on control at 8 GiB and one fresh candidate at 2 GiB;
   both use manual runtime, concurrency 2, greedy 8-token completions,
   `decode_active_prefill_token_budget=512`, and disabled persistence.
3. Record terminal identity, replay hit/miss, retained/evicted entries and
   bytes, replay TTFT/prefill steps, lifecycle stages, and cleanup.

The frozen source order is QMSUM
`fdd371de2668a6f1e7914fe9a67aef33927ecc392fdc2606`,
`ff770653bb3cc4fd9b0921c77d356e3769d01ed91292299e`,
`fb7891826fb3f47e3750d45365e8c0c97e23613752335cc0`, and
`e408de08efe124bbc147640da859edafe3a9ffd398d76134`, followed by replay of
the first ID. The control passes `--snapshot-budget-bytes 8589934592`; the
candidate passes `--snapshot-budget-bytes 2147483648`.

## Gates

1. All five control/candidate outputs retain exact workload identity,
   completion-token count, output-token hash, text hash, and finish reason.
2. The 8 GiB control must retain an exact first-record replay. The 2 GiB
   candidate must both stay within its explicit budget and retain that replay
   before it can be selected.
3. Any candidate eviction or replay loss is retained as contrary evidence and
   rejects the lower default. Host-global swap remains context only.

## Results

All five control/candidate outputs retained exact workload IDs, completion
counts, output-token hashes, text hashes, and finishes, with zero running,
waiting, and pending requests after each row.

- The 8 GiB control retained four snapshots totaling 1,988,067,328 bytes and
  replayed the first record as an exact hit with zero prefill steps and
  `0.226819s` TTFT.
- The temporary 2 GiB candidate finished under its configured budget at
  1,591,541,760 bytes, but incurred two evictions totaling 786,628,608 bytes.
  The first-record replay missed, required eight prefill steps, and had
  `27.859819s` TTFT.

Reading Aster's existing `_reserve_snapshot_capacity()` explains why final
bytes are insufficient to infer retention: before cloning, it reserves twice
the candidate snapshot size and evicts the store below
`effective_snapshot_budget - reserved_bytes`. I076 has only aggregate final
counters, so it does not identify the per-reservation numbers that caused the
two evictions.

The compact evidence is
`artifacts/ITER-20260729-076-snapshot-budget-replay-depth/snapshot-budget-replay-depth-rejection.json`;
it binds both raw rows by SHA-256.

## Decision

Reject a 2 GiB snapshot-budget default. Being under the final budget does not
clear the useful-replay gate when clone-reserve eviction removes the replay
entry. I077 is measurement-only: it will expose bounded per-reservation
capacity decisions before another budget candidate is considered.

## Non-Goals

- Do not alter the configured 8 GiB default, entry cap, eviction policy, or
  cache representation during this screen.
- Do not generalize one QMSUM key order into a whole-workload cache policy.
