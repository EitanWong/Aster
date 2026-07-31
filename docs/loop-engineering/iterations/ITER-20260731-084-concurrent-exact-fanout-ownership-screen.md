# Iteration 084: Concurrent Exact Fanout Ownership Screen

- Date: 2026-07-31
- Phase: completed
- Baseline: completed I083 exact/strict/cancellation lifecycle artifact at
  working-tree head `5e32228f2014`; the I081-I083 candidate remains uncommitted
- Scope: measurement first; no production cache representation or budget
  change is pre-authorized

## Objective

Measure whether concurrent exact-prefix fanout still pays material per-request
snapshot-clone ownership after duplicate exact-hit checkpointing was removed.
Use that evidence to select or reject a future shared-block/COW cache candidate.

## Hypothesis

Sequential exact hits are stable, but each active request still receives a
mutable clone of the retained prompt cache. A cold-plus-B2/B4/B8 public QMSUM
fanout may therefore scale transient active bytes, MLX peak, RSS, or tail
latency with the number of simultaneous hits even though the retained store
stays at one entry.

## Reference Mapping

- SGLang `RadixCache`: radix nodes protect shared KV segments with `lock_ref`
  and exclude protected leaves from eviction.
- vLLM `BlockPool`: cached blocks use `ref_cnt`; a hit touches shared blocks and
  only zero-reference blocks enter the free/eviction queue.
- Rapid-MLX: the optional radix is a lookup index beside existing storage, not
  proof that Aster should replace its already-indexed bounded lookup path.
- TraceLab (`arXiv:2606.30560`): agent sessions have long contexts, short
  outputs, and high but imperfect prefix reuse, making shared-prefix fanout a
  more representative next boundary than another serial lookup microbenchmark.

## Predeclared Work

1. Add an opt-in, compact engine-status sampler to the public arrival harness
   if current final-only status cannot capture peak active bytes and pins.
   Preserve byte-identical plans and default result schemas.
2. Run the locked `shared-prefix` scenario in fresh Qwen3.5-9B processes at
   B2, B4, and B8, with at least three repetitions per width and rotated width
   order. Keep 8 GiB snapshots, 512-token decode-aware prefill, greedy 8-token
   outputs, and one public QMSUM source fixed.
3. Require source/plan/model/generation parity, exact token/text/finish identity
   within each fanout, the expected hit count, one retained store/entry, zero
   eviction/preflight/admission failure, bounded trace, and zero terminal
   requests/queues/pins/active bytes.
4. Report median, p95, min, max, dispersion, peak RSS/MLX/active estimates, TTFT,
   end-to-end latency, aggregate throughput, and swap context by width. Do not
   infer ownership from host-global swap alone.

## Selection Gate

Advance a shared-block/COW ownership design only if repeated B4/B8 evidence
shows material clone-correlated memory or latency growth beyond noise while all
correctness gates pass. Reject that design direction if fanout remains bounded
or the signal is not repeatable. No result in I084 may lower the 8 GiB budget
or claim a cross-engine ranking.

## Implementation

- Added an opt-in `--sample-engine-lifecycle` observer to the public arrival
  harness. It records only compact maxima and a final status snapshot at a
  configurable interval; prompts, token IDs, and request payloads are not
  retained.
- Covered disabled-schema compatibility and active/final sampling with two
  tests that failed before implementation. The arrival harness now has 24
  passing tests, and the locked I080 plan remains byte-identical at
  `9612ccb4d1da60c0a3474795bbd22f012f1a7093ad305b59da6ebcc300e6ee1e`.
- A fresh B2 ABBA no-op screen compared untraced/sampled/sampled/untraced
  processes. Exact output and zero swap held. Sampled median deltas were
  elapsed `+0.855%`, replay TTFT `-2.854%`, replay latency `-0.919%`, and
  replay generation throughput `-0.183%`; all clear the absolute 3% gate.

## Locked Matrix

- Frozen plan SHA-256 values are B2 `4698d0f6fdd243681f2b3b00a1a8cd03ce7afbc82bdde63452ce1609cf36317c`,
  B4 `8b8ce42198adef0275b2d0c9839eaf9066c1c583d8f03870335565d2b96ac651`,
  and B8 `2b9641ae6d09b4c0b52659d285b397ff2845a7185ab8900abdf8ec684425ee66`.
  Every embedded plan matches its frozen file.
- Fresh-process order was `B2/B4/B8`, `B8/B2/B4`, then `B4/B8/B2`.
  Each process used Qwen3.5-9B, one public 10,334-token QMSUM prompt, greedy
  eight-token generation, the configured 8 GiB snapshot budget, and the
  512-token decode-aware prefill budget.
- All 42 matrix requests completed with one token hash, one text hash, and
  `length` finish. Every row recorded `exact_hits = B - 1`, one retained
  store/entry, zero eviction/preflight skip/error, one bounded reservation
  event, no dropped events, and zero terminal active/request/queue/pin state.

## Results

| Width | Elapsed median | Replay TTFT median / p95 | Replay latency median / p95 | Replay TPS median | Peak active estimate | Peak MLX | Swap context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | 16.747s | 0.150s / 0.152s | 0.464s / 0.473s | 22.073 | 390,397,952 B | 8.258 GB | -125,829,120..0 B |
| B4 | 17.425s | 0.361s / 0.591s | 0.985s / 1.142s | 11.345 | 1,171,193,856 B | 8.258 GB | -75,497,472..0 B |
| B8 | 20.930s | 0.952s / 3.791s | 3.913s / 6.512s | 2.954 | 2,732,785,664 B | 10.588 GB | +943,456,256..+4,565,434,368 B |

The peak active estimate is exactly `(B - 1) * 390,397,952` bytes in all nine
processes. Peak live snapshot bytes are the same value plus the single retained
390,103,040-byte snapshot. B8 also raises the reported MLX peak by 2.330 GB
relative to B4 and produces repeatable queue/tail-latency pressure. Host-global
swap is not process-owned and is retained only as pressure context.

## Decision

Advance a narrowly scoped exact-hit shared-state/COW feasibility iteration.
The evidence is repeatable, clone-correlated, and materially larger than the
observer noise while every correctness and cleanup gate passes. This does not
authorize a generic shallow copy: current `ModelRunner.clone_cache` uses
`copy.deepcopy`, and the cache types have different mutation semantics.
I085 must prove type-specific isolation through exact, append, decode,
cancellation, persistence, and fanout tests before any production-path A/B.

Keep the 8 GiB budget, reservation/eviction policy, snapshot representation,
and rollback switch unchanged in I084. The compact hash-bound result is
`docs/loop-engineering/artifacts/ITER-20260731-084-concurrent-exact-fanout-ownership-screen/concurrent-exact-fanout-summary.json`.

## Verification

- `561 passed, 9 skipped, 1 warning` in the full suite.
- The 24-test arrival harness and touched-file Ruff checks pass.
- Both tracked JSON files parse, `git diff --check` passes, and strict workspace
  audit reports WARN with no blockers.
