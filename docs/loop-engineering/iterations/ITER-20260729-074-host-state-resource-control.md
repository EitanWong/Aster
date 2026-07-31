# Iteration 074: Host-State Resource Control

- Date: 2026-07-29
- Phase: rejected
- Scope: measurement-only control for I073's host-global swap observation; no
  cache policy or runtime default is in scope.

## Problem

I073 proved exact prefix reuse, existing one-entry eviction, and cancellation
cleanup under locked public sources. Its only positive global-swap movement was
in the shared-prefix workload, but it appeared with both cache off
(+883,752,960 bytes) and cache on (+364,576,768 bytes). The harness records
`psutil.swap_memory().used`, a host-global counter, so one process per state
cannot assign that movement to Aster or to a cache policy.

## Hypothesis

An outside-timed no-request lifecycle control plus two order-balanced
shared-prefix cache pairs will distinguish ambient/warmup movement from a
repeatable cache-state-associated workload stage. If no direction repeats, the
result remains a measurement boundary rather than a cache candidate.

## Predeclared Matrix

All rows use the locked `cross-engine-core` workload, greedy generation,
concurrency 2, 8-token caps, manual runtime, a 120-second timeout, and
`decode_active_prefill_token_budget=512`. Warm-cache persistence remains off.

1. Run `idle-lifecycle` once cache off and once cache on. The engine starts,
   warms, records lifecycle snapshots, and closes without a request.
2. Run two fresh shared-prefix pairs in order `off,on` then `on,off`. Each pair
   uses the same locked QMSUM record and expects the existing exact reuse only
   in its cache-on member.
3. Record process RSS and host-global swap at creation, start, warmup,
   before/after workload, and close; none of these probes enters a timed
   request step.

The `idle-lifecycle` plan is explicitly empty. Its harness contract is covered
by nine focused arrival/load tests: it reports an engine status snapshot
without resolving a prompt or invoking `engine.submit()`.

## Gates

1. All shared-prefix rows retain exact token-ID/text/finish identity and zero
   running, waiting, and pending state after completion.
2. Each cache-on shared-prefix row reports an exact hit and nonzero reused
   tokens; cache-off rows report no snapshots or reuse hits.
3. Idle rows submit no requests, finish cleanly, and retain no snapshots.
4. A cache-policy candidate is permitted only if a cache-state resource
   direction repeats in both orders, differs from idle lifecycle movement, and
   is supported by process-owned RSS or explicit cache bytes rather than the
   host-global swap counter alone.

## Results

The first idle execution exposed an empty-plan metadata bug: the result payload
called `max()` over zero entries before writing output. A focused regression now
covers both the empty plan and its `max_output_tokens = null` representation;
nine arrival/load tests pass. The corrected six-row matrix then completed.

- Idle cache-off and cache-on submitted zero requests, retained zero snapshots,
  ended cleanly, and had zero workload-stage global-swap movement.
- All four shared-prefix rows had identical completion counts, output-token
  hashes, text hashes, and finishes. Both cache-on rows recorded one exact hit,
  10,333 reused tokens, and one 390,103,040-byte snapshot; both cache-off rows
  recorded no snapshot or reuse hit.
- The workload-stage global-swap values in order `off,on,on,off` were
  `0, 0, 0, +78,577,664` bytes. The only positive value is the second
  cache-off row. It neither repeats by cache state nor differs from idle in
  both orders.

The compact evidence is
`artifacts/ITER-20260729-074-host-state-resource-control/host-state-resource-control-rejection.json`;
it binds each ignored raw result by SHA-256.

## Decision

Reject cache-policy selection again. I074 narrows the interpretation of I073:
the global swap meter is useful host context but not a reproducible cache-state
signal. Retain `idle-lifecycle` as a no-request resource control, but alter no
cache default. I075 instead measures explicit snapshot-budget utility:
distinct-key retention followed by exact-reuse replay.

## Non-Goals

- Do not alter `snapshot_budget_bytes`, `snapshot_max_entries`, eviction
  policy, checkpoint behavior, or cache representation.
- Do not infer a global host-memory explanation from one workstation's swap
  value or make a cross-engine performance claim.
