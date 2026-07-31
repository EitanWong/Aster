# Iteration 073: Prefix Cache Resource Boundary

- Date: 2026-07-29
- Phase: rejected
- Scope: baseline and attribution only; no prefix-cache policy change is selected.

## Problem

I071 observed process-level swap growth in B8 and shared-prefix traffic. I072
showed that one 10,333-token prefix snapshot occupies 390,103,040 bytes without
new swap in a single locked lifecycle run, while clearing the decode-aware
prefill candidate's resource gate. That result does not establish how retained
snapshots behave under multiple keys, eviction, cancellation, or sustained
shared-prefix load.

## Hypothesis

The remaining resource risk is owned primarily by prefix-snapshot lifetime and
host memory state, not by the admitted decode-aware prefill cap. A source-bound
baseline that separates cache-on/cache-off creation, reuse, eviction, and close
will identify whether a bounded cache policy is worth testing.

## Predeclared Baseline

1. Use the locked public QMSUM and MT-Bench sources with greedy, hash-only
   outputs and fixed 8-token completions.
2. Record lifecycle snapshots before engine creation, after start, after warmup,
   before/after workload, and after close for cache-on and cache-off runs.
3. Exercise one exact reuse, one distinct-key admission, and one cancellation
   path before considering any cache-size, eviction, or snapshot-representation
   candidate.
4. Require exact token/text/finish parity, zero leaked active state, and
   repeatable stage-attributed resource movement before selecting a change.

## Local Reference Signals

- The local vLLM checkout's `BlockPool._maybe_evict_cached_block()` clears
  cache-hash metadata when an allocated block reclaims a cached block, and
  emits removal metrics at that ownership boundary
  (`examples/vllm/vllm/v1/core/block_pool.py`). This motivates recording Aster
  entry capacity and eviction counters independently from aggregate RSS.
- The local SGLang checkout's `ScheduleBatch.check_decode_mem()` asks its tree
  cache to evict only the next decode-step shortfall before testing capacity
  (`examples/sglang/python/sglang/srt/managers/schedule_batch.py`). This
  motivates measuring whether capacity pressure, rather than generic cache
  size, owns any resource movement. No reference implementation is imported.

## Frozen Matrix

Each row is one fresh Aster process using
`run/loop-engineering/public-benchmarks/cross-engine-core.json`
(`d6c7fa000ec3daca7a9756f906ab997624b678bfa9128949a7630fc4a9444e46`) and
the active public source lock
(`d6d0877b452ed5627bf0fd39ebc1e59ccad6284cdb4eace27a954603a5211c16`). All
rows use manual runtime, concurrency 2, greedy 8-token completions,
`decode_active_prefill_token_budget=512`, no warm-cache persistence, and a
120-second request timeout.

| Path | Cache state | Capacity condition | Required evidence |
| --- | --- | --- | --- |
| exact reuse (`shared-prefix`) | off, on | configured default | on has an exact reuse hit; matching cache-off/on output identities |
| distinct QMSUM (`distinct-prefix`) | off, on | `snapshot_max_entries=1` | on exposes an existing capacity eviction; matching cache-off/on output identities |
| cancellation (`cancel-during-prefill`) | off, on | configured default | accepted cancellation, deterministic MT-Bench follow-up, zero active/pending state |

The harness's `distinct-prefix` plan runs the first locked QMSUM record to
completion before the second locked QMSUM record. It does not materialize
prompts in the plan and changes no cache behavior; the one-entry cap is a
temporary measurement condition passed only to the experiment process.

## Decision Gates

1. Corresponding cache-off/cache-on rows must match workload identity,
   completion-token count, terminal output-token hash, text hash, and finish
   reason. A cancelled primary is compared by cancellation code rather than a
   terminal output hash.
2. The cache-on exact-reuse row must report at least one reuse hit and nonzero
   reused tokens. The cache-off counterpart must report no cache entries or
   reuse hits.
3. The cache-on distinct-key row must report at least one eviction under the
   one-entry cap. Cache-off must report zero snapshot entries and evictions.
4. All rows must finish with `active=0` and `pending=0`; cancellation must be
   accepted and its follow-up must finish deterministically.
5. A resource direction is actionable only if lifecycle stage deltas identify
   the responsible stage consistently across the paired states. One aggregate
   RSS or swap value cannot select a cache-policy candidate.

## Results

The six fresh-process rows met the source, terminal identity, cache-behavior,
and cleanup gates:

- Cache-off/cache-on pairs had identical workload IDs, completion counts,
  output-token hashes, text hashes, and `length` finishes. The cancellation
  pairs both returned `request_cancelled` for the primary request and produced
  identical deterministic MT-Bench follow-up output.
- Exact shared-prefix reuse was observed only with the cache enabled: one
  exact hit reused 10,333 tokens and retained one 390,103,040-byte snapshot.
  The cache-off row retained no snapshots or reuse hits.
- Under the temporary one-entry condition, the distinct-QMSUM cache-on row
  recorded one existing eviction of 390,103,040 bytes and retained one
  396,525,568-byte snapshot. Its cache-off counterpart recorded no entries or
  evictions. No cache implementation behavior changed.
- Both cancellation rows accepted the cancellation and ended with zero running,
  waiting, and pending requests. Cache-on retained one 85,065,728-byte
  1,024-token cancellation checkpoint; cache-off retained none.

The resource-selection gate did not pass. The workload-stage global swap delta
was +883,752,960 bytes with shared-prefix cache off and +364,576,768 bytes
with cache on, while all four distinct-prefix and cancellation rows were zero.
The meter is `psutil.swap_memory().used`, which is host-global rather than
process-owned. With one fresh row per state and a shared-prefix increase in
both cache states, it does not establish a cache-specific lifecycle owner.

The compact evidence is
`artifacts/ITER-20260729-073-prefix-cache-resource-boundary/prefix-cache-resource-boundary-rejection.json`;
it binds each ignored raw result by SHA-256.

## Decision

Reject cache-policy selection for this iteration. Retain the arrival/load
measurement additions and the observed existing cache behavior, but change no
snapshot budget, entry limit, eviction policy, or representation. The next
bounded iteration must add an outside-timed host-state control before another
cache-policy candidate is considered.

## Non-Goals

- Do not alter the admitted I072 512-token decode-aware prefill default.
- Do not infer a global engine ranking from a scheduler-specific Aster load
  case or a short cross-engine compatibility smoke.
