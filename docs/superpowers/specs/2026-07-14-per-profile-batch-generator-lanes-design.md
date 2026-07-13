# Per-Profile BatchGenerator Lanes Design

**Date:** 2026-07-14

## Goal

Allow heterogeneous prompt/cache profiles to make progress in parallel
without merging incompatible hybrid `ArraysCache + KVCache` state, while
preserving deterministic output, cancellation cleanup, and the current
manual-runtime default.

## Context

Iteration 028 proved that one `mlx_lm.BatchGenerator` must not mix requests
with different prompt lengths, cache modes, or cache offsets. The current
guard keeps one global generator and defers every incompatible request. This
preserves parity but serializes mixed workloads.

## Design

`BatchedEngine` will own a bounded set of `_BatchLane` objects. Each lane has
one `BatchGenerator`, one immutable profile key, a request/UID map, and the
requests currently admitted to it. A profile is
`(prompt_token_count, use_prefix_cache, cache_token_count)`.

The engine loop remains the sole owner of MLX execution. It schedules waiting
requests into a compatible existing lane, creates a lane when the configured
lane limit allows it, and otherwise leaves the request queued. It then calls
each lane's `next()` sequentially in deterministic insertion order. This
avoids concurrent MLX calls and keeps cancellation/error handling local to the
lane that owns the request.

`engine.batch_generator_max_lanes` defaults to `1`, exactly preserving the
current behavior. The benchmark enables `2` explicitly. Prefix cache
extraction, generator removal, response processing, and abort cleanup use the
request's owning lane rather than a global generator. Empty lanes remain
available for reuse and are closed with the engine.

## Alternatives

1. **Bounded sequential lanes (selected):** restores heterogeneous progress
   while preserving one MLX execution owner and a simple rollback switch.
2. **One asyncio task per lane:** may overlap Python scheduling, but risks MLX
   stream ownership and makes deterministic parity harder to prove.
3. **Keep the single-lane guard:** safest but does not address the measured
   mixed-workload queueing loss.

## Correctness and performance gates

- Exact token/text hash parity against the one-lane baseline for greedy
  reuse, mixed, divergent-reuse, staggered, structured, and long workloads.
- Zero request errors and zero swap growth in the matched A/B matrix.
- Cancellation, streaming, follow-up, and prefix pin cleanup leave no running
  requests, UID mappings, or pinned entries.
- Mixed-workload warm elapsed time improves by at least 3% to be considered a
  performance win; peak memory must remain within 5% unless explicitly
  accepted in `DECISIONS.md`.
- Failure keeps the default lane limit at `1` and records the candidate as
  experimental or rolled back.

## Rollback

Set `engine.batch_generator_max_lanes: 1` to restore the Iteration 028
single-generator behavior. Revert the implementation commit if lane-local
ownership or correctness gates fail.
