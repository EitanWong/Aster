# Iteration 077: Prefix Reservation Decision Trace

- Date: 2026-07-29; completed 2026-07-31
- Phase: admitted
- Baseline commit: `d69557e1b1801cf47619b2bbe2978d36e356e661`
- End commit: pending; the compact artifact binds the uncommitted source by
  SHA-256 because repository policy requires explicit user approval to commit.
- Scope: bounded observability for existing snapshot reservation decisions; no
  clone-reserve, eviction, or cache-default behavior changes.

## Problem

I076 proves that a 2 GiB final store can be below its configured cap while the
first replay was already evicted. Existing aggregate counters report only final
entries/bytes and total evictions. The current engine pre-reserves twice the
candidate clone size and evicts before storing, but its effective budget,
reserve target, and per-call evictions are not exposed.

## Hypothesis

A bounded, prompt-free reservation trace can explain each existing capacity
decision without changing it. A no-op source-bound screen must show that the
observer retains exact output and does not materially perturb replay timing
before it is used to profile another budget.

## Predeclared Instrumentation

1. Record a bounded FIFO of reservation events outside model execution. Each
   event contains request ID, logical prefix token count, estimated bytes,
   configured/state/effective budget, two-clone reserve, target store bytes,
   store entries/bytes before and after, per-call evictions/evicted bytes, and
   accepted/skip reason. It contains no prompt text or token IDs.
2. Add focused unit coverage for accepted, reserve-eviction, and preflight-skip
   decisions, including FIFO bounds and no mutation of existing cache policy.
3. Run a source-bound traced/untraced 8 GiB four-key replay smoke. Require exact
   terminal identity, zero active state, and a <=3% replay TTFT movement before
   using the trace for a budget candidate.

## Non-Goals

- Do not modify `evict_until_below`, snapshot cloning, LRU selection, budget
  defaults, or snapshot representation.
- Do not use the observer to make a timing or cross-engine ranking claim unless
  the no-op gate passes.

## Reference Review

Read-only upstream heads were checked on 2026-07-31. vLLM head
`b2fb83e7ffbc30a1aa4667b1dad7ca3e2c342bcf` carries structured
`KVCacheEvictionEvent` samples separately from aggregate cache hit statistics.
SGLang head `fd28242b683f367dbee47736a361cc694906d067` returns a structured
`EvictResult` and emits per-call eviction metrics. Aster adopts the structured
result boundary, but keeps its existing bounded snapshot store and does not
copy either allocator or eviction policy.

The local reference pins (`vLLM 2ac1251`, `SGLang 99b29bf`) are behind those
read-only heads. They were not refreshed because I077 needed only the stable
observability pattern and no reference code change.

## Implementation

- `engine.snapshot_reservation_trace_max_events` defaults to 64, accepts 0 to
  disable collection, and is hard-bounded at 256.
- Each immutable event records request ID, logical prefix length, live-cache
  estimate, configured/state/effective budgets, two-clone reserve, target
  store bytes, store entries/bytes before and after reservation, per-call
  eviction deltas, and accepted/reason. It carries no prompt, prompt tokens,
  token IDs, or text.
- `status()` and `get_cache_stats()` expose the FIFO, its capacity, and dropped
  count. The arrival/load harness can explicitly select traced or untraced
  operation.
- The existing estimate, memory-headroom budget, long-context cap,
  `reserved_bytes = approx_bytes * 2`, and `evict_until_below()` call are
  unchanged. Reads around that call derive the event.

## Verification

Focused red/green coverage includes accepted, reserve-eviction, preflight-skip,
disabled, FIFO drop, and configuration-bound behavior. The affected suite
passed `77` tests and Ruff passed before the real-model smoke.

The source-bound smoke reused I076's four distinct QMSUM records plus replay of
the first, manual runtime, 8 GiB configured budget, greedy 8-token output,
concurrency 2, disabled persistence, and 512-token decode-aware prefill. It ran
in separate processes with trace capacities 0 and 64.

| Metric | Untraced | Traced | Delta |
|---|---:|---:|---:|
| replay TTFT | 0.169433 s | 0.166854 s | -1.522% |
| replay total latency | 0.483926 s | 0.479015 s | -1.015% |
| replay decode | 0.384575 s | 0.380071 s | -1.171% |
| whole-plan elapsed | 81.432807 s | 79.490258 s | -2.385% |
| snapshots / bytes | 4 / 1,988,067,328 | 4 / 1,988,067,328 | exact |
| evictions | 0 | 0 | exact |

All five terminal workload IDs, completion counts, output-token hashes, text
hashes, and length finishes match. Both rows end at zero running, waiting, and
pending requests. Replay is an exact hit with zero prefill steps in both rows.
The traced row records five accepted decisions, drops none, and contains none
of the forbidden payload fields. Replay TTFT movement is inside the
predeclared absolute 3% gate.

Process RSS and host-global swap started from materially different OS/MLX
states across the two fresh processes. They are archived as context and are
not used to claim observer memory benefit or regression.

The compact evidence is
`artifacts/ITER-20260729-077-prefix-reservation-decision-trace/prefix-reservation-decision-trace-admission.json`;
it binds both ignored raw rows and all implementation/test sources by SHA-256.

## Decision

Admit the bounded prompt-free reservation observer. It passes the scoped no-op
gate and changes no clone-reserve, eviction, entry-limit, cache-budget, or
snapshot-representation behavior. This is not a cross-engine or global speed
claim.

At a hypothetical 3 GiB configured budget, the five observed reserve targets
would be 2,441,019,392; 2,428,174,336; 1,964,048,384; 2,075,525,120; and
2,441,019,392 bytes. Every corresponding pre-reservation store was lower than
its target. I078 therefore tests 3 GiB as the next evidence-selected boundary,
but cannot change the production default from this single key order.

## Rollback

Set `engine.snapshot_reservation_trace_max_events=0` to remove runtime
collection, or revert the telemetry/config/harness changes. Prefix store
contents and eviction behavior require no migration.
