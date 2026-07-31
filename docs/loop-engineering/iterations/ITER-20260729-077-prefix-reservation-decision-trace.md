# Iteration 077: Prefix Reservation Decision Trace

- Date: 2026-07-29
- Phase: planned
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
