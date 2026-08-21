# ITER-20260826-098: Persistent-Load Crossed Decode Control

## Entry From I097

I097 proved that this desktop host cannot satisfy the preregistered two-second
quiescence gate: the sole formal attempt retained 1,137 rejected windows over
120 seconds and launched no inference child. The lowest rolling CPU p95 was
`17.025%` against a `12%` ceiling. Relaxing that threshold after seeing the data
would convert known external load into an apparently valid benchmark.

## Objective

Determine whether an in-process, independently owned, crossed control can make
the common decode boundary stable under persistent host load. This is
measurement infrastructure only; do not change production inference behavior.

## Hypothesis

If slow host/load/model-start state owns the I093-I096 variance, alternating two
byte-identical decode branches at four-token granularity inside one loaded-model
process will cancel that drift and keep both overall and execution-order control
deltas within `1%`. If the crossed control remains unstable, this host session
cannot support another performance attribution and a dedicated headless run is
required.

## Frozen Design

- Reuse the locked Qwen3.5-9B model/tokenizer, public B4-short/B4-mixed records,
  cache-off greedy settings, 32-token output cap, source hashes, and declared
  warmup.
- For each engine/cell/repetition, prefill once and create two independently
  owned decode branches with identical state. Prove construction does not share
  mutable KV/recurrent storage before timing.
- Decode both branches in eight corresponding four-token blocks. Alternate
  block order as AB/BA/BA/AB within each process and balance the outer order
  over four fresh processes per engine/cell.
- Use the same token materialization boundary for both branches. Retain every
  block; do not select blocks from timing, CPU, output, or acceptance results.
- Record per-block elapsed time, tokens, finish state, cache/state digest,
  allocator/RSS/swap envelopes, and before/after host/process CPU snapshots.
  Host fields remain diagnostic and cannot delete a block.

## Predeclared Gates

1. Both branches start from byte-identical immutable input and independent
   mutable state; all 32 generated tokens, text, finish, and final state match.
2. Every process completes all 16 timed block executions with clean terminal
   state, zero fallback, declared warmup, and zero workload swap growth.
3. Per engine/cell, paired median decode-block TPS delta and both block-order
   strata are within `1%`; no individual process may exceed `3%` median drift.
4. The result reports all block values, sample counts, host context, and a
   linked I096 valid baseline. A failed gate invalidates attribution.
5. No production candidate is admitted without exact semantics, stable
   resources, rollback, and a repeatable `>=3%` end-to-end improvement.

## Research Inputs

- `On Evaluating Performance of LLM Inference Serving Systems` (`2507.09019`)
  supplies the fairness, heterogeneous-workload, metric, and variability
  anti-pattern checklist.
- `LLM Inference at the Edge ... Under Sustained Load` (`2603.23640`) makes
  warm-condition thermal/frequency behavior an explicit deployment variable.
- mlxcel `fdc19666` demonstrates bounded measured-cost policy decisions,
  fail-closed block/chain exactness probes, per-row verification semantics, and
  dynamic paged-decode planning. These are contract references only.

MemSpec (`2608.10362`), Windowed-MTP (`2607.21535`), AngelSpec (`2607.25852`),
and mlxcel's MTP runtime remain research-only until this foundation and the full
rollback/sampler/stop/streaming/mixed-load gates close.
