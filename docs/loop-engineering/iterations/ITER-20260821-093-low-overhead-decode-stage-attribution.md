# ITER-20260821-093: Low-Overhead Decode-Stage Attribution

## Baseline

I092 establishes a valid Aster-only B4 observer matrix but rejects the
implementation as production instrumentation. With the observer enabled,
decode changes `-1.063%` (B4-short) and `-4.804%` (B4-mixed); TTFT/e2e p95 and
mixed-load memory also regress. The default remains
`decode_stage_observer_max_events=0`.

## Objective

Measure the same decode-driver ownership boundaries without per-step Python
event dictionaries or multiple `perf_counter()` calls on the hot path. The
result must identify whether cache preparation, model enqueue, sampler graph
construction, lazy evaluation, or result delivery is worth a production
experiment while preserving MLX's existing synchronization behavior.

## Candidate Designs

- Aggregate-only counters with one start/end timestamp per decode call and a
  fixed numeric stage accumulator; no event list or copied context metadata.
- Deterministic periodic sampling (for example one call in 32) with a bounded
  sample count, leaving unsampled calls identical to the disabled path.
- A separate benchmark process trace that correlates existing engine timing,
  decode batch diagnostics, and allocator counters rather than adding hooks to
  ModelRunner.

The first implementation must compare these designs against the I092
observer-off baseline. It must not add an explicit `mx.eval`, change cache
ownership, alter sampling, or change request scheduling.

## Gates

1. Focused unit tests prove disabled mode has no timer/event calls and enabled
   mode remains bounded.
2. A fresh balanced B4-short/B4-mixed matrix has exact output/finish identity,
   clean terminal state, zero fallbacks, and comparable source/input hashes.
3. Observer overhead is below `1%` for decode TPS, TTFT p95, e2e p95, peak MLX,
   and peak RSS in both cells; any failed gate rejects the observer.
4. Only after the no-op screen passes may stage shares guide one minimal
   runtime candidate. That candidate needs a fresh A/B matrix, >=3% repeated
   end-to-end gain, rollback, and cancellation/structured-processor coverage.

## Research Inputs

LLMVisor (`arXiv:2608.08382`) supplies the piecewise attribution framing.
TileMix (`arXiv:2608.17336`) and CoRun (`arXiv:2608.14376`) are current watch
items for tile-level attention precision and fixed-shape batching. Their
reported gains are not transferred to Aster without an Apple/MLX reproduction.
MTP, DFlash, EAGLE-family, and tree speculation remain foundation-gated.
