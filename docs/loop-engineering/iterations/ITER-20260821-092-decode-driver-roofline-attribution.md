# ITER-20260821-092: Decode-Driver Roofline Attribution

## Baseline

I091 leaves the locked Qwen3.5-9B B4 baseline unchanged. Batch-wide logprob
normalization is exact but does not reproduce a 3% decode-driver improvement:
B4-short changes `54.105929 -> 53.248829 tok/s` (`-1.584%`) and B4-mixed
changes `33.850635 -> 33.859096 tok/s` (`+0.025%`).

## Objective

Build a benchmark-only, no-forced-evaluation attribution model for the Aster
decode-driver boundary. Separate model/cache graph work, normalization and
sampling graph work, grouped evaluation, and required result materialization
without changing production scheduling, cache ownership, or sampling.

## Method

- Use the same locked B4-short and B4-mixed public cohorts, 9B model,
  generation settings, hashes, fresh processes, and balanced order as I091.
- Adapt the piecewise/roofline attribution approach described by LLMVisor to
  Apple Silicon and MLX lazy graphs. Attribute observable boundary deltas,
  rather than timing private Python calls that move GPU barriers.
- Record decode TPS, absolute seconds/tokens, TTFT p95, end-to-end p95,
  aggregate throughput, MLX/RSS/swap, output identity, fallbacks, and terminal
  cleanup for every row.
- Keep every diagnostic timing-invalid unless its observer passes an absolute
  3% no-op screen against the untraced baseline.

## Selection Gate

A production candidate is considered only when one stage owns a reproducible
minimum 3% of the valid boundary and a minimal change can remove that cost in
both B4 cells. It must retain exact output, processor/sampler semantics,
balanced-order improvement, memory/swap ceilings, cancellation cleanup, and an
explicit rollback. Otherwise I092 records attribution and advances no code.

MTP, DFlash, EAGLE-family, tree speculation, and adaptive multi-token heads
remain research-only until the foundation gap and target/cache rollback
contracts are closed.

## Implementation

The benchmark-only observer is exposed as
`engine.decode_stage_observer_max_events`. The default is `0`, which bypasses
the timing calls, event allocation, and accumulator updates. A value of `64`
keeps a bounded FIFO and aggregate counters for the timed process only. The
observer records cache preparation, model enqueue, sampler enqueue, the
existing lazy-evaluation/materialization window, result delivery, batch shape,
context-token totals, processor-row count, and cache reuse/rebuild mode. It
does not call `mx.eval` or add a new synchronization point. The benchmark
subtracts warmup counters before retaining the timed observer window.

## Fresh Matrix

The final-source matrix contains two observer states x two B4 cells x four
fresh repetitions (16 Aster processes). Each state uses the same model,
tokenizer, workload, input manifests, execution contract, and balanced engine
order metadata. All 16 commands exited `0`; outputs and finish reasons are
identical between observer-off and observer-on, terminal cleanup is clean, and
decode fallback count is zero.

| Cell | Observer off decode TPS | Observer on decode TPS | On/off delta | TTFT p95 delta | E2E p95 delta | Peak MLX delta | Peak RSS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4-short | 52.376785 | 51.820241 | -1.063% | +6.980% | +5.646% | 0.000% | -6.174% |
| B4-mixed | 33.019685 | 31.433437 | -4.804% | +3.140% | +3.099% | +3.650% | +7.055% |

The observer-on timed window retained 11 events (9 batch, 2 single) in
B4-short and 18 events (8 batch, 10 single) in B4-mixed, with zero dropped
events. Its median diagnostic stage shares are evaluation window
`93.116%`/`94.648%`, cache preparation `4.527%`/`1.875%`, and model enqueue
`2.271%`/`3.109%` for short/mixed respectively. These shares describe the
existing lazy boundary only; they are not private-kernel attribution because
deferred MLX work is released by the existing materialization step.

## Decision

The measurement is valid, but the observer fails the declared 3% no-op gate:
B4-mixed decode falls `4.804%`, both cells increase TTFT and end-to-end p95 by
at least `3.099%`, and B4-mixed peak MLX/RSS rise `3.650%`/`7.055%`. The
observer is therefore rejected as production instrumentation and remains
benchmark-only with the default disabled. No scheduler, cache, sampler, model,
or inference default changes are admitted. The complete recomputable evidence
is [decode-stage-observer-rejection.json](../artifacts/ITER-20260821-092-decode-driver-roofline-attribution/decode-stage-observer-rejection.json)
(SHA-256 `285560917ecf6f9c50018526d605cb1f5e2efec3b9f52fa187fc0ec86bfb7f44`).

## Next Gate

I093 replaces per-step timestamp/event allocation with a lower-overhead
attribution design. It must first reproduce the off/on no-op screen, target
less than `1%` decode and tail overhead, preserve zero forced evaluations and
bounded memory, and only then use LLMVisor-inspired stage features to select a
runtime experiment. TileMix (`2608.17336`) and CoRun (`2608.14376`) are added
to the frontier watchlist; neither is an Aster implementation source yet.
