# Iteration 063: Short Decode Measurement Stability

- **Date:** 2026-07-28
- **Phase:** complete
- **Scope:** benchmark infrastructure only; no production runtime candidate is
  authorized until short-decode order interaction is bounded.

## Objective

Make short single-request decode measurements stable enough to distinguish a
real runtime effect from first/second-call, MLX allocator/cache, stream, and
host scheduling interaction.

## Primary Metric

For the frozen Qwen3.5-0.8B 128-word/256-token greedy workload, reduce the
absolute difference between order strata to a documented tolerance while
retaining exact output and zero swap. The measurement method, not a candidate
throughput increase, is the only current deliverable.

## Hypothesis

The I062 sign reversal is caused by unpriced state carried across serial and
pipeline measurement calls. Explicit cold/warm state classification and a
balanced independent-process schedule can expose or remove that interaction.

## Predeclared Gates

1. Preserve I061/I062 model, raw prompt IDs, sampler, output cap, and token/
   finish/swap parity checks.
2. Record every cold/warm and first/second-call observation; do not discard
   unfavorable order strata.
3. Profile allocator/cache and stream state without changing Aster production
   code.
4. Do not authorize an asynchronous decode candidate unless the improved
   protocol has stable interval and order-stratum evidence above the 3% gate.

## Profile Result So Far

Six exact two-call probes tested serial-first and pipeline-first order with no
explicit clearing, `mx.clear_cache()`, and `mx.clear_cache()` plus
`gc.collect()`. All generated the same 256 tokens and had non-growing swap.

| State treatment | Serial-first pipeline gain | Pipeline-first pipeline gain |
| --- | ---: | ---: |
| No explicit pre-clear | -13.029% | +52.311% |
| `mx.clear_cache()` | -24.125% | +16.305% |
| `mx.clear_cache()` + `gc.collect()` | -30.484% | +58.026% |

Without GC, serial completion left approximately 424 MB active MLX memory and
pipeline completion approximately 460 MB. GC fixed both post-measurement active
states near 424 MB, but did not remove the sign reversal. Thus allocator cache
release and delayed Python collection are not sufficient explanations. The
remaining profile target is host/stream scheduling or another unobserved
first/second-call state; no production candidate is authorized.

## Crossed Prewarm Screen

The fixed `serial -> pipeline` prewarm sequence itself was then tested as a
confounder. Eight independent processes crossed both prewarm orders with both
measurement orders, with two records in every 2x2 cell. Each record used fresh
prompt caches, preserved all 256 completion IDs and `length` finish, and did
not grow swap.

| Grouping | Pipeline elapsed gain p50 |
| --- | ---: |
| Pipeline measured first (4 records) | +16.746% |
| Serial measured first (4 records) | -14.022% |
| Pipeline was terminal prewarm (4 records) | +3.759% |
| Serial was terminal prewarm (4 records) | +6.845% |

For every matched `(repeat, warmup-order)` pair, the reported pipeline gain was
higher when pipeline was measured first: `+12.085%`, `+17.663%`, `+27.845%`,
and `+89.357%`. Changing the terminal warmup graph therefore did not remove the
interaction. The host probe recorded coarse load, thread, and frequency values,
but MLX did not expose stream counts in this environment, so it did not identify
the remaining state owner.

## Evidence And Verification

- `measurement-stability-evidence.tar.gz`: 14 minimal state and crossed-prewarm
  records / 19,498 bytes / SHA-256
  `2c7f437b992f539656ccd919453f1d9e01c4e4f6f41108c58fbc2cb8bc20a038`.
- `screen-summary.json`: source-bound rejection and aggregate summaries.
- `test_i063_artifacts.py`: source/hash validation and archive-only crossed
  prewarm aggregation recomputation passed (`2 passed`).
- All work was benchmark-only; Aster production and test sources were unchanged
  during I063.

## Conclusion And Next Priority

Decision: **reject** the fixed-prewarm explanation and retain no asynchronous
runtime change. I064 will isolate same-variant first/second-call pairs with
fresh prompt caches, using the I063 mixed-order result only as a counterfactual.
