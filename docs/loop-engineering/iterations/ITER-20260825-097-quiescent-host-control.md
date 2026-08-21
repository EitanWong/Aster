# ITER-20260825-097: Quiescent Host Control

## Entry From I096

I096 made the common decode boundary observable with same-session fresh-process
pairs, 50 ms external sampling, and MLX allocator counters. Every semantic,
terminal, fallback, swap, telemetry, and allocator contract passed, but each of
the four engine/cell combinations retained at least one decode order stratum
outside `1%`. System-CPU pair deltas correlate negatively with decode deltas,
but total system CPU includes the inference child and does not establish an
external-load cause.

## Objective

Test whether a predeclared rolling host-quiescence barrier makes the locked
off/off B4 decode control stable enough for attribution. Separate pre-launch
external load from a clearly labelled child-normalized estimate during
inference. Do not change production inference behavior.

## Hypothesis

If unrelated host load owns the remaining timing variance, admitting each
fresh child only after the same rolling CPU/memory/swap contract should reduce
both Aster and direct MLX-LM control order strata to `<=1%`. If the gate is met
but timing remains unstable, the next owner is below the host-admission
boundary, such as frequency, stream, graph, or kernel state.

## Frozen Design

- Reuse I096's locked Qwen3.5-9B model/tokenizer, public B4 short/mixed
  records, greedy settings, cache-off configuration, 32-token cap, source
  hashes, declared warmup, fresh-process boundary, and four repetitions.
- Before every child launch, collect 20 system snapshots at 100 ms intervals.
  Admit only when the rolling two-second window has CPU median `<=6%`, CPU
  p95 `<=12%`, available memory `>=20%`, and unchanged host swap.
- Wait at most 120 seconds. Retain every rejected window, total wait, admitted
  window, and timeout. A timed-out attempt invalidates the formal matrix and
  cannot be silently replaced or removed from the ledger.
- Alternate observer-off/control-off order within every engine/cell and retain
  all 32 planned rows. Gate both rows of each pair independently; do not use
  post-run performance to select or discard a row.
- During child execution, retain total system CPU and child-process CPU. Report
  `max(0, system_cpu - child_cpu / logical_cpu_count)` only as an estimated
  external-CPU field, with the formula and sample alignment stored in the raw
  artifact. It is not direct thermal, power, or frequency evidence.
- Keep load average diagnostic because its decay lags the launch boundary.
  Preserve explicit `unavailable` states for thermal and power commands.

## Predeclared Gates

1. All 32 planned rows satisfy the rolling admission contract without a
   timeout, and every rejected/admitted window remains recomputable.
2. Source, input, exact token/text/finish, output cap, terminal cleanup,
   fallback, prewarm, allocator, telemetry, and zero workload-swap contracts
   pass for every row.
3. Estimated during-child external CPU has median `<=6%` and p95 `<=12%` in
   every engine/cell/state stratum. Failure invalidates attribution but does
   not authorize replacement samples.
4. Observer-off/control-off decode-driver order strata are `<=1%` for Aster
   and direct MLX-LM in both B4 cells.
5. No runtime candidate is admitted without exact semantics, stable resources,
   rollback, and a repeatable `>=3%` end-to-end improvement.

## Planned Analysis

- Primary metric: absolute and paired `decode_driver_tps`, reported for every
  engine/cell and control order.
- Secondary metrics: TTFT p95, end-to-end p95, peak MLX/RSS, swap, admission
  wait, admitted CPU median/p95, and estimated during-child external CPU.
- Report individual pair deltas, medians, order strata, sample counts, and
  Pearson correlations only as diagnostics. A correlation cannot satisfy a
  causal or production gate.
- If the host gate passes and control stability fails, move I098 to an isolated
  stream/graph/frequency or lower-level kernel control. If stability passes,
  rerun one predeclared observer no-op screen before selecting a runtime owner.

MTP, S2-MoE, DFlash, EAGLE-family, adaptive tree verification, and other
speculative paths remain reference-only until the foundation gate closes.
