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

## Implementation

- `host_state_telemetry.py` now evaluates deterministic rolling admission
  windows, retains every raw sample and rejected window, and returns an
  explicit non-replaceable timeout.
- During-child sampling stores the declared
  `max(0, system_cpu_percent - child_cpu_percent / logical_cpu_count)` estimate
  for every aligned sample plus recomputable median/p95 summaries. The contract
  independently recalculates every derived sample before accepting a row.
- The control harness gates every row independently, stops at the first formal
  timeout without launching that child, and hard-gates every
  engine/cell/state external-CPU stratum when a complete matrix exists.
- Focused tests cover admission statistics, rolling retention, timeout
  retention, sample-aligned estimates, per-stratum rejection, no child launch
  after timeout, and full recomputation of the retained formal artifact.

## Calibration And Formal Result

The independent 20-second calibration did not satisfy the declared CPU gate:
its last window had CPU median `18.65%`, p95 `26.11%`, available memory
`35.94%`, and stable swap. This calibration did not count as the formal run.

The one formal attempt preserved the preregistration exactly. Its first planned
row was repetition 1, B4-short, direct MLX-LM, control-off. The rolling gate
waited `120.084624` seconds and timed out before a child was launched. It
retained 1,156 raw samples and 1,137 rejected windows. Across all samples CPU
median was `16.2%` and nearest-rank p95 was `33.3%`; the minimum rolling-window
p95 was still `17.025%`, above the `12%` gate. Available memory never fell below
`35.097%`, and all samples retained the same `1,961,689,088` swap-used value.

The formal status is `invalid-quiescence-timeout`: one row was attempted, zero
inference rows completed, and the remaining 31 rows were not started. No retry
or replacement sample was collected.

## Performance Delta Ledger

I096 is the latest valid control baseline. I097 has no valid decode sample, so
every candidate value and delta is explicitly null rather than being reported
as a speedup or slowdown.

| Cell / engine | I096 observer-off -> control-off median TPS | I096 paired median / order strata | I097 candidate TPS | Absolute delta | Relative delta | Samples (I096 / I097) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B4-short / Aster | `74.262821 -> 76.051570` | `+0.012%`; `+5.263%/-0.102%` | `null` | `null` | `null` | `4 / 0` |
| B4-short / MLX-LM | `85.869765 -> 86.018931` | `+0.748%`; `+0.370%/+1.674%` | `null` | `null` | `null` | `4 / 0` |
| B4-mixed / Aster | `45.154499 -> 44.398127` | `-1.492%`; `+0.186%/-4.459%` | `null` | `null` | `null` | `4 / 0` |
| B4-mixed / MLX-LM | `66.207854 -> 67.737609` | `+1.266%`; `+1.203%/+2.294%` | `null` | `null` | `null` | `4 / 0` |

Measurement status is `invalid-quiescence-timeout`. Admission, complete-row,
telemetry, allocator, external-CPU, semantic, resource, and control-stability
gates are not satisfied because no child ran. The decision is
`reject-quiescence-timeout`; no runtime candidate or production default changes.

## Evidence And Decision

- Raw artifact:
  `docs/loop-engineering/artifacts/ITER-20260825-097-quiescent-host-control/quiescent-host-control.json`
  (`2,628,512` bytes, SHA-256
  `92042c9841029b4b887188e188ff9bcd1f15e18e368e49a2c85f62d6697649e8`).
- Every retained rolling window recomputes from its raw 20 samples in the
  artifact regression test.
- The transaction rollback restores disabled admission, unrecorded rejected
  windows, and no external-CPU estimate on a separate copy while leaving the
  modified fixture enabled.
- The benchmark-only gate is retained because it prevents invalid performance
  claims. Production inference behavior remains unchanged.

I098 moves to a same-process, independently owned, crossed decode control that
can cancel persistent desktop load at a much shorter time scale. A failure at
that boundary makes a dedicated headless benchmark environment a hard
prerequisite. MTP and other speculative paths remain deferred.
