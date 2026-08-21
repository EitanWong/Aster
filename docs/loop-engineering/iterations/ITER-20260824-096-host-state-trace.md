# ITER-20260824-096: Host-State Trace Before Decode Attribution

## Entry From I095

I095 added an off/off fresh-process control to the locked 32-token B4
observer matrix. Semantic and resource contracts passed, but Aster B4-mixed
control-first decode TPS moved `+25.825%` versus `+1.464%` observer-off-first;
the retained control rows include one `+48.771%` paired delta. The common
decode boundary is therefore still state-confounded.

## Objective

Make host, thermal, process, and allocator state explicit at the decode
boundary before evaluating another observer, scheduler change, kernel, MTP, or
speculative decoder.

## Design

- Reuse the locked Qwen3.5-9B model/tokenizer, public B1/B4 records, greedy
  settings, cache-off state, output cap, source hashes, and exact output
  contract.
- Add a benchmark-only state envelope containing process ID, launch order,
  elapsed idle interval, CPU/GPU utilization where available, memory pressure,
  MLX allocator/free-cache counters, RSS, swap, and thermal/power availability.
- Cross explicit idle intervals and prewarm completion with balanced AB/BA
  process order. Keep one timed decode per fresh process and retain every raw
  row, including unavailable telemetry fields.
- Do not change Aster production inference behavior until the control itself
  clears the `<=1%` decode-TPS order-strata gate.

## Gates

1. Exact source/input/output/finish/terminal parity and zero fallback/swap
   growth for every required public cell.
2. Complete telemetry envelope or explicit `unavailable` values; no inferred
   thermal or allocator state.
3. Aster and MLX-LM off/off control order strata within `1%` for decode TPS
   before any implementation owner is assigned.
4. No candidate without a repeatable `>=3%` end-to-end gain, rollback,
   resource stability, and exact semantics.

MTP, DFlash, EAGLE-family, tree speculation, and multi-token prediction remain
foundation-gated.

## Benchmark-Only Implementation

- `scripts/dev/host_state_telemetry.py` captures parent/host snapshots and
  samples each fresh child from outside its timed inference path. The retained
  envelope includes child RSS/CPU, system CPU, available memory, swap, load,
  requested/observed idle time, and explicit command availability.
- `scripts/dev/benchmark_foundation_parity.py` records MLX
  `active/cache/peak` allocator bytes immediately before and after each timed
  cell. These are official MLX counters, not inferred RSS components.
- `scripts/dev/benchmark_decode_boundary_control.py` runs same-session
  observer-off/control-off pairs in alternating order, requires both telemetry
  and allocator contracts, and preserves pair-level state deltas plus
  diagnostic Pearson correlations.
- No file under `aster/` and no production configuration changed.

## Failed Screens Retained

1. The first 16-row trace compared new control rows with the historical I094
   matrix. Although its telemetry was structurally valid, the rows were not
   adjacent same-session pairs, so it was rejected as attribution evidence.
2. A first 32-row paired matrix reduced I095's `+48.771%` extreme but omitted
   the planned system-CPU and child-side allocator fields. Its hash is
   `d30f32c1c46d221771b57acafe1bc953545fe85a7ed35b939ccfe81da4c29c7a`;
   it is a development screen, not the retained formal result.

## Formal Matrix

The final matrix contains 32 fresh processes: B4-short/B4-mixed x
Aster/direct MLX-LM x observer-off/control-off x four repetitions. Every pair
uses the locked Qwen3.5-9B model/tokenizer, public source lock, greedy settings,
cache-off configuration, 32-token output cap, declared warmup, 2-second idle,
50 ms external sampling, and alternating state-first order.

Primary metric values are medians of the four absolute row measurements.
Relative values are the median of four paired control/baseline ratios, so they
need not equal the ratio of the two absolute medians.

| Cell | Engine | Observer-off baseline | Control-off | Absolute delta | Paired median | Control-first / observer-first strata |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B4 short | Aster | 74.262821 tok/s | 76.051570 tok/s | +1.788749 tok/s | +0.012% | +5.263% / -0.102% |
| B4 short | MLX-LM | 85.869765 tok/s | 86.018931 tok/s | +0.149167 tok/s | +0.748% | +0.370% / +1.674% |
| B4 mixed | Aster | 45.154499 tok/s | 44.398127 tok/s | -0.756372 tok/s | -1.492% | +0.186% / -4.459% |
| B4 mixed | MLX-LM | 66.207854 tok/s | 67.737609 tok/s | +1.529755 tok/s | +1.266% | +1.203% / +2.294% |

Individual paired decode deltas are:

- B4-short Aster: `+10.824%, +0.321%, -0.297%, -0.524%`.
- B4-short MLX-LM: `+0.948%, +2.799%, -0.208%, +0.549%`.
- B4-mixed Aster: `-7.159%, +1.597%, -1.759%, -1.226%`.
- B4-mixed MLX-LM: `+1.233%, +1.298%, +3.356%, +1.107%`.

## State Evidence

- Source, plan, input, exact output/finish, terminal cleanup, output cap,
  prewarm, telemetry, allocator, and zero-fallback gates all pass. Every row
  has zero workload-stage swap growth; host swap remains fixed at
  `2,045,575,168` bytes throughout the trace.
- `memory_pressure` is explicitly available in all 32 envelopes.
  `powermetrics` exits unavailable and `pmset -g thermlog` times out, so no
  thermal or power value is inferred.
- Timed-start MLX active memory has only the two adjacent values
  `5,038,041,610` and `5,038,041,614` bytes. Allocator state therefore does not
  explain the large decode outliers in this sample.
- System-CPU pair deltas have diagnostic correlations with decode-TPS deltas of
  `r=-0.822` for Aster and `r=-0.798` for MLX-LM (`n=8` each). The all-engine
  value is `r=-0.673` (`n=16`). The sample is small and observational, so these
  values select a controlled follow-up but do not establish causality.
- System CPU includes the measured inference child. A diagnostic subtraction
  of child CPU normalized by the host's ten logical CPUs still gives
  decode-delta correlations of `r=-0.799` for Aster, `r=-0.792` for MLX-LM,
  and `r=-0.631` overall. This derived quantity is not promoted to a measured
  external-load field: subtraction can compound sampler error and the same
  workload drives both terms. I097 must measure the pre-launch external state
  directly and retain the normalized subtraction only as a declared estimate
  during child execution.

## Decision

Reject runtime attribution and make no production change. All four
engine/cell combinations have at least one decode order stratum outside the
predeclared `<=1%` control boundary. A fixed two-second idle interval exposes
host state but does not control it sufficiently.

The retained evidence is
`docs/loop-engineering/artifacts/ITER-20260824-096-host-state-trace/host-state-trace.json`
with SHA-256
`1dbddd3a4db469cc94e753804e0a2f5f4a30a2cbc900840538d062bc172639c2`.
Its 16 baseline and 16 control rows recompute exactly in the focused artifact
test.

## Reference Intake And Next Gate

All 23 reference repositories were fetched from their configured official
branches. Eight existing gitlinks advanced and the MIT-licensed author
S2-MoE implementation was added at `fba914c3`. Current Rapid-MLX capability
tiering, vLLM adaptive verification/acceptance telemetry, SGLang hybrid cache
work, and arXiv papers `2608.15018`, `2608.14787`, `2608.13524`, `2608.13076`,
and `2608.19147` are recorded in `FRONTIER_RADAR.md`. None bypasses the
foundation gate.

I097 will replace fixed idle with a predeclared rolling quiescence barrier and
retain all successes/timeouts. It must still clear `<=1%` decode order strata
before another observer, kernel, MTP, or speculative candidate is evaluated.

## Verification

- Focused telemetry/control/foundation coverage passes `30/30`; the retained
  artifact's dedicated recomputation test passes independently.
- The full suite passes `638 passed, 9 skipped, 1 warning in 7.90s`.
- Touched-file Ruff and format checks, JSON parsing, `git diff --check`, and the
  strict workspace audit pass. The audit has no blockers and reports only the
  completed I096 artifact, nine reference updates, and 24 generated caches.
- The generic `--reuse-existing` path explicitly rejects paired host-state
  rows; it is not a replay contract for this matrix. Deterministic
  recomputation loads the immutable artifact and calls `summarize_control`
  over its 16 baseline and 16 control rows.
- Exact commands, transaction paths, hashes, outputs, and rollback behavior are
  archived in `VERIFICATION.txt` beside the matrix.
