# ITER-20260823-095: Decode-Boundary Control Classification

## Entry From I094

I094 extended the observer window to 32 generated tokens. Correctness and
resource contracts passed, but the mixed Aster decode order strata were
`-2.174%/+30.592%`; the direct MLX-LM control also exceeded the `1%` stability
gate. The apparent mixed `+6.393%` paired median is not attributable to the
observer.

## Objective

Classify host and control-engine state at the common decode boundary before
adding another observer or selecting a runtime optimization. The primary metric
is `decode_driver_tps`; secondary metrics are TTFT p95, end-to-end p95, peak
MLX/RSS, swap, and thermal/power availability.

## Hypothesis

If the remaining variance is caused by host state or process position rather
than observer work, explicit idle/prewarm/order controls will move Aster and
MLX-LM together. A state-balanced control matrix can then define a valid
comparison boundary or reject the current timing as unassignable.

## Design

- Keep the locked public B1/B4 workload, model, tokenizer, greedy settings,
  prefill step, cache-off state, and source hashes.
- Compare a predeclared host-state control (idle/prewarm/allocator state) with
  the same adjacent off/on observer pair; keep fresh process isolation.
- Preserve AB/BA engine order and observer-state balance, all raw rows,
  terminal/fallback/swap data, and unavailable thermal/power fields.
- Do not modify production inference behavior while the control boundary is
  unresolved.

## Gates

1. Every required engine/cell covers the public workload exactly once per
   repetition with source/input/output/finish/terminal parity.
2. Control-engine and Aster order strata are within `1%` for the primary metric
   before timing is assigned to an implementation owner.
3. No observer or runtime candidate is admitted without a repeatable `>=3%`
   end-to-end gain, exact semantics, rollback, and resource stability.

MTP, DFlash, EAGLE-family, tree speculation, and multi-token prediction remain
foundation-gated.

## Implementation

This iteration adds only the benchmark harness
`scripts/dev/benchmark_decode_boundary_control.py` and its focused contract
tests. The harness reuses the I094 observer-off rows as a retained baseline,
then runs an identical observer-off configuration again as `control-off` in
fresh processes. It balances the control/baseline process position and the
Aster/MLX-LM engine order per cell and repetition, retains source/input/output
hashes, terminal state, fallback counts, memory, RSS, and swap, and verifies
the foundation-declared warmup contract. No serving or inference default was
changed.

The transaction fixture changes only `benchmark.state_control` from
`observer-off` to `control-off`. `ROLLBACK.sh` restores a separate copy while
the modified fixture remains changed; the complete evidence is under
`docs/loop-engineering/artifacts/ITER-20260823-095-decode-boundary-control/`.

## Formal Matrix

The locked Qwen3.5-9B 4-bit model, tokenizer, public `cross-engine-core`
records, greedy settings, cache-off configuration, prefill step, 32-token
window, and I094 source hashes were reused. The matrix adds 16 fresh control
rows: two B4 cells, two engines, four repetitions. Each control row uses the
same off configuration and declared warmup as its paired observer-off row.

All original structural gates pass: source comparability, exact output token/
text/finish identity, clean terminal lifecycle, zero decode fallbacks,
output-cap parity, and positive warmup requests. All 16 new control rows have
zero workload swap growth. A later I096 audit found one reused I094
observer-off baseline row with `458,752` bytes of host-global swap growth; the
historical I095 summary did not hard-gate that field. I096 and later telemetry
matrices require zero workload swap on both sides. The control stability gate
does not pass:

| Cell | Engine | Control-off first | Observer-off first | Control median |
| --- | --- | ---: | ---: | ---: |
| B4 short | Aster | -0.187% | +1.293% | -0.157% |
| B4 short | MLX-LM | -0.157% | +1.695% | -0.070% |
| B4 mixed | Aster | +25.825% | +1.464% | +2.734% |
| B4 mixed | MLX-LM | -1.403% | +1.274% | +0.871% |

The mixed Aster control-first stratum is driven by one retained `+48.771%`
paired control delta. The retained I094 observer off/on strata remain
`-2.174%/+30.592%` for mixed Aster and `-3.114%/-3.438%` for mixed MLX-LM.
The matching control instability means these deltas cannot be assigned to
observer work, an Aster stage, or a runtime optimization.

## Decision

**Reject performance attribution and keep production unchanged.** The
fresh-process off/off control is not within the predeclared `1%` decode-TPS
order-strata gate, even though all semantic and resource contracts pass. This
classifies the current B4 mixed decode boundary as host/process-state
confounded. No observer, scheduler, sampler, cache, speculative, or MTP
candidate is admitted.

The next experiment must reduce this variance before another timing claim:
run a host-state/thermal/allocator trace with explicit idle intervals and
machine telemetry, or move to a lower-level kernel boundary whose control has
the same state contract. MTP and tree speculation remain deferred until the
foundation gates in `CURRENT.json` are closed.

## Verification

- Focused regression and retained-artifact tests pass; the artifact recomputes
  the complete control summary from its embedded baseline/control rows.
- Ruff and diff checks pass for the touched harness and tests.
- Formal raw control matrix: `CONTROL_MATRIX.json`, SHA-256
  `230e72b54e05827d70dd6520002df7a83741592aaeff1980b17c284f66e9984e`.
- Retained I094 baseline matrix: `BASELINE_MATRIX.json`, SHA-256
  `4eaad8503963788e590a71c0c55cc91170ba7010b154a9ba55420d6aa8447334`.
- Transaction and exact commands/results: `VERIFICATION.txt`.
