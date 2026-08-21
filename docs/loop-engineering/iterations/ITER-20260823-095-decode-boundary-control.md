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
