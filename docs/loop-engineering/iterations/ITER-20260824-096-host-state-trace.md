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
