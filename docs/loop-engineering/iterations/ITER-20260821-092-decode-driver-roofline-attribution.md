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
