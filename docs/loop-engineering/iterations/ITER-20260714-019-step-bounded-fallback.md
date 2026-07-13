# Iteration 019: Bound Paged KV Fallback Growth

- Iteration ID: `ITER-20260714-019-step-bounded-fallback`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `8d562d0`
- Code end commit: `89dc086`
- Evidence: `iterations/artifacts/ITER-20260714-019-step-bounded-fallback/current/`

## Problem And Hypothesis

The storage-only paged boundary removed the persistent pool from normal serving,
but the contiguous fallback still used geometric doubling. During 8K chunked
prefill, the final append from 8,192 to 8,373 tokens doubled capacity to
16,384, while native MLX-LM grew in configured steps. The hypothesis was that
step-bounded growth would reduce transient and retained memory without adding
per-token reallocations.

## Change

`PagedKVCacheLayer` now grows by `max(step, overflow)` rather than doubling.
The initial allocation still respects `block_size`, and appends within the
existing capacity remain in-place. A regression test proves an 8-to-9 token
append with step 4 grows to 12 rather than 16.

## Evidence

The saved capacity profile shows that for the 8,373-token workload native KV
layers ended at shape `[1, 2, 10240, 256]`, while all six paged full-attention
layers ended with materialized capacity `8373`.

| Path | Prompt | Elapsed | Completion tok/s | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: |
| Native | 2,229 | `2.782s` | `46.01` | `1.677 GB` |
| Storage-only paged | 2,229 | `2.769s` | `46.23` | `1.654 GB` |
| Native | 8,373 | `5.437s` | `23.54` | `2.297 GB` |
| Storage-only paged | 8,373 | `5.431s` | `23.57` | `2.286 GB` |

The randomized 8K order was `paged, native, native, paged, paged, native`.
Medians were native `5.4353s` versus paged `5.4259s` (`-0.17%`), throughput
`23.550` versus `23.591` completion tok/s (`+0.17%`), and peak memory
`2.297 GB` versus `2.286 GB` (`-0.46%`). This is below the 3% performance
gate, but it removes the measured memory regression without a timing regression.

All six randomized requests completed 128 tokens with zero swap delta. Current
greedy parity matched in text, prompt tokens, completion tokens, and finish
reason. Pool lifecycle still reclaimed all manager blocks after release.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests scripts/dev
.venv/bin/pip check
git diff --check
```

Result: `412 passed, 9 skipped, 1 warning` across 421 collected tests.

## Decision And Next Priority

Keep step-bounded fallback growth in the opt-in paged boundary. It is a
measured memory improvement and remains behaviorally compatible, but does not
justify changing the production default. Next, revisit direct attention over
the persistent pool; the existing Metal kernel remains disabled until it can
clear the same correctness and randomized end-to-end gates.
