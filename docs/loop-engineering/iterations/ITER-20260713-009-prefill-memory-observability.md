# Iteration 009: Prefill Memory Observability

- Iteration ID: `ITER-20260713-009`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Start commit: `f27ac4f`
- End commit: `fee6db4`
- Runtime: Python 3.14.5, MLX `0.32.0`, MLX-LM `0.31.3`, manual engine
- Model: Qwen3.5-9B-4bit

## Problem

The benchmark exposed a request-level MLX peak, but could not identify whether
that peak came from prefill or decode. The next optimization decision therefore
had weak evidence about the actual working-set source.

## Change

`PrefillChunkResult` now carries:

- `peak_memory_gb`: MLX peak after the chunk evaluation;
- `active_memory_gb`: active MLX allocation after cache evaluation and cleanup.

The engine aggregates maximum prefill values into `engine_timing`, preserves
prefill peak in the response's overall `peak_memory_gb`, and direct benchmark
records expose both prefill fields.

## 9B Evidence

Default cap-0 checkpoint policy, one active request, greedy sampling, and
512-token prefill chunks:

| Prompt tokens | Prefill peak | Prefill active | Overall peak | Elapsed | Completion tok/s | Snapshot stores |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12,181 | 9.121 GB | 6.866 GB | 9.121 GB | 35.665s | 3.59 | 1 |
| 30,181 | 12.124 GB | 6.149 GB | 12.124 GB | 88.132s | 1.45 | 1 |

Both overall peaks equal prefill peaks, so decode is not the dominant memory
source for these long prompts. Active memory after cleanup is much lower than
peak, which indicates transient prefill allocations and/or allocator working
set pressure rather than only retained KV state. Timings are single trials and
are not a randomized performance claim.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q aster scripts/dev/benchmark_live.py tests
```

Results: `389 passed, 9 skipped, 1 warning`; `pip check` and compileall passed.

## Conclusion and Next Priority

Keep `fee6db4`. The new metrics close the observability gap without changing
model execution policy. The next implementation should target retained/transient
prefill working-set memory through a lossless paged KV or allocator-aware cache
strategy, with exact greedy output parity and 12K/16K/30K memory gates.
