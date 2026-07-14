# Iteration 032: Independent MLX Streams for BatchGenerator Lanes

## Scope

- Starting code commit: `2d1bb49`
- Ending code commit: `2d1bb49` (no source promotion)
- Machine: macOS 27.0, Apple Silicon arm64, Python 3.14.5, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`
- Working tree contained an uncommitted opt-in `lane_streams` candidate; it was
  not overwritten or committed by this iteration.

## Problem and hypothesis

Iteration 031 kept per-profile lanes sequential because concurrent MLX calls
were not proven safe. The candidate assigned each opt-in lane its own
`mx.new_stream(...)` through the installed `BatchGenerator(stream=...)` API.
The hypothesis was that MLX's asynchronous evaluation could overlap work from
independent lanes without changing lane admission, ownership, or token output.

## Validation

Correctness and static checks:

```text
.venv/bin/pytest -q
437 passed, 9 skipped, 1 warning

.venv/bin/ruff check aster/inference/batched_engine.py \
  aster/core/config.py scripts/dev/benchmark_batched_engine.py \
  tests/test_batched_engine.py tests/test_benchmark_batched_engine.py \
  tests/test_config.py
All checks passed!

.venv/bin/python -m compileall -q aster scripts tests
/opt/homebrew/bin/uv lock --check
.venv/bin/pip check
git diff --check
```

The first three checks passed. The initial combined command used the wrong
path `.venv/bin/uv`; rerunning with `/opt/homebrew/bin/uv` is the valid lock
check command. No request errors, cancellation leaks, or peak-memory changes
were observed in the live probes.

Benchmark configuration for the matched A/B:

```text
model: Qwen3.5-0.8B 4-bit MLX
temperature: 0.0
prefix cache: off
max lanes: 2
admission window: 160 ms
cohort target: 3
longest-lane quantum: 2
workloads: mixed, staggered
concurrency: 2, 4
rounds: 2
```

The benchmark was run once with `--lane-streams` and once without it. All 8
records completed with zero errors, identical response hashes, zero swap
growth, and `1.495 GB` peak MLX memory. Elapsed time changed by `-0.84%` to
`-2.53%`, p95 changed by the same range in that matched pair, and completion
throughput improved by `+0.85%` to `+2.60%`.

Additional interleaved reruns showed only about `0.5%~1.1%` mixed-workload
improvement. Staggered results varied with arrival timing, including a
`+4.79%` p95 regression in one matched workload/round. Hash changes also
appeared in both stream-enabled and stream-disabled arms, so they are an
existing arrival-sensitive cohort effect rather than evidence that streams
preserve or improve determinism.

Cancellation probes for both arms completed 3 requests, cancelled 1 request,
accepted an 8-token follow-up, and ended with zero running requests and zero
pinned prefix entries.

## Root cause and interpretation

`BatchGenerator.next()` already wraps `_next()` in its configured stream and
performs synchronous `mx.eval` work for current tokens. The engine then calls
each lane sequentially and immediately processes prompt/response objects.
Consequently, assigning a second stream does not create a sufficiently large
overlap window in this workload; the remaining cost is scheduler and
arrival/cohort behavior rather than stream selection.

## Decision

**INVESTIGATING — do not promote.** Keep the candidate opt-in and leave the
production/default lane settings unchanged. It does not clear the 3% core
performance gate, and the staggered workload still has timing-sensitive hash
variation that must be solved at the admission boundary.

Rollback for the uncommitted candidate is to discard only the candidate-owned
`lane_streams` changes after confirming ownership with the user; no rollback
was performed here because the working tree contained pre-existing edits.

## Next priority

Evaluate event-driven cohort closure or model-native fixed-batch/state
isolation. The candidate must make lane membership independent of arrival
timing, retain exact greedy token parity, and beat the current safe lane-1
baseline on p50/p95 without increasing peak memory or swap.
