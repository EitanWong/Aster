# Iteration 033: Event-Driven Cohort Closure

## Scope

- Starting code commit: `bcad7ee`
- Ending code commit: `bcad7ee` (candidate rolled back)
- Machine: macOS 27.0, Apple Silicon arm64, Python 3.14.5, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`

## Problem and hypothesis

The safe multi-lane configuration waits up to 160 ms for a staggered cohort,
which raises tail latency. The candidate introduced an opt-in event-driven
closure mode: requests already in the current event-loop admission pass form
the batch, the lane seals before its first generator step, and late requests
use a new lane with the same profile. This required owning-lane references and
support for repeated profile keys.

Hypothesis: removing the fixed wait would preserve token parity while reducing
staggered latency and retaining enough batching for mixed workloads.

## TDD and verification

The new repeated-profile-lane, configuration, and benchmark override tests
were first run red (`3 failed`), then passed after the minimal implementation.
The candidate implementation passed:

```text
.venv/bin/pytest -q
438 passed, 9 skipped, 1 warning

.venv/bin/ruff check ...
All checks passed!

.venv/bin/python -m compileall -q aster scripts tests
/opt/homebrew/bin/uv lock --check
.venv/bin/pip check
git diff --check
```

## Benchmark

Both arms used Qwen3.5-0.8B 4-bit MLX, greedy sampling, prefix cache off,
max lanes 2, cohort target 3, longest-lane quantum 2, workloads
`mixed,staggered`, concurrency 2/4, and two rounds. The baseline used a 160 ms
admission window; the candidate used `--event-driven-cohorts` with a zero
window.

All 16 records completed with zero request errors, zero swap growth, and
`1.495 GB` peak MLX memory. Cancellation in both arms completed 3 requests,
cancelled 1, accepted an 8-token follow-up, and left zero running requests and
zero pinned entries.

Compared with the fixed-window baseline:

| Workload | Event elapsed | Event p95 | Event completion TPS | Response hashes |
| --- | ---: | ---: | ---: | --- |
| mixed C=2 | `-2.48%~-2.77%` | same | `+2.54%~+2.85%` | equal |
| mixed C=4 | `-0.89%~-0.98%` | same | `+0.90%~+0.99%` | equal |
| staggered C=2 | `+24.50%~+25.38%` | `+7.98%~+8.02%` | `-19.68%~-20.24%` | different |
| staggered C=4 | `+26.45%~+27.73%` | `+8.85%~+11.22%` | `-20.92%~-21.71%` | different |

The hash differences are not an acceptable correctness regression claim by
themselves because the existing staggered workload is arrival-sensitive, but
they demonstrate that the candidate did not preserve the fixed-window batch
trajectory. The large staggered slowdown is decisive.

## Root cause

With staggered arrivals, the event-driven policy seals the first request before
later same-profile requests arrive. Those requests are correctly isolated into
additional lanes, but each lane often contains only one request. The engine
still advances lanes sequentially, so the candidate trades a short admission
wait for repeated single-request generation and loses continuous-batching
efficiency.

## Decision and rollback

**ROLLED_BACK.** The event-driven source and tests were removed without
touching the pre-existing independent-MLX-stream working-tree changes. The
default configuration and the validated 160 ms safety window remain unchanged.

## Next priority

Do not create repeated same-profile lanes until there is a model-native way to
keep their execution batch-independent without serial single-request lanes.
Investigate fixed-shape padding/masking or a BatchGenerator state-isolation
boundary, with lane-1 comparison, exact greedy parity, and memory/swap gates.
