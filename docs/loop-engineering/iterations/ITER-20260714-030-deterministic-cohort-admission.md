# Iteration 030: Deterministic Cohort Admission

Date: 2026-07-14

## Scope

- Starting commit: `5b0096a`
- Source commit: `9dbfc7d`
- Machine: macOS 27.0 Apple Silicon arm64, Python 3.14.5, 24 GB unified
  memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`, mlx-audio `0.4.5`,
  transformers `5.12.1`

Iteration 029 showed that lane=2 preserved parity for simultaneous mixed
requests but not for staggered arrivals. The root cause was late membership:
the first isolated secondary request could execute before later same-profile
requests arrived, and adding them afterward changed BatchGenerator's greedy
output.

## Hypothesis and design

An isolated secondary lane should wait briefly before its first `next()` call,
allowing a bounded arrival cohort to form. Once the first `next()` executes,
the lane is sealed and later requests cannot join it. Simultaneous workloads
with backlog do not pay this delay because their cohort is already visible;
only a new profile created while another lane is active and the waiting queue
is empty receives the window.

Added:

- `engine.batch_generator_lane_admission_window_ms`, default `0.0`.
- `_BatchLane.created_at`, `admission_window_ms`, and `sealed` state.
- Lane sealing before the first generator step.
- Configuration and runtime rejection of `max_lanes > 1` without a positive
  admission window, preventing the known unsafe mode.
- Benchmark option `--lane-admission-window-ms` and recorded metadata.

The production default remains one lane and zero window. A measured safe
candidate is lane `2` with a `160ms` window for this machine and workload; it
is not promoted because its staggered p95 tradeoff is still material.

## Benchmark and threshold results

Baseline: Iteration 029 lane `1`, prefix cache off, Qwen3.5-0.8B 4-bit,
temperature `0.0`, workloads `mixed,staggered`, concurrency `2,4`, two rounds,
long prompt words `64`.

Candidate command:

```text
python scripts/dev/benchmark_batched_engine.py \
  --config /tmp/aster-qwen08-native.yaml \
  --workloads mixed,staggered --concurrency-levels 2,4 --rounds 2 \
  --long-prompt-words 64 --prefix-cache off --max-lanes 2 \
  --lane-admission-window-ms 160 \
  --output artifacts/.../cohort-dynamic-160.json
```

The 8-record candidate matrix had exact response-hash parity against lane `1`,
zero errors, zero swap delta, and `1.495 GB` peak MLX memory. Elapsed time
changed by `-0.19%` to `-4.99%` (average `-2.52%`). p95 changed by
`-4.99%` to `+12.12%` (average `+3.50%`), with the positive tail caused by
the isolated secondary cohort waiting for the window.

Window threshold probes on the same staggered workload found two mismatched
records at `100ms` and `140ms`, and zero mismatches at `160ms` and `200ms`.
This is workload evidence, not a universal safe window for all arrival
patterns.

Additional checks with lane `2` and a `160ms` window:

- Structured output: 6/6 valid JSON responses, all `stop`, zero errors and
  zero swap delta.
- Streaming: 2/2 requests completed, equal response hashes, zero running or
  failed requests afterward.
- Cancellation: 3 completed, 1 cancelled, follow-up completion length 8,
  zero running requests and zero pinned prefix entries.

## Verification

```text
pytest -q                         # 435 passed, 9 skipped, 1 warning
ruff check ...                    # All checks passed
python -m compileall -q aster scripts tests
uv lock --check                   # passed
pip check                         # No broken requirements found
git diff --check                  # passed
```

## Decision and rollback

Retain cohort admission as an explicit experimental capability. Keep
`batch_generator_max_lanes=1` and `batch_generator_lane_admission_window_ms=0`
by default because the p95 regression fails the default-profile gate. A
consumer that explicitly opts into lane `2` must provide a positive window.
Operational rollback is setting lane limit to `1`; source rollback is
`git revert 9dbfc7d`.

Raw artifacts:

- `artifacts/ITER-20260714-030-deterministic-cohort-admission/baseline-lanes-1.json`
- `artifacts/ITER-20260714-030-deterministic-cohort-admission/cohort-dynamic-160.json`
- `artifacts/ITER-20260714-030-deterministic-cohort-admission/cohort-dynamic-200.json`
- `artifacts/ITER-20260714-030-deterministic-cohort-admission/cohort-dynamic-100.json`
- `artifacts/ITER-20260714-030-deterministic-cohort-admission/cohort-dynamic-140.json`
- `artifacts/ITER-20260714-030-deterministic-cohort-admission/structured-160.json`
- `artifacts/ITER-20260714-030-deterministic-cohort-admission/stream-160.json`
- `artifacts/ITER-20260714-030-deterministic-cohort-admission/summary.json`

## Next priority

Reduce the p95 cost without reopening membership drift. Candidate directions
are event-driven cohort closure, a model-native fixed-batch policy, or a
request-independent BatchGenerator state isolation boundary. Any candidate
must retain exact parity under staggered arrival before it can change the
default profile.
