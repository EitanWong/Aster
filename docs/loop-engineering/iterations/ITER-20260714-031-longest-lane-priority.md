# Iteration 031: Prioritize the Longest BatchGenerator Lane

Date: 2026-07-14

## Scope

- Starting commit: `6c839b5`
- Source commit: `5807641`
- Machine: macOS 27.0 Apple Silicon arm64, Python 3.14.5, 24 GB unified
  memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`, mlx-audio `0.4.5`,
  transformers `5.12.1`

Iteration 030 restored exact greedy parity for staggered multi-lane requests
by sealing isolated cohorts, but the secondary lane's admission window raised
staggered p95. The request-level diagnostic showed why: a round-robin loop
allowed the long prompt and short prompts to advance at the same rate, making
the long request the aggregate p95 tail.

## Hypothesis and design

Give the longest active prompt lane an explicit step quantum after cohort
admission has completed. A quantum of `2` runs two generator steps for the
longest prompt lane before advancing to the next lane; all other lanes retain
quantum `1`. Lane membership remains sealed, so this changes scheduling
priority without reopening the hash-drift failure mode.

Added:

- `engine.batch_generator_longest_lane_step_quanta`, default `1` and bounded
  to positive values.
- Benchmark option `--longest-lane-step-quanta` and recorded metadata.
- A lane-step helper and guarded multi-step loop in `BatchedEngine`.
- Explicit `engine.batch_generator_lane_target_size`, default `0`, to bound
  the first cohort independently from the active-request capacity.

The production defaults remain one lane, zero admission window, and quantum
one. The new controls are experimental and require explicit opt-in.

## Benchmark

Baseline: Iteration 029 lane `1`, prefix cache off, Qwen3.5-0.8B 4-bit,
temperature `0.0`, workloads `mixed,staggered`, concurrency `2,4`, two rounds,
long prompt words `64`.

Candidate command:

```text
python scripts/dev/benchmark_batched_engine.py \
  --config /tmp/aster-qwen08-native.yaml \
  --workloads mixed,staggered --concurrency-levels 2,4 --rounds 2 \
  --long-prompt-words 64 --prefix-cache off --max-lanes 2 \
  --lane-admission-window-ms 160 --cohort-target-size 3 \
  --longest-lane-step-quanta 2 \
  --output /tmp/aster-cohort-priority2.json
```

Against the safe cohort candidate with quantum `1`, the candidate changed:

- Mixed elapsed/p95: `+0.42%` to `+2.82%`.
- Staggered elapsed: `+19.80%` to `+22.57%`.
- Staggered p95: `-3.82%` to `-6.42%`.

Against the lane-1 baseline, mixed elapsed improved `2.18%` to `4.08%`,
while staggered elapsed remained `18.13%` to `18.77%` slower and staggered
p95 remained `3.72%` to `4.78%` higher. The candidate's eight records had
exact response-hash parity with both the lane-1 baseline and the quantum-1
cohort candidate. Peak MLX memory was `1.495 GB`, with zero errors and zero
swap growth.

A request-level staggered probe recorded the long request at `2.122s` and the
three short requests at `1.815s`, `1.704s`, and `2.067s`. This confirms the
priority shift toward the long request; it does not make the multi-lane path a
global throughput win.

## Verification

```text
pytest -q                         # 436 passed, 9 skipped, 1 warning
ruff check ...                    # All checks passed
python -m compileall -q aster scripts tests
uv lock --check                   # passed
pip check                         # No broken requirements found
git diff --check                  # passed
```

## Decision and rollback

Retain longest-lane priority and bounded cohort size as explicit experimental
controls. Do not change the default lane count, admission window, or quantum:
the p95 improvement over the safe cohort is useful, but the remaining
staggered elapsed penalty versus lane `1` fails the default performance gate.

Operational rollback is leaving `batch_generator_longest_lane_step_quanta=1`
and `batch_generator_lane_target_size=0`, or setting
`batch_generator_max_lanes=1`. Source rollback is `git revert 5807641`.

Raw artifact:

- `iterations/artifacts/ITER-20260714-031-longest-lane-priority/cohort-priority-2.json`

## Next priority

Evaluate event-driven cohort closure or model-native fixed-batch isolation to
remove the admission and scheduling cost while preserving exact staggered
hash parity.
