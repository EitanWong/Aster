# Iteration 034: Greedy Batch Argmax Fast Path

## Scope

- Starting code commit: `a268e69`
- Ending code commit: `a268e69` (candidate rolled back)
- Machine: macOS 27.0, Apple Silicon arm64, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`

## Hypothesis

Manual decode batch execution was applying `logsumexp`, the sampler, and
scalar extraction once per request even for deterministic greedy requests.
The candidate added a `DecodeWorkItem.greedy` marker and a batch-wide argmax
path, with a fallback for sampling and logits processors.

## Verification

The candidate passed the focused correctness test and a full suite of
`438 passed, 9 skipped, 1 warning`. Ruff, compileall, lock validation,
`pip check`, and `git diff --check` passed.

The matched manual-runtime workload used Qwen3.5-0.8B 4-bit, decode batch 4,
mixed workload, concurrency 4, greedy sampling, and no prefix cache. Three
baseline runs had median elapsed `1.6682s` and `172.642 tok/s`. Six candidate
runs had median elapsed `1.6860s` and `170.884 tok/s`; peak MLX memory was
`1.541 GB` and swap was unchanged in every run.

## Root cause and decision

The batch-wide argmax did not amortize its extra reduction/evaluation work on
this MLX workload. It regressed elapsed time by `1.07%` and throughput by
`1.02%`, so it failed the 3% gate and was rolled back. No source or test
changes from this experiment remain.

## Next priority

Measure preprocessing overhead in real Agent chat requests, then return to
model-native fixed-shape/state isolation for the unresolved staggered lane
bottleneck.
