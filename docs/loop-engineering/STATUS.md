# Loop Engineering Status

Updated: 2026-07-13

## Current State

- Current commit: `aa2a8f4` (`docs: record admission scheduling loop`); scheduler code commit: `32addf1`.
- Orthogonal baseline repair: `25067b8` (`fix: report continuous batching compatibility warning`).
- Manual runtime is the production path. `BatchGeneratorRuntimeKernel` remains an unavailable adapter boundary.
- The first loop iteration improved mixed scheduling by admitting waiting requests before the next prefill step and prioritizing those new requests ahead of existing prefill continuations.

## Evidence

- Full suite: `376 passed, 9 skipped`.
- Affected scheduler suite: `52 passed`.
- `compileall` and `git diff --check`: passed.
- Seven-trial 0.8B mixed A/B: median elapsed time `3.8727s -> 3.3468s` for equal-token samples (`-13.6%`); completion throughput `74.367 -> 86.051 tok/s` (`+15.7%`).
- This is recorded as a mixed-workload scenario result, not a global performance claim. The benchmark uses stochastic sampling and the A/B groups were run sequentially.
- `powermetrics` is unavailable without superuser privileges. `memory_pressure` reported 58% system-wide free memory and no thermal/performance warning was recorded by `pmset`.

## Active Risks

- The direct benchmark does not yet fix sampling deterministically, randomize A/B order, or collect per-process RSS/MLX memory for every trial.
- The 9B/32K mixed-agent matrix and sustained-run matrix are not yet complete.
- Paged KV, SSD tiering, KV quantization, and the MLX-LM BatchGenerator serving adapter remain incomplete.

## Next Priority

Make the benchmark runner produce fixed-sampling, randomized A/B records with hardware/software metadata and explicit memory fields, then rerun the scheduler candidate under that harness before accepting it as a default profile.
