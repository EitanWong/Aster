# Loop Engineering Status

Updated: 2026-07-13

## Current State

- Current commit: `96ff8a6` (`perf: record benchmark resource metadata`).
- Orthogonal baseline repair: `25067b8` (`fix: report continuous batching compatibility warning`).
- Manual runtime is the production path. `BatchGeneratorRuntimeKernel` remains an unavailable adapter boundary.
- The first loop iteration improved mixed scheduling by admitting waiting requests before the next prefill step and prioritizing those new requests ahead of existing prefill continuations.

## Evidence

- Full suite: `376 passed, 9 skipped`.
- Affected scheduler suite: `52 passed`.
- `compileall` and `git diff --check`: passed.
- Seven-trial 0.8B mixed A/B: median elapsed time `3.8727s -> 3.3468s` for equal-token samples (`-13.6%`); completion throughput `74.367 -> 86.051 tok/s` (`+15.7%`).
- The benchmark now defaults to explicit greedy sampling (`temperature=0.0`); seven validation trials all produced 288 completion tokens and 4/4 successful requests.
- Resource-aware validation now records platform, Python, MLX-LM, total memory, RSS peak, and swap before/after values; seven trials showed zero swap growth.
- The scheduler result remains a mixed-workload scenario result, not a global performance claim. A/B ordering and per-trial resource collection are still incomplete.
- `powermetrics` is unavailable without superuser privileges. `memory_pressure` reported 58% system-wide free memory and no thermal/performance warning was recorded by `pmset`.

## Active Risks

- The direct benchmark does not yet randomize A/B order or collect MLX allocator-level peak memory for every trial.
- The 9B/32K mixed-agent matrix and sustained-run matrix are not yet complete.
- Paged KV, SSD tiering, KV quantization, and the MLX-LM BatchGenerator serving adapter remain incomplete.

## Next Priority

Make the benchmark runner randomize A/B order and add MLX allocator-level memory when available, then rerun the scheduler candidate under that harness before accepting it as a default profile.
