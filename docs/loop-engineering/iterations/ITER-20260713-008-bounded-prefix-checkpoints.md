# Iteration 008: Bound Prefix Checkpoint Growth

- Iteration ID: `ITER-20260713-008`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Code commits: `817e808`, `3494bee`
- Runtime: Python 3.14.5, MLX `0.32.0`, MLX-LM `0.31.3`, manual engine
- Model: Qwen3.5-9B-4bit

## Problem and Hypothesis

`_maybe_checkpoint()` cloned the growing prompt cache at every prefill chunk
boundary. For a 30K prompt this produced 53 snapshots. Each snapshot retained a
larger cache, creating avoidable O(n^2) allocation and cache pressure. The
current Qwen3.5 `ArraysCache` cannot safely rewind divergent LCP matches, so
these opportunistic snapshots were not useful for the measured agent workload.

## Change

Add `engine.snapshot_chunk_checkpoint_max_tokens`:

- explicit chat/agent `reuse_points` are always preserved;
- the final full-prefix checkpoint is always preserved;
- automatic chunk-boundary checkpoints are bounded by this setting;
- the default is now `0`, disabling opportunistic checkpoints;
- models with a rewindable cache can opt in to a positive limit.

The setting is part of `EngineSettings`, so it is visible and reversible rather
than hidden in the scheduler. Tests cover the default, opt-in cap, and explicit
reuse point behavior.

## Benchmark Evidence

Fresh-process long runs used one active request, greedy sampling, 128 output
tokens, and `prefill_token_budget=512`.

| Prompt | Policy | Stores | MLX peak | Elapsed | Completion tok/s | Swap delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 12,181 | previous uncapped | 17 | 12.187 GB | 48.580s | 2.63 | +1.73 GiB |
| 12,181 | cap 4,096 | 8 | 9.951 GB | 52.757s | 2.43 | -0.18 GiB |
| 30,181 | previous uncapped | 53 | unavailable | 160.188s | 0.80 | +3.56 GiB |
| 30,181 | cap 4,096 | 8 | 12.955 GB | 93.486s | 1.37 | +1.46 GiB |
| 30,181 | cap 0 | 1 | 12.124 GB | 92.562s | 1.38 | +1.10 GiB |

The cap-0 result is the best measured policy for this Qwen3.5 workload: it
removes all opportunistic clones while retaining the final cache. The cap-4K
result demonstrates the configurable middle ground for models where arbitrary
prefix checkpoints are useful. Single-trial results are not a randomized A/B
claim; the reduction in stores is the direct causal diagnostic.

## Correctness and Cache Evidence

Fresh default-policy reuse runs completed without failures:

- exact `reuse`: 1 exact hit and 188 reused tokens;
- divergent `reuse-divergent`: 1 safe `unsafe_lcp_skip`, 0 false hits.

This confirms that disabling periodic checkpoints does not remove exact reuse
or explicit safety behavior for the current model.

## Control Experiments

- Setting the MLX memory limit to the device recommended working set
  (`19,069,665,280` bytes) did not reduce the 12K allocator peak (`12.187 GB`)
  or reliably reduce swap, so it was not adopted.
- Reducing prefill chunks from 512 to 128 increased the 12K MLX peak from the
  cap-4K result's `9.951 GB` to `13.208 GB` and elapsed time to `57.282s`; it
  was not adopted.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q aster scripts/dev/benchmark_live.py tests
.venv/bin/python -m pytest -q tests/test_engine_runtime.py -k checkpoint
```

The targeted checkpoint suite passed with `6 passed`. The final full suite
passed with `388 passed, 9 skipped, 1 warning` after the default-policy change.

## Conclusion and Next Priority

Keep both commits. The default policy materially reduces long-context snapshot
growth and improves the 30K end-to-end result, while exact and explicit agent
reuse remain available. The next bottleneck is actual prefill working-set and
swap behavior, not prefix snapshot cloning. Continue with MLX active-memory
sampling and a paged/quantized KV experiment only after lossless output and
memory-pressure gates are defined.
