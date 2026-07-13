# Loop Engineering Status

Updated: 2026-07-13

## Current State

- Current commit: `4de13e1` (hybrid-attention memory accounting and allocator benchmark metrics).
- Orthogonal baseline repair: `25067b8` (`fix: report continuous batching compatibility warning`).
- Dependency refresh: `1a0b993` (latest compatible MLX and serving package set).
- Manual runtime is the production path. `BatchGeneratorRuntimeKernel` remains an unavailable adapter boundary.
- The admission-before-prefill scheduler experiment was rolled back after randomized mixed and staggered A/B did not show a reliable short-request benefit.

## Evidence

- Full suite: `386 passed, 9 skipped, 1 warning`.
- Runtime, cache, scheduler, and benchmark suites: `55 passed`.
- `compileall` and `git diff --check`: passed.
- The initial grouped 0.8B mixed A/B suggested `-13.6%` elapsed time, but randomized interleaving invalidated that as a global claim: current was `+2.86%` slower in elapsed median and `-2.78%` lower in completion throughput, with bootstrap intervals containing zero.
- The benchmark now defaults to explicit greedy sampling (`temperature=0.0`); seven validation trials all produced 288 completion tokens and 4/4 successful requests.
- Resource-aware validation now records platform, Python, MLX-LM, total memory, RSS peak, and swap before/after values; seven trials showed zero swap growth.
- The benchmark harness now includes staggered arrival and request-level latency diagnostics; the scheduler candidate is not retained.
- Exact prefix reuse is now measured separately from divergent LCP reuse; 9B produced one exact hit for the repeated workload and safely skipped divergent LCP matches for `ArraysCache`.
- The 9B long-context probe completed at 8,181 prompt tokens with 1.61 GiB swap growth, while 12,181, 16,181, and 30,181 token probes were rejected by memory pressure.
- Hybrid-attention accounting removed the false 9B admission rejects: fresh 12K, 16K, and 30K prompts all completed; the 30K run took 160.2s, reached 0.80 completion tok/s, and added 3.56 GiB swap.
- Direct benchmark records now include MLX allocator peak memory; the 9B single smoke measured 5.169 GB and the 12K run measured 12.187 GB.
- `powermetrics` is unavailable without superuser privileges. `memory_pressure` reported 58% system-wide free memory and no thermal/performance warning was recorded by `pmset`.

## Active Risks

- The direct benchmark does not yet randomize A/B order or collect MLX allocator-level peak memory for failed requests and every long-context artifact.
- The 9B/32K mixed-agent matrix and sustained-run matrix are not yet complete; long-context requests now run past 30K but incur severe swap and tail-latency costs.
- Paged KV, SSD tiering, KV quantization, and the MLX-LM BatchGenerator serving adapter remain incomplete.

## Next Priority

Reduce actual long-context swap and tail latency: compare prefill-time allocator sampling, prefix eviction, and paged/quantized KV candidates at 12K, 16K, and 30K before attempting 32K mixed-agent traffic.
