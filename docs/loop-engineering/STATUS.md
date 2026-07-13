# Loop Engineering Status

Updated: 2026-07-14

## Current State

- Current commit: `f062efc` (reproducible persistent paged-attention benchmark).
- Orthogonal baseline repair: `25067b8` (`fix: report continuous batching compatibility warning`).
- Dependency refresh: `1a0b993` (latest compatible MLX and serving package set).
- Manual runtime is the production path. `BatchGeneratorRuntimeKernel` remains an unavailable adapter boundary.
- The admission-before-prefill scheduler experiment was rolled back after randomized mixed and staggered A/B did not show a reliable short-request benefit.

## Evidence

- Full suite: `401 passed, 9 skipped, 1 warning` across 410 collected tests.
- Runtime, cache, scheduler, and benchmark suites: `55 passed`.
- `compileall` and `git diff --check`: passed.
- The initial grouped 0.8B mixed A/B suggested `-13.6%` elapsed time, but randomized interleaving invalidated that as a global claim: current was `+2.86%` slower in elapsed median and `-2.78%` lower in completion throughput, with bootstrap intervals containing zero.
- The benchmark now defaults to explicit greedy sampling (`temperature=0.0`); seven validation trials all produced 288 completion tokens and 4/4 successful requests.
- Resource-aware validation now records platform, Python, MLX-LM, total memory, RSS peak, and swap before/after values; seven trials showed zero swap growth.
- The benchmark harness now includes staggered arrival and request-level latency diagnostics; the scheduler candidate is not retained.
- Exact prefix reuse is now measured separately from divergent LCP reuse; 9B produced one exact hit for the repeated workload and safely skipped divergent LCP matches for `ArraysCache`.
- The 9B long-context probe completed at 8,181 prompt tokens with 1.61 GiB swap growth, while 12,181, 16,181, and 30,181 token probes were rejected by memory pressure.
- Hybrid-attention accounting removed the false 9B admission rejects: fresh 12K, 16K, and 30K prompts all completed.
- Bounding automatic prefix snapshots reduced 30K stores from 53 to 8 at cap 4K and to 1 at default cap 0; the cap-0 run took 92.6s, reached 1.38 completion tok/s, and added 1.10 GiB swap.
- Prefill memory is now separately observable: default cap-0 12K measured 9.121 GB peak / 6.866 GB active; 30K measured 12.124 GB peak / 6.149 GB active. Overall request peaks matched prefill peaks.
- 4-bit/8-bit MLX KV prototypes were evaluated and not adopted: 4-bit changed fixed greedy output, while 8-bit showed no material gain in the measured 12K trial.
- Native KV growth step 2048 reduced single-trial 12K latency to 33.2s and 30K latency to 79.8s with exact greedy smoke parity; peak memory was unchanged, so this is an allocation-copy optimization rather than a paged-KV solution.
- The experimental paged KV adapter now writes full-attention K/V into fixed blocks with reference-counted table forks and COW; Qwen3.5-0.8B 2K chunked prefill matched native logits exactly (`max_abs_logit_difference=0.0`).
- The adapter's contiguous materialization fallback did not clear the 3% performance gate: 2K median was `1.29%` slower and 8K median was statistically flat (`0.03%` slower); it is not enabled by default.
- A block-indexed `mx.fast.metal_kernel` now consumes persistent physical block pools and logical block indices with GQA and causal offsets. A tiled SIMD path reduces duplicate softmax work and reaches Qwen3.5-shaped FP16 parity at or below `3.1e-05` max absolute difference for 512/2K/8K probes. The corrected dispatch benchmark is still slower than native attention: median ratios were `1.56x`, `3.42x`, and `7.44x` in the recorded run, so it remains disabled.
- The persistent pool removes per-call `mx.stack` packing and preserves per-layer COW data when a shared block table forks. It is an experimental storage boundary; pool capacity and release lifecycle are not yet integrated with serving.
- Package audit on 2026-07-14: direct MLX/MLX-LM/MLX-Audio/FastAPI packages were already at the current PyPI versions (`mlx 0.32.0`, `mlx-lm 0.31.3`, `mlx-audio 0.4.5`, `fastapi 0.139.0`); `transformers 5.12.1` is the latest version compatible with the current `mlx-audio` upper bound `<5.13.0`. `pip check` passed. No dependency change was justified this iteration.
- Direct benchmark records now include MLX allocator peak memory; the 9B single smoke measured 5.169 GB and the 12K run measured 12.187 GB.
- `powermetrics` is unavailable without superuser privileges. `memory_pressure` reported 58% system-wide free memory and no thermal/performance warning was recorded by `pmset`.

## Active Risks

- The new paged-attention benchmark randomizes A/B order and records allocator peak memory, but it is a synthetic kernel probe rather than a full model serving benchmark; failed-request allocator data and energy remain unavailable.
- The 9B/32K mixed-agent matrix and sustained-run matrix are not yet complete; long-context prefill still incurs substantial transient memory and swap costs.
- Paged KV ownership, a persistent GPU block pool, and a block-indexed Metal contract are experimental boundaries. The tiled kernel misses the 3% gate beyond the shortest decode case; hybrid-cache bundle fork/release, pool reclamation, SSD tiering, KV quantization, and the MLX-LM BatchGenerator serving adapter remain incomplete.

## Next Priority

Benchmark a production-shaped long-context path with the native attention baseline, then decide whether to continue kernel optimization or prioritize hybrid-cache bundle fork/release and pool reclamation before attempting 32K mixed-agent traffic.
