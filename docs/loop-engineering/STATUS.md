# Loop Engineering Status

Updated: 2026-07-14

## Current State

- Current commit: `b721554` (reuse merged decode cache across batch steps).
- Previous dependency commit: `86ed15c` (refresh compatible dependency lock).
- Orthogonal baseline repair: `25067b8` (`fix: report continuous batching compatibility warning`).
- Dependency refresh: `1a0b993` (latest compatible MLX and serving package set).
- Manual runtime is the production path. `BatchGeneratorRuntimeKernel` remains an unavailable adapter boundary.
- The admission-before-prefill scheduler experiment was rolled back after randomized mixed and staggered A/B did not show a reliable short-request benefit.

## Evidence

- Full suite: `418 passed, 9 skipped, 1 warning` across 427 collected tests.
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
- Package refresh on 2026-07-14: `uv lock --upgrade` resolved 72 compatible packages, including `mlx 0.32.0`, `mlx-lm 0.31.3`, `mlx-audio 0.4.5`, `fastapi 0.139.0`, `numpy 2.5.1`, `uvicorn 0.51.0`, and `transformers 5.12.1`. `transformers 5.13.1` remains excluded by the current `mlx-audio` and project bound `<5.13.0`. `pydub` and `python-multipart` were added to `pyproject.toml` after a locked sync exposed that they were only present in `requirements.txt`; `pip check` and `uv lock --check` pass.
- A production-shaped 0.8B manual-runtime baseline completed without swap growth: 2,229 prompt tokens took `2.638s` at `48.52` completion tok/s with `1.677 GB` MLX peak / `0.999 GB` active; 8,373 prompt tokens took `5.279s` at `24.25` completion tok/s with `2.297 GB` peak / `1.277 GB` active. These are baselines, not an optimization claim.
- Paged KV lifecycle probing showed `2,097,152` pool bytes retained after a child fork was released, then `0` pool bytes and `0` manager allocated blocks after the source bundle was released. After `mx.clear_cache()`, active MLX memory fell to `16` bytes in the isolated probe.
- An opt-in hybrid prompt-cache boundary now preserves Qwen3.5's `ArraysCache + KVCache` list shape, deep-copies recurrent state on fork, and releases full-attention pools during request cleanup. Native and opt-in greedy parity matched exactly for a 10-token prompt and 32-token completion.
- The same-model opt-in A/B did not pass the performance gate: at 2,229/8,373 prompt tokens elapsed time regressed by `19.9%/39.0%`, peak MLX memory increased from `1.677/2.297 GB` to `2.285/10.681 GB`, and both paths completed successfully without swap growth.
- Reusing a geometrically grown contiguous fallback removed the repeated full-table concatenation: the 8,373-token opt-in path fell to `5.420s` and `2.471 GB` peak in a single run. Randomized 3×3 A/B measured native median `5.448s` versus paged median `5.425s` (`-0.4%`, below the 3% gate), with paged peak memory still `2.471 GB` versus native `2.297 GB`.
- Lazy pool promotion now keeps the opt-in serving path storage-only until a block-indexed consumer requests `block_pool()`. The 8,373-token randomized 3×3 A/B measured native median `5.4541s` versus paged median `5.4526s` (`-0.03%`, below the 3% gate); peak MLX memory was `2.297 GB` versus `2.374 GB` (`+3.38%`). Greedy output parity and zero swap delta held across the probe.
- Step-bounded fallback growth removes the final 8K chunk's geometric overshoot: native KV ended at capacity `10240`, while paged materialized capacity ended at `8373`. Randomized 8K 3×3 A/B now measures native `5.4353s` versus paged `5.4259s` (`-0.17%`), with peak memory `2.297 GB` versus `2.286 GB` (`-0.46%`). This clears the memory regression but remains below the 3% speed gate.
- The paged Metal boundary now partitions long KV scans across 32 simdgroups and reduces partial online-softmax states. Qwen3.5-shaped kernel medians are `0.880x`, `0.976x`, and `0.733x` of native at 512/2K/8K tokens, with max absolute differences `0`, `0`, and `3.05e-05`. This is a kernel-level result only; serving still uses contiguous MLX-LM SDPA.
- An explicitly disabled Qwen3.5 direct-attention bridge now uses the pool kernel for decode (`Q<=8`) and native SDPA for long prefill. Randomized 8K direct/native A/B measured `5.4561s/5.4423s` (`+0.25%`) and `2.286/2.297 GB` peak memory (`-0.46%`), with exact greedy parity and zero swap delta. It is functional and memory-neutral, but not a speed win.
- The direct bridge now reuses a cached `uint32` block-index tensor until block-table or COW topology changes. A fresh randomized 8K 3x3 A/B measured native/direct elapsed medians `5.4306s/5.4597s` (`+0.54%`) and completion throughput `23.570/23.445 tok/s` (`-0.53%`), with unchanged peak memory `2.297/2.286 GB`; all six requests completed and the result does not clear the 3% speed gate.
- Native manual decode now keeps a persistent merged cache across stable multi-request steps, returning lightweight per-request references and remerging only after batch membership changes. In a 0.8B no-prefix 4-request A/B, batch=4 improved from `19.460` to `29.476 tok/s` (`+51.5%`) and reduced elapsed median from `26.310s` to `17.371s` (`-34.0%`) with unchanged `1.829 GB` peak and zero failures/swap growth. Randomized 9B 2x2 A/B measured batch=2 at `13.576 tok/s / 37.715s` versus batch=4 at `23.247 tok/s / 22.025s` (`+71.2%` throughput, `-41.6%` elapsed); peak memory was `6.256/6.220 GB`, with zero failures and zero swap delta. Greedy response hashes, token counts, and finish reasons matched batch=1 exactly; mixed and staggered membership-change probes completed 4/4.
- A controlled native prefill-batching experiment was rolled back: 0.8B 4x8K baseline elapsed was `17.390s` at `29.442 tok/s`, while batch=4 prefill was `23.423s` at `21.859 tok/s` with `12.886 GB` peak and `0.93 GiB` swap growth; batch=2 was `20.892s` at `24.507 tok/s` with `3.282 GB` peak. The root cause is activation and merged-cache memory scaling with batch times chunk size; simple prefill batching does not pass the speed or memory gates.
- Direct benchmark records now include MLX allocator peak memory; the 9B single smoke measured 5.169 GB and the 12K run measured 12.187 GB.
- `powermetrics` is unavailable without superuser privileges. `memory_pressure` reported 58% system-wide free memory and no thermal/performance warning was recorded by `pmset`.

## Active Risks

- The new paged-attention benchmark randomizes A/B order and records allocator peak memory, but it is a synthetic kernel probe rather than a full model serving benchmark; failed-request allocator data and energy remain unavailable.
- The 9B/32K mixed-agent matrix and sustained-run matrix are not yet complete; long-context prefill still incurs substantial transient memory and swap costs.
- Paged KV ownership, a persistent GPU block pool, and a block-indexed Metal contract are experimental boundaries. Hybrid bundle ownership now has opt-in lazy promotion and a Qwen3.5-only direct bridge, but default integration, broader model/mask/batch support, SSD tiering, KV quantization, and the MLX-LM BatchGenerator serving adapter remain incomplete.

## Next Priority

Build a memory-aware prefill microbatch benchmark across chunk sizes and batch sizes before implementing another prefill path; keep the current serial prefill fallback and validated native decode batching, and require parity, lifecycle, memory, and randomized multi-trial evidence for any further default change. Keep the dependency lock aligned with project declarations on each package refresh.
