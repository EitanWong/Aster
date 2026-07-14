# Loop Engineering Status

Updated: 2026-07-14

## Current State

- Current code commit: `07bd566` (avoid branch-only full prompt snapshots).
- Working tree: an uncommitted opt-in independent-MLX-stream candidate is
  present; it remains experimental and is not part of the default path.
- Previous dependency commit: `86ed15c` (refresh compatible dependency lock).
- Orthogonal baseline repair: `25067b8` (`fix: report continuous batching compatibility warning`).
- Dependency refresh: `1a0b993` (latest compatible MLX and serving package set).
- Manual runtime is the production path. `BatchGeneratorRuntimeKernel` remains an unavailable adapter boundary.
- The admission-before-prefill scheduler experiment was rolled back after randomized mixed and staggered A/B did not show a reliable short-request benefit.

## Evidence

- Full suite: `447 passed, 9 skipped, 1 warning` across 456 collected tests.
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
- BatchGenerator audit: with the installed `mlx-lm 0.31.3` API, the experimental `BatchedEngine` completed a 4-request 0.8B smoke and cleaned up cancellation. Prefix reuse is not correct yet: a second identical 196-token request recomputed its prompt and reported `prefill_cache_hit=false` because the current adapter does not pass stored caches into `BatchGenerator.insert()`; response cache flags are also hardcoded false. No serving change was made.
- BatchGenerator prefix restore is now implemented in the experimental engine: prompt-boundary cache extraction, cloned `caches=` insertion, correct cached-token history, terminal pin release, and response cache flags. The 0.8B 4-request repeated-prompt matrix improved hot median throughput from `204.0` to `261.6 tok/s` (`+28.2%`) and reduced elapsed from `1.255s` to `0.979s` (`-22.0%`) with identical hashes and unchanged `1.486 GB` peak. Exact 12,295-token reuse measured `5.725s -> 0.484s` on 0.8B and `35.755s -> 3.375s` on 9B, both with zero swap delta and exact greedy parity. Cancellation and streaming probes left zero pinned entries.
- Iteration 028 hardened the experimental BatchGenerator path after finding hybrid-cache batch invariance failures: active requests are now grouped by prompt length and cache profile. The corrected 0.8B 30-record on/off matrix had exact response-hash parity, zero errors, and zero swap delta. Warm cache-on elapsed improved `9%~34%` across reuse, mixed, divergent-reuse, staggered, and long workloads at concurrency 2/4/8. Structured JSON output passed at concurrency 2/4, and the 8-request cancellation probe left zero running/pinned entries. This is a conservative profile guard, not unrestricted continuous batching.
- Iteration 029 added opt-in bounded per-profile BatchGenerator lanes with `engine.batch_generator_max_lanes`, default `1`. Lane limit `2` preserved exact hashes for simultaneous mixed requests and improved their elapsed time by `2.90%~5.78%`, with unchanged `1.495 GB` peak and zero swap delta. Real staggered arrival still changed batch membership and produced hash drift in all four records, so lane `2` remains experimental and is not a default performance claim. Structured output, prefix reuse, cancellation, follow-up, and lane cleanup passed their probes.
- Iteration 030 added an opt-in cohort window and lane sealing. With lane `2` and a `160ms` window applied only to isolated secondary lanes, the 8-record mixed/staggered matrix restored exact hash parity, zero errors/swap, and `1.495 GB` peak; elapsed improved `0.19%~4.99%`. Staggered p95 increased `9%~12%`, so the default remains one lane. Multi-lane configurations without a positive window are now rejected.
- Iteration 031 added explicit cohort target sizing and longest-lane step quanta. With lane `2`, a `160ms` window, target size `3`, and quantum `2`, all 8 mixed/staggered records retained exact hashes against both lane `1` and quantum `1`, with zero errors/swap and `1.495 GB` peak. Staggered p95 improved `3.82%~6.42%` versus quantum `1`, but elapsed remained `18.13%~18.77%` slower than lane `1`; the controls remain opt-in and defaults are unchanged.
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
- A corrected cache-only microbatch probe found a narrow performance region: batch=4 with 128/256-token chunks had lower isolated model time than four serial calls, while 512/1024-token chunks had no stable gain. End-to-end 0.8B prefill256 A/B improved elapsed `22.818s -> 20.182s` (`-11.5%`) and throughput `22.439 -> 25.369 tok/s` (`+13.1%`), with peak memory `1.662 -> 1.931 GB`; however, greedy parity failed for one of four prompts (different text SHA). A 9B one-shot prefill256 probe improved `25.355s -> 24.358s` but raised peak `5.997 -> 6.622 GB`; no swap growth. The implementation was rolled back because deterministic correctness is a hard gate.
- Direct benchmark records now include MLX allocator peak memory; the 9B single smoke measured 5.169 GB and the 12K run measured 12.187 GB.
- `powermetrics` is unavailable without superuser privileges. `memory_pressure` reported 58% system-wide free memory and no thermal/performance warning was recorded by `pmset`.
- Iteration 032 evaluated a separate MLX stream per opt-in BatchGenerator
  lane. The matched 8-record A/B had zero errors, exact hashes, zero swap
  growth, and unchanged `1.495 GB` peak MLX memory, but elapsed improvement
  was only `0.84%~2.53%`. Interleaved reruns did not clear the 3% gate and
  staggered p95 once regressed `4.79%`; the candidate is not promoted.
- Iteration 033 evaluated event-driven cohort closure and rolled it back:
  mixed elapsed improved `0.89%~2.77%`, but staggered elapsed regressed
  `24.50%~27.73%`, p95 `7.98%~11.22%`, and completion throughput about 20%.
  The cause was repeated single-request lanes after staggered arrivals.
- Iteration 034 evaluated a greedy batch-wide argmax path and rolled it back:
  six candidate runs were `1.07%` slower and `1.02%` lower throughput than
  three baselines at manual decode batch 4.
- Iteration 035 keeps the disabled-prefix chat reuse guard: 40-turn Agent
  encoding improved `73.136ms -> 1.787ms`, and five end-to-end runs improved
  median elapsed `2.7199s -> 1.8380s` with identical output hash and finish
  reason.
- Iteration 036 keeps a bounded chat prompt token/reuse-point LRU. On a
  prefix-enabled 40-turn Agent workload, repeated encode time improved
  `74.082ms -> 0.028ms`; fresh-process e2e medians improved `29.8%` for exact
  hot reuse and `6.1%` for append-only turns, with cold latency `1.6%` lower.
- Iteration 037 keeps only the most recent eight chat snapshot reuse points by
  default. In a fresh-process 40-turn Agent A/B, cold latency improved
  `2.5375s -> 1.8549s` (`-26.9%`), while exact/append/branch hashes and cache
  hits remained identical. Snapshot memory fell from `1.192 GB` / 39 entries
  to `0.326 GB` / 9 entries (`-72.7%`), with zero swap growth.
- Iteration 038 adds a sparse older tier only for prompts at least 2048 tokens.
  In an 80-turn Agent A/B, cold latency improved `3.6749s -> 2.2592s`
  (`-38.5%`), initial snapshot memory fell `3.092 GB -> 0.628 GB`
  (`-79.7%`), and exact/recent/mid/old branches all retained cache hits and
  output hashes. A 40-turn probe stayed on the recent-only path, avoiding the
  short-chat memory and latency regression seen in the first sparse trial.
- Iteration 039 skips full-prompt snapshots for non-exact prefix-hit branches.
  In a sustained 80-turn / 12-branch A/B, post-recovery snapshot memory fell
  `1.511 GB -> 0.739 GB` (`-51.1%`) with identical hashes, hits, and saved
  tokens. Randomized 4-seed pairing showed branch deltas of `-0.17%~+0.49%`,
  append `+0.63%`, and recovery `+0.51%`; the initial grouped branch variance
  did not reproduce.
- Iteration 040 resets the active priority to core manual-runtime work. A
  prefix-off Qwen3.5-0.8B baseline at concurrency 4 measured `5.307s` and
  `31.16` average generation tok/s for the 4,820-token long workload, with
  `1.626GB` peak MLX memory and zero swap growth. DFlash remains reference-only.

## Active Risks

- The new paged-attention benchmark randomizes A/B order and records allocator peak memory, but it is a synthetic kernel probe rather than a full model serving benchmark; failed-request allocator data and energy remain unavailable.
- The 9B/32K mixed-agent matrix and sustained-run matrix are not yet complete; long-context prefill still incurs substantial transient memory and swap costs.
- The bounded chat snapshot policy has been measured at 40 and 80 turns, but
  sustained longer runs and more branch-diverse traces may need a different
  recent/sparse budget or retention policy.
- The tiered policy trades old-branch reuse depth for memory: in the 80-turn
  probe mid/old branches saved fewer tokens and were slower than unlimited
  snapshots, although they still hit and preserved output parity.
- Skipping branch-only full snapshots reduces sustained memory; randomized
  sustained ordering found no material branch-latency regression. Cold/exact
  deltas remain about `+2.13%/+1.34%` and should be monitored.
- Paged KV ownership, a persistent GPU block pool, and a block-indexed Metal contract are experimental boundaries. BatchGenerator exact/strict-prefix cache restore now works in `BatchedEngine`, while per-profile lanes, cohort windows, and lane priority remain opt-in because the safe multi-lane path still carries a staggered elapsed cost. Broader model/mask/batch coverage, lower-cost deterministic cohort closure, SSD tiering, KV quantization, and the separate `BatchGeneratorRuntimeKernel` serving adapter remain incomplete.

## Next Priority

Core-first priority: do not integrate DFlash or re-enable manual prefill batching until the hybrid `ArraysCache + KVCache` batched-state parity issue is resolved. First profile and improve the manual runtime's long/concurrent prefill-decode path, KV ownership, scheduling, and correctness gates; then evaluate model-native fixed-shape padding/masking or BatchGenerator state isolation. Do not create repeated same-profile single-request lanes. Keep the dependency lock aligned with project declarations on each package refresh.
