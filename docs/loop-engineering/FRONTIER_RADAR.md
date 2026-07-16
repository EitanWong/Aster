# Local Inference Frontier Radar

Updated: 2026-07-17

This radar tracks inference papers and implementations that could improve
Aster's Apple Silicon core. Recency is not an admission criterion. A mechanism
enters Aster only after source and license review, a fixed-version local
reproduction, deterministic correctness, resource-aware A/B, stress/corner
coverage, and a rollback path.

## Intake rules

1. Prefer author repositories, paper artifacts, and mature engine code over
   secondary summaries.
2. Separate kernel, model, scheduler, and end-to-end claims. Never promote a
   lower-layer speedup as an engine result.
3. Reproduce one variable at a time on the same machine, model, quantization,
   prompt/output, and randomized or interleaved order.
4. Record latency, TTFT/TPOT, throughput, peak MLX/RSS, swap delta, output
   tokens/hash, failures, cancellation cleanup, and thermal context.
5. Require at least a 3% repeatable end-to-end speed win or a material memory
   reduction with no correctness/resource regression.
6. Treat closed cores, missing licenses, missing code, and hardware-specific
   CUDA/RDMA paths as evidence or watch items, not code sources.

## Current candidates

| Priority | Work | What is useful | Local status | Decision / next gate |
| --- | --- | --- | --- | --- |
| P0 | [vllm-metal](https://github.com/vllm-project/vllm-metal), commit `4c18ee0`, Apache-2.0 | Fused K/V scatter, lazy MLX C++ Primitive, unified varlen paged attention, hybrid Qwen3.5 handling | Split-KV, attention-boundary, and fused-scatter reproductions complete | Reference scatter wins in its own layout, but Aster transfer fails. Retain as evidence; stop direct operator imports. |
| P0 | [Uzu](https://github.com/trymirai/uzu), commit `15b8e73`, MIT | Native Rust/Metal command ownership, explicit GPU timing, traceable graphs, quantized kernels, DFlash integration | Pinned under `examples/`; source audit started; Rust toolchain not yet installed | Use as native-runtime ceiling and ownership reference. Benchmark same Qwen3.5 model before considering a backend boundary. |
| P0 | [OMLX](https://github.com/jundot/omlx) TurboQuant at `e3a4fe4`, pinned mlx-vlm `78b96eb`, and [Open-TQ-Metal](https://arxiv.org/abs/2604.16957) | Compressed-domain K/V attention, hybrid-cache conversion, long-context capacity, two-pass decode | `51/51` reference tests plus 5-process kernel and 20-process Qwen3.5 matrices complete | Reject measured 4-bit path: cache shrinks, but default MLX speed and model quality fail. Preserve as a capacity reference only. |
| P1 | [Native LLM and MLLM Inference at Scale on Apple Silicon](https://arxiv.org/abs/2601.19139) / [vllm-mlx](https://github.com/waybarrios/vllm-mlx) | Production-shaped MLX batching, prefix reuse, lifecycle | Existing pinned reference and extensively cross-checked | Continue using for scheduler and lifecycle parity. |
| P1 | [DFlash](https://github.com/z-lab/dflash) and the two MLX ports already under `examples/` | Parallel draft/verify and rollback for diffusion-style speculation | References cloned; not admitted | Defer until cache ownership and batch-state parity are stable. Require acceptance and real load A/B. |
| P1 | [SSSD](https://github.com/huawei-csl/sssd_speculator), ACL 2026, BSD-3-Clause-Clear | Training-free suffix-array/prompt/self-output speculation | Source/license verified remotely; not cloned | Later candidate after core: compare against prompt lookup and DFlash without a draft model. |
| P1 | [CONCUR](https://arxiv.org/abs/2601.22705) | Agent-level cache-pressure feedback and proactive admission | Paper found; no author code located in first pass | Reproduce only after a sustained Agent KV-thrashing workload exists. |
| P2 | [LONGSPEC](https://aclanthology.org/2026.acl-long.83/) | Constant-size long-context drafter and hybrid tree verification | Paper evidence only; CUDA/Triton implementation assumptions | Watch. Not compatible with the current core without training and a tree verifier. |
| P2 | [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580) | Production-grade evidence that verification/load can erase speculative gains | Used as a gating reference | Require load-adaptive measurements; never enable speculation from batch-1 results alone. |
| Benchmark only | [BaseRT paper](https://arxiv.org/abs/2607.00501) / [repository](https://github.com/basecompute/baseRT) | Native Metal ceiling and dispatch/fusion hypotheses | Public CLI/format are Apache-2.0, but the inference engine is a proprietary binary | Black-box competitor only; do not borrow or call it open-core evidence. |
| Rejected 4-bit | [gemma4metal](https://github.com/mutable-state-inc/gemma4metal), commit `0f09466`, MIT | Public Open-TQ-Metal C++/Metal implementation and compressed-cache formulas | Source audited locally; native build reaches the Metal compiler step but the separate Apple Metal Toolchain component is absent | Public test coverage/error threshold is too weak for admission. Keep the ignored local clone as a source reference; use the stronger OMLX/model evidence. |
| Quarantine | [mlx-inference-bench](https://github.com/AtomGradient/mlx-inference-bench) | Useful negative results for speculative decoding and bottleneck profiling | No license and no independent validation; zero-star WIP at intake | Read-only hypothesis source; do not copy code or promote claims. |

## Reproduced result: vllm-metal split-KV

Environment: Apple M5, 24GB unified memory, macOS 27.0, Python 3.13.12,
MLX 0.32.0, vllm-metal `4c18ee0`. The reference uses 512-token partitions
and reports an M5 occupancy threshold of 80 threadgroups.

The unmodified correctness matrix passed all 19 cases. For performance, one
temporary benchmark-only runtime switch forced single-pass execution in the
same compiled binary; three 30-warmup/200-measurement process pairs used the
same Qwen3 shape (`Hq=16`, `Hkv=8`, `D=128`). Positive deltas mean adaptive
split was slower.

| KV tokens | Batch 1 | Batch 2 | Batch 4 | Batch 5 control | Batch 8 control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | +3.52% | -2.18% | +6.61% | -1.03% | +3.32% |
| 8,192 | +7.27% | +30.93% | +69.63% | -1.83% | +0.22% |

The occupancy gate therefore does not transfer to this M5. Aster's existing
kernel retained max absolute error <= `6.10e-05`; at 8K it was `0.67%~25.33%`
faster than native MLX SDPA and `6%~32%` faster than vllm-metal's single-pass
kernel across batch 1/2/4/5/8. Different block layouts prevent an overall
engine claim. The remaining opportunity is graph/cache integration, not a
replacement attention algorithm.

## Reproduced result: attention Primitive boundary

A guarded FP16 `D=V=128` MLX C++ Primitive retained zero cross-path error and
at most `6.10e-05` error against independent native MLX over 18 block/causal
corners. A second random-data check covered every timed 2K/8K batch shape at
`7.63e-06` maximum error. Five fresh processes used randomized cell order, 30
warmups, and 200 measurements for 2K/8K x batch 1/2/4/8. The primary control
used direct cached `mx.fast` with the same physical-block guard and invalid
reduction as the Primitive. Five intervals excluded zero in favor of the
Primitive, but no interval established a >=3% gain. Two cells had nominal
>=3% medians; in a five-process confirmation 2K/batch-1 fell to a 2.13% gain
and 8K/batch-2 reversed to a 3.51% regression.

Five-process 32K and 64K stress intervals all crossed zero. The guarded control
showed that the earlier unguarded 64K regression mixed boundary and guard work,
so it is not retained as a Primitive claim. Peak probe memory was at most
`268,813,526 B`, post-clear active memory was `16 B`, and swap delta was zero
across 22 archived process records. Matching MLX 0.32.0 also required nanobind
2.13.0; 2.10.2 linked but could not share MLX array casters. Source and raw
results are preserved under the Iteration 047 artifact directory. The private
ABI and packaging cost is therefore not admitted.

## Reproduced result: fused K/V scatter

Three layers were cross-validated on the same M5. A pure MLX combined K/V
storage included its real `mx.stack` cost; it established no gain and showed a
directional 64-token batch-2 regression. Even a pre-stacked ceiling did not
establish a stable 3% gain.

The pinned vllm-metal `reshape_and_cache` Primitive was then rebuilt with its
exact MLX 0.32.0/nanobind 2.10.2 pins. FP16/BF16/FP32, sparse slots, and
negative padding slots were byte-identical to two MLX scatters. It improved 64
and 128 token paired effects by `8.35%` and `11.65%`, with intervals
`[-11.60%, -3.98%]` and `[-15.40%, -10.15%]`; 1/4/8/16 tokens also cleared the
3% gate. The mechanism is valid for the
reference's token-contiguous scheduler-owned layout.

An Aster-layout standalone Primitive retained exact full-pool parity across
start/end offsets, rotated writes, timed loops, alias lifetime checks, and lazy
chaining. It supported unequal K/V dimensions and rejected invalid blocks,
offsets, dtype, rank, real/spoofed Python objects, cache shapes, and overlapping
storage before dispatch. It did not transfer the gate. No cell repeated a >=3%
gain in both groups; 64-token single-request confirmation was only `0.85%`
faster and crossed zero. Batch 4/8 instead confirmed `7.10%/8.22%` regressions;
batch 8 cleared the regression gate in both groups. A 1,000-iteration matrix
had zero post-loop error/swap growth, no thermal warning, and a `52,428,824 B`
peak. No runtime or private ABI was retained.

## Reproduced result: real paged graph and TurboQuant

The complete opt-in paged graph was measured in Qwen3.5-0.8B, prefix-off,
greedy batch 1. Ten fresh native/direct controls per context retained exact
tokens and text. Direct elapsed changed `+0.37%` at 2,229 prompt tokens and
`+0.20%` at 8,373; generation changed `-0.58%/+1.94%`. All intervals crossed
zero. Peak allocator memory
fell only `9.12/6.09 MB`. Five profile processes put sampled-token sync at
about `6.97 ms` of an `8.08 ms` median decode step; direct attention and pool
write enqueue totals were only a few milliseconds over all 64 tokens.

TurboQuant's fused kernel was then tested at `B=1,Hq=8,Hkv=2,Q=1,D=256` in
five fresh processes, 30 warmups, and 200 interleaved calls per method at
2K/8K/32K/64K. It reduced isolated cache bytes `3.94x` and beat Aster paged by
`18%~60%`, but versus default MLX it was `3.47%` slower at 2K, inconclusive at
8K, `34.13%` slower at 32K, and `25.08%` slower at 64K. Fused/dequant error
stayed <=`1.53e-5`; swap stayed zero.

The model gate used 20 fresh Qwen3.5 processes and a hash-fixed WikiText-2
windows. At 2K, only 3/5 greedy windows matched, teacher top-1 fell to
`89.06%`, and decode was `5.22%` slower. At 8K, 3/5 windows matched, PPL
changed by up to `3.38%`, and decode was `5.72%` slower. Whole hybrid
cache storage fell `1.72x/2.67x`; model-weight-dominated peak did not improve.
The 4-bit path is rejected rather than introduced as a capacity switch.

## Next reproduction

Prove whether post-sample `_eval_cache()` is required after token materialization.
Use a 10,000-step hybrid-cache RAW/WAW stress matrix before changing runtime
behavior, and compare exact cache arrays, tokens, trim/COW/cancellation, peak
memory, and swap. Uzu's explicit command ownership and vllm-metal's lazy graph
provenance remain the reference. A shape-specific split-KV probe is secondary;
it must beat native MLX, not only Aster's slower paged kernel.
