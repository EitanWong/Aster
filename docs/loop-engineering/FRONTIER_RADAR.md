# Local Inference Frontier Radar

Updated: 2026-07-15

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
| P0 | [vllm-metal](https://github.com/vllm-project/vllm-metal), commit `4c18ee0`, Apache-2.0 | Fused K/V scatter, lazy MLX C++ Primitive, unified varlen paged attention, hybrid Qwen3.5 handling | Pinned under `examples/`; split-KV correctness `19/19`; M5 performance A/B complete | Reject current split gate. Reproduce fused scatter + Primitive around Aster's kernel. |
| P0 | [Uzu](https://github.com/trymirai/uzu), commit `15b8e73`, MIT | Native Rust/Metal command ownership, explicit GPU timing, traceable graphs, quantized kernels, DFlash integration | Pinned under `examples/`; source audit started; Rust toolchain not yet installed | Use as native-runtime ceiling and ownership reference. Benchmark same Qwen3.5 model before considering a backend boundary. |
| P1 | [Native LLM and MLLM Inference at Scale on Apple Silicon](https://arxiv.org/abs/2601.19139) / [vllm-mlx](https://github.com/waybarrios/vllm-mlx) | Production-shaped MLX batching, prefix reuse, lifecycle | Existing pinned reference and extensively cross-checked | Continue using for scheduler and lifecycle parity. |
| P1 | [DFlash](https://github.com/z-lab/dflash) and the two MLX ports already under `examples/` | Parallel draft/verify and rollback for diffusion-style speculation | References cloned; not admitted | Defer until cache ownership and batch-state parity are stable. Require acceptance and real load A/B. |
| P1 | [SSSD](https://github.com/huawei-csl/sssd_speculator), ACL 2026, BSD-3-Clause-Clear | Training-free suffix-array/prompt/self-output speculation | Source/license verified remotely; not cloned | Later candidate after core: compare against prompt lookup and DFlash without a draft model. |
| P1 | [CONCUR](https://arxiv.org/abs/2601.22705) | Agent-level cache-pressure feedback and proactive admission | Paper found; no author code located in first pass | Reproduce only after a sustained Agent KV-thrashing workload exists. |
| P2 | [LONGSPEC](https://aclanthology.org/2026.acl-long.83/) | Constant-size long-context drafter and hybrid tree verification | Paper evidence only; CUDA/Triton implementation assumptions | Watch. Not compatible with the current core without training and a tree verifier. |
| P2 | [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580) | Production-grade evidence that verification/load can erase speculative gains | Used as a gating reference | Require load-adaptive measurements; never enable speculation from batch-1 results alone. |
| Benchmark only | [BaseRT paper](https://arxiv.org/abs/2607.00501) / [repository](https://github.com/basecompute/baseRT) | Native Metal ceiling and dispatch/fusion hypotheses | Public CLI/format are Apache-2.0, but the inference engine is a proprietary binary | Black-box competitor only; do not borrow or call it open-core evidence. |
| Quarantine | [Open-TQ-Metal](https://arxiv.org/abs/2604.16957) | Compressed-domain int4 KV attention claim | No implementation repository found in first pass | Do not reproduce or adopt until author code and license are available. |
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

## Next reproduction

Build a standalone optional extension that keeps Aster's block-indexed kernel
but exposes it as an MLX Primitive and fuses K/V pool updates. First compare
the boundary against the current `mx.fast.metal_kernel` path with pre-resident
inputs. Only after a repeatable boundary win should it enter an opt-in Qwen3.5
bridge and a real 0.8B/9B mixed, staggered, long-context, cancellation, and
memory-pressure matrix.
