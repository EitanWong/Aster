# Iteration 046 — Frontier Metal reproduction

Date: 2026-07-15
Baseline commit: `5a92477`

## Objective

Establish a source-ranked frontier intake and reproduce the first Apple
Silicon candidate before changing Aster. The selected mechanism was
vllm-metal's occupancy-gated split-KV decode because it directly targets
Aster's current gap between a fast block-indexed kernel and a neutral/slower
end-to-end paged path.

## Reference intake

- Added Uzu `15b8e73` (MIT) and vllm-metal `4c18ee0` (Apache-2.0) as pinned
  submodules under `examples/`.
- BaseRT was downgraded to a black-box benchmark: its public repository is
  Apache-2.0, but the actual inference engine ships as a proprietary binary.
- CUDA/RDMA-only systems, paper-only work, and repositories without a license
  remain watch/quarantine entries in `FRONTIER_RADAR.md`.

## Environment

- Apple M5, 24GB unified memory, macOS 27.0.
- Python 3.13.12 in an isolated `/tmp` environment.
- MLX 0.32.0 and vllm-metal commit `4c18ee0` built from source.
- Reference partition size: 512 tokens; reported M5 split threshold: 80
  threadgroups (`10 GPU cores * 8`).

No Aster dependency or virtual environment was changed. The reference test
needed a temporary `vllm.logger` stub and an exact copy of its pure page-size
formula to avoid installing unrelated Torch/vLLM modules; kernel, cache
encoding, reference math, assertions, and test inputs were unchanged.

## Correctness reproduction

`tests/test_split_kv_decode.py` passed `19/19` cases. Coverage included:

- FP16, BF16, and FP32;
- one and multiple sequences with aligned and unaligned KV lengths;
- the high-occupancy single-pass side of the gate;
- 512/700/1,300-token sliding windows and fully masked partitions;
- odd partition counts;
- q8/q4 TurboQuant and TurboQuant plus sliding-window masking.

## Same-binary A/B

The first two-binary attempt was rejected because compiler dead-code removal
made the gate-off controls noisy. The final experiment added one temporary
runtime-only force-single-pass switch to a single compiled binary. Three
process pairs used 30 warmups and 200 measurements per cell. Values below are
median-of-process-medians; positive delta means split was slower.

| KV tokens | Batch | Single pass | Adaptive split | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 0.307 ms | 0.318 ms | +3.52% |
| 2,048 | 2 | 0.401 ms | 0.392 ms | -2.18% |
| 2,048 | 4 | 0.598 ms | 0.637 ms | +6.61% |
| 2,048 | 5 | 0.685 ms | 0.678 ms | -1.03% (gate off) |
| 2,048 | 8 | 1.151 ms | 1.189 ms | +3.32% (gate off) |
| 8,192 | 1 | 0.589 ms | 0.631 ms | +7.27% |
| 8,192 | 2 | 1.031 ms | 1.350 ms | +30.93% |
| 8,192 | 4 | 2.091 ms | 3.547 ms | +69.63% |
| 8,192 | 5 | 2.481 ms | 2.435 ms | -1.83% (gate off) |
| 8,192 | 8 | 4.885 ms | 4.896 ms | +0.22% (gate off) |

## Aster same-shape cross-check

Aster's current block-indexed kernel and MLX native SDPA were interleaved with
the same `Hq=16/Hkv=8/D=128` shape, 30 warmups, 200 measurements, and three
processes. Aster uses a different block layout, so this isolates the kernel
boundary rather than the complete cache system.

| KV tokens | Batch | Aster | MLX native | Aster delta |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 0.324 ms | 0.302 ms | +7.22% |
| 2,048 | 4 | 0.543 ms | 0.501 ms | +8.27% |
| 2,048 | 8 | 0.831 ms | 0.808 ms | +2.77% |
| 8,192 | 1 | 0.551 ms | 0.555 ms | -0.67% |
| 8,192 | 4 | 1.693 ms | 2.007 ms | -15.66% |
| 8,192 | 8 | 3.312 ms | 4.048 ms | -18.17% |

Across the 2K/8K parity subset at batch 1/4/8, max absolute difference was
`0` to `6.10e-05`. At 8K Aster was also approximately `6%~32%` faster than
the reference single-pass kernel across batch 1/2/4/5/8.

## Decision

Do not import the vllm-metal split-KV gate or replace Aster's attention math.
The reference gate regresses on this M5, while Aster already has the stronger
long-context kernel boundary. Retain vllm-metal's better integration ideas:
fused K/V scatter, token/page graph provenance, and a lazy MLX C++ Primitive
with no per-layer evaluation or synchronization.

## Next experiment

Prototype that integration as a standalone, optional exact-MLX-version
extension. First prove a boundary win over `mx.fast.metal_kernel`; only then
wire an opt-in real-model bridge and run output, throughput, peak-memory, swap,
mixed/staggered, cancellation, and long-context stress gates.
