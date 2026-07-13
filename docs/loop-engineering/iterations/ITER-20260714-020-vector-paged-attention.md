# Iteration 020: Token-Parallel Paged Attention Kernel

- Iteration ID: `ITER-20260714-020-vector-paged-attention`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `dfc17d1`
- Code end commit: `9d577a8`
- Evidence: `iterations/artifacts/ITER-20260714-020-vector-paged-attention/current/`

## Problem And Hypothesis

The first paged Metal kernel assigned one 32-lane simdgroup to each query
head, leaving the long KV scan serial within that head. MLX's vector SDPA
uses 32 simdgroups to partition the KV sequence and then combines partial
softmax states. Applying the same reduction structure to physical block rows
should remove the long-token serialization without materializing K/V.

## Change

- Added a token-parallel simdgroup kernel for aligned long sequences.
- Each query head uses 32 simdgroups, with online softmax partials reduced in
  threadgroup memory.
- Kept the existing tiled and proof kernels for short or non-aligned shapes.
- Added a causal multi-query regression covering tokens beyond the current
  query position, GQA, block indirection, and FP16 output.
- The model runner still uses native contiguous attention; this commit only
  improves the explicit `PagedAttentionView.attention()` boundary.

## Kernel Benchmark

The benchmark used Qwen3.5-shaped tensors: `queries=[1,8,1,256]`,
`pool=[ceil(N/64),1,2,64,256]`, FP16, reversed block indices, and seven
interleaved measurements per shape.

| KV tokens | Max abs diff | Native median | Paged median | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | `0.0` | `0.285ms` | `0.251ms` | `0.880x` |
| 2,048 | `0.0` | `0.326ms` | `0.318ms` | `0.976x` |
| 8,192 | `3.05e-05` | `0.590ms` | `0.432ms` | `0.733x` |

Paged median allocator memory was slightly lower in all three probes. Paged
p95 was higher because of occasional launch/JIT outliers, so the result is a
kernel median improvement rather than a tail-latency claim.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python scripts/dev/benchmark_paged_attention.py --tokens 512 2048 8192 --measurements 7
.venv/bin/python -m compileall -q aster tests scripts/dev
.venv/bin/pip check
git diff --check
```

Result: `413 passed, 9 skipped, 1 warning` across 422 collected tests. The
existing paged lifecycle probe still reclaimed all pool bytes and manager
blocks after release.

## Decision And Next Priority

Keep the vector kernel behind the explicit paged attention boundary. Do not
enable it in serving yet: the model path still calls MLX-LM's contiguous SDPA,
and direct integration needs a model-specific bridge, mask handling, decode
parity, and a production-shaped end-to-end A/B. Next, prototype that bridge
behind a separate opt-in flag and measure it against native serving.
