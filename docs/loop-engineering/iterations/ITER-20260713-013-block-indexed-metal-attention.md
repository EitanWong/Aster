# Iteration 013: Block-Indexed Metal Attention Contract

- Iteration ID: `ITER-20260713-013`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Start commit: `6c4e551`
- End commit: `31f47cf`
- Runtime: Python 3.14.5, MLX `0.32.0`, MLX-LM `0.31.3`
- Status: `SUCCESS` for correctness contract; `INVESTIGATING` for performance

## Problem and Hypothesis

The previous paged adapter exposed block IDs but native MLX attention still
required contiguous K/V. MLX's `mx.fast.metal_kernel` can accept a dense block
pool and a logical-to-physical index vector, so a first kernel can validate the
attention ABI before investing in a production tiled implementation.

## Contract

`aster.inference.metal_paged_attention.paged_block_attention()` accepts:

- queries `[B, Hq, Q, Dk]`;
- key/value pools `[P, B, Hkv, block, D]`;
- logical `block_indices` mapping to physical pool rows;
- `query_offset`, `total_kv_tokens`, and attention scale.

The kernel maps GQA heads, applies causal positions using `query_offset`, and
computes a numerically stable softmax without concatenating sequence K/V
arrays. `PagedAttentionView.block_pool()` and `.attention()` expose the same
contract from the Python adapter. The current view packs selected blocks with
`mx.stack`; a persistent physical pool is still required for a real
memory/throughput win.

## References

- MLX custom kernel API: `examples/mlx/docs/src/dev/custom_metal_kernels.rst`.
- Existing Apple Silicon kernel patterns:
  `examples/omlx/omlx/custom_kernels/qwen35_prefill/gdn.py`.
- Block-indexed attention layout:
  `examples/omlx/omlx/custom_kernels/common/csrc/kernels/steel_attention_block_token.h`.
- Installed MLX-LM Qwen3.5 attention and GQA mapping:
  `.venv/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py`.

## Correctness Evidence

- FP32 GQA block-indexed causal parity test passed.
- FP16 decode parity test passed.
- `PagedAttentionView.attention()` dispatch test passed.
- Full Qwen3.5-0.8B attention shape smoke used `Hq=8`, `Hkv=2`,
  `head_dim=256`, `block_size=64`, and 512 KV tokens. Maximum absolute
  difference from native MLX attention was `6.103515625e-05`.
- Full project suite: `398 passed, 9 skipped, 1 warning`.

## Performance Evidence

The Qwen-shaped smoke used one decode query and one warmup/compile pass:

| Path | Time | Peak memory |
| --- | ---: | ---: |
| Native MLX attention | `0.000191s` | `0.002109488 GB` |
| Proof block kernel | `0.016443s` | `0.002109500 GB` |

The proof kernel measured `85.96x` slower. It uses one thread per output
element and repeats the Q/K dot-product loop for each value dimension; this is
an intentional correctness baseline, not a production kernel. No performance
claim or default runtime integration is made.

## Decision and Risks

Keep `31f47cf` as an experimental kernel boundary. Do not route the manual or
batched runtime through it. The kernel proves that block tables, GQA mapping,
causal offsets, and MLX JIT invocation can be represented correctly, but the
current implementation has unacceptable compute redundancy and the Python
adapter still repacks blocks.

## Next Priority

Replace per-call `mx.stack` with a persistent per-layer GPU block pool, then
implement a tiled/simdgroup attention kernel that shares Q/K work across value
dimensions. Re-run FP16 token/logit parity and a 512/2K/8K decode/prefill A/B
before considering any opt-in runtime path.
