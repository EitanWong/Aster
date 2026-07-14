# Local Reference Repositories

These repositories are cloned locally for read-only architecture and
benchmark comparison. Their source is not copied into Aster without a
separate license and correctness review.

| Repository | Local path | Pinned commit | License | Relevant reference |
| --- | --- | --- | --- | --- |
| [z-lab/dflash](https://github.com/z-lab/dflash) | `examples/dflash-z-lab` | `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756` | MIT | Block-diffusion speculative decoding, target verification, supported-model integration |
| [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | `examples/dflash-mlx-bstnxbt` | `9ca002898b48e14c9727dec17299f497e8467870` | Apache-2.0 | Native MLX draft/verify loop, tape-replay rollback, verify-specific Metal kernels, L1/L2 prefix cache |
| [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) | `examples/dflash-mlx-aryagm` | `786a9c9ec454ae5a33ce815316f49ffa19aa162e` | MIT | MLX adapters, cache rollback, streaming generation, OpenAI-compatible serving boundary |

## Evaluation order

1. Read the Aster runtime and benchmark baseline first.
2. Compare DFlash cache ownership, rollback, draft/verify scheduling, and
   MLX/Metal execution boundaries against the corresponding Aster modules.
3. Reproduce any proposed gain on this Mac with deterministic output parity,
   memory/swap measurements, and a rollback path before integrating behavior.

The clones were created with shallow history on 2026-07-14. Refresh the
commit pins explicitly before relying on newer upstream behavior.
