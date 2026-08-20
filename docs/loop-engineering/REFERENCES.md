# Local Reference Repositories

These repositories are cloned locally for read-only architecture and
benchmark comparison. Their source is not copied into Aster without a
separate license and correctness review.

| Repository | Local path | Pinned commit | License | Relevant reference |
| --- | --- | --- | --- | --- |
| [z-lab/dflash](https://github.com/z-lab/dflash) | `examples/dflash-z-lab` | `07ebd93db9f472af339b644bb70221ad8428328a` | MIT | Block-diffusion speculative decoding, target verification, supported-model integration |
| [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | `examples/dflash-mlx-bstnxbt` | `60803233af4589e18588b9bacbb03880801c828a` | Apache-2.0 | Native MLX draft/verify loop, tape-replay rollback, verify-specific Metal kernels, L1/L2 prefix cache |
| [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) | `examples/dflash-mlx-aryagm` | `786a9c9ec454ae5a33ce815316f49ffa19aa162e` | MIT | MLX adapters, cache rollback, streaming generation, OpenAI-compatible serving boundary |

## Current refresh snapshot

All 20 reference working copies were fetched with tags and aligned to their
configured upstream branches on 2026-08-20. The 16 `.gitmodules` entries are
recorded as parent-repository gitlinks; the four local-only clones are pinned
in this document.

| Repository | Branch | Current commit | Parent tracking | Refresh result |
| --- | --- | --- | --- | --- |
| `Rapid-MLX` | `main` | `0ca6441287c22f05f465ef1126d7c78fb3ab1328` | gitlink | Updated |
| `lmstudio-mlx-engine` | `main` | `3b6d6145ab48332e9a21f6acc0589a369ba8ea89` | gitlink | Updated |
| `uzu` | `main` | `968a25e245ee05ad9fe769c1342f28e4c004cd87` | gitlink | Updated |
| `cider` | `main` | `4d91fcee9439f7aea17ae6e965271d9536c604a0` | gitlink | Already current |
| `llama.cpp` | `master` | `70aff25250075bf23b533c207b55168a4f926350` | gitlink | Updated |
| `vllm-metal` | `main` | `bd32be884647aecc425cc6199068dd0d785e6e12` | gitlink | Updated |
| `mistral.rs` | `master` | `1c92c44197c0af5890c9753d8bd23c20ac139168` | gitlink | Updated |
| `exo` | `main` | `b5375f8cee4368d09e1ce96a56b9f81fb0bc81aa` | gitlink | Already current |
| `vllm` | `main` | `6259572b283a4df3d0e8690aad5da003b012c103` | gitlink | Updated |
| `dflash-mlx-aryagm` | `main` | `786a9c9ec454ae5a33ce815316f49ffa19aa162e` | local-only | Already current |
| `dflash-mlx-bstnxbt` | `main` | `60803233af4589e18588b9bacbb03880801c828a` | local-only | Updated |
| `omlx` | `main` | `146d27241e9b01bab08e4768fddda749f6f085fa` | gitlink | Updated |
| `sglang` | `main` | `82c6fc2db9cbada6533022567a5b50ca548a0397` | gitlink | Updated |
| `dflash-z-lab` | `main` | `07ebd93db9f472af339b644bb70221ad8428328a` | local-only | Updated |
| `ollama` | `main` | `b7871fc0d1d82fe109536efa3e0e8e411c766c75` | gitlink | Updated |
| `mlc-llm` | `main` | `9fa644f54b04983adea4d0168f49fc6af4a893ba` | gitlink | Updated |
| `mlx` | `main` | `27fec909a3df9e572f5195607a453e273e7d80d0` | gitlink | Updated |
| `gemma4metal` | `main` | `0f09466b7fde772a4876bf7bee3ccdeb34313304` | local-only | Already current |
| `vllm-mlx` | `main` | `7afa61aff396254f3948b98a9a84fe26ebf77bb7` | gitlink | Updated |
| `mlx-lm` | `main` | `d06c5374a12e1f9384aad5fece583d7be9d2619d` | gitlink | Updated |

## Evaluation order

1. Read the Aster runtime and benchmark baseline first.
2. Compare DFlash cache ownership, rollback, draft/verify scheduling, and
   MLX/Metal execution boundaries against the corresponding Aster modules.
3. Reproduce any proposed gain on this Mac with deterministic output parity,
   memory/swap measurements, and a rollback path before integrating behavior.

The local-only clones were created with shallow history. Refresh all remote
references and commit pins explicitly before relying on newer upstream behavior.
