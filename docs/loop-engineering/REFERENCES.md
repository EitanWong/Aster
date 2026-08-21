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

All 24 reference working copies were checked against their configured upstream
branches on 2026-08-21. Rapid-MLX, vLLM, and vllm-metal advanced, and mlxcel
was added as a new pinned research reference; all configured branch refs are
now locally fetched and pinned. The 20 `.gitmodules` entries
are recorded as parent-repository gitlinks; the four local-only clones are
pinned in this document. Each row records whether the branch advanced or was
already current, rather than presenting a stale local commit as a new upstream
observation.

| Repository | Branch | Current commit | Parent tracking | Refresh result |
| --- | --- | --- | --- | --- |
| `Rapid-MLX` | `main` | `51e5b47bfaa8b715291a1f211ad327c4b667571c` | gitlink | Advanced; fetched 2026-08-21 |
| `lmstudio-mlx-engine` | `main` | `3b6d6145ab48332e9a21f6acc0589a369ba8ea89` | gitlink | Fetch succeeded; no branch advance |
| `uzu` | `main` | `9022ad3b69fd10f5026e1b1f2538c82d9f5754a7` | gitlink | Fetch succeeded; no I097 branch advance |
| `cider` | `main` | `4d91fcee9439f7aea17ae6e965271d9536c604a0` | gitlink | Fetch succeeded; no branch advance |
| `llama.cpp` | `master` | `749f688fcaa4c472ec034b08cb8a907c45cfaa02` | gitlink | Fetch succeeded; no I097 branch advance |
| `vllm-metal` | `main` | `aa6d9611c0441270935921ce4f680fc11232129d` | gitlink | Advanced; fetched 2026-08-21 |
| `mistral.rs` | `master` | `d184053f2441f897cf81429b98b0d868f4d96ff3` | gitlink | Fetch succeeded; no branch advance |
| `exo` | `main` | `b5375f8cee4368d09e1ce96a56b9f81fb0bc81aa` | gitlink | Fetch succeeded; no branch advance |
| `vllm` | `main` | `ba07e4a48fc951300d97eb506217dd530583dea3` | gitlink | Advanced; fetched 2026-08-21 |
| `dflash-mlx-aryagm` | `main` | `786a9c9ec454ae5a33ce815316f49ffa19aa162e` | local-only | Already current |
| `dflash-mlx-bstnxbt` | `main` | `60803233af4589e18588b9bacbb03880801c828a` | local-only | Already current |
| `omlx` | `main` | `fa3e94b3b93afebe7fa5f39ea195404976244f69` | gitlink | Fetch succeeded; no branch advance |
| `sglang` | `main` | `6127d1daeee33ec758a737bca0a1ddd2d17f02ca` | gitlink | Fetch succeeded; no I097 branch advance |
| `dflash-z-lab` | `main` | `07ebd93db9f472af339b644bb70221ad8428328a` | local-only | Already current |
| `ollama` | `main` | `8f912415e867d86a511ac51afbd6b79e0d1bbc35` | gitlink | Fetch succeeded; no branch advance |
| `mlc-llm` | `main` | `9fa644f54b04983adea4d0168f49fc6af4a893ba` | gitlink | Fetch succeeded; no branch advance |
| `mlx` | `main` | `27fec909a3df9e572f5195607a453e273e7d80d0` | gitlink | Fetch succeeded; no branch advance |
| `gemma4metal` | `main` | `0f09466b7fde772a4876bf7bee3ccdeb34313304` | local-only | Already current |
| `vllm-mlx` | `main` | `8c814e30f54ee2a8e06acf768713cf0f24e22850` | gitlink | Fetch succeeded; no branch advance |
| `mlx-lm` | `main` | `d53e70f7a3870b85a3262bdab10999ada307ffc8` | gitlink | Fetch succeeded; no I097 branch advance |
| `mlx-swift-lm` | `main` | `130e3f0cd68e120019b0864c390d0309b0ec5618` | gitlink | Fetch succeeded; no I097 branch advance |
| `mlxcel` | `main` | `fdc196663579c73228fe2bfdbaf9a31f8114bbfb` | gitlink | Added from the author repository; Apache-2.0; fetched 2026-08-21 |
| `SpecForge` | `main` | `de0ea2f5de78e306e916bd58162932a2ea1dcc77` | gitlink | Fetch succeeded; no I097 branch advance |
| `S2-MoE` | `orin` | `fba914c3455403f0de2b50be3faeba0a4195e0f6` | gitlink | Fetch succeeded; no I097 branch advance |

## Evaluation order

1. Read the Aster runtime and benchmark baseline first.
2. Compare DFlash cache ownership, rollback, draft/verify scheduling, and
   MLX/Metal execution boundaries against the corresponding Aster modules.
3. Reproduce any proposed gain on this Mac with deterministic output parity,
   memory/swap measurements, and a rollback path before integrating behavior.

The local-only clones were created with shallow history. Refresh all remote
references and commit pins explicitly before relying on newer upstream behavior.
