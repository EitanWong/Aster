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

All 22 reference working copies were checked against their configured upstream
branches on 2026-08-21. Eleven gitlinks advanced to the fetched branch tips in
this iteration; all configured branch refs are now locally fetched and pinned.
The 18 `.gitmodules` entries are recorded as
parent-repository gitlinks; the four local-only clones are pinned in this
document. Each row records whether the branch advanced or was already current,
rather than presenting a stale local commit as a new upstream observation.

| Repository | Branch | Current commit | Parent tracking | Refresh result |
| --- | --- | --- | --- | --- |
| `Rapid-MLX` | `main` | `58ad76927b8a0e3486ac7282803b90e9bb04446f` | gitlink | Advanced; fetched 2026-08-21 |
| `lmstudio-mlx-engine` | `main` | `3b6d6145ab48332e9a21f6acc0589a369ba8ea89` | gitlink | Fetch succeeded; no branch advance |
| `uzu` | `main` | `1fd0c4610da650a63d8d6bf520bf41593cf2dc82` | gitlink | Advanced; fetched 2026-08-21 |
| `cider` | `main` | `4d91fcee9439f7aea17ae6e965271d9536c604a0` | gitlink | Fetch succeeded; no branch advance |
| `llama.cpp` | `master` | `0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd` | gitlink | Advanced; fetched 2026-08-21 |
| `vllm-metal` | `main` | `67100ba77780dec48adeb569724efaf8fe928b19` | gitlink | Advanced; fetched 2026-08-21 |
| `mistral.rs` | `master` | `d184053f2441f897cf81429b98b0d868f4d96ff3` | gitlink | Advanced; fetched 2026-08-21 |
| `exo` | `main` | `b5375f8cee4368d09e1ce96a56b9f81fb0bc81aa` | gitlink | Fetch succeeded; no branch advance |
| `vllm` | `main` | `bfb6c134997aace3e801c9ae3251728bd5312003` | gitlink | Advanced; fetched 2026-08-21 |
| `dflash-mlx-aryagm` | `main` | `786a9c9ec454ae5a33ce815316f49ffa19aa162e` | local-only | Already current |
| `dflash-mlx-bstnxbt` | `main` | `60803233af4589e18588b9bacbb03880801c828a` | local-only | Updated |
| `omlx` | `main` | `fa3e94b3b93afebe7fa5f39ea195404976244f69` | gitlink | Advanced; fetched 2026-08-21 |
| `sglang` | `main` | `0f744b684836edadb0b6ab18d6dd4beda457ccb2` | gitlink | Advanced; fetched 2026-08-21 |
| `dflash-z-lab` | `main` | `07ebd93db9f472af339b644bb70221ad8428328a` | local-only | Updated |
| `ollama` | `main` | `8f912415e867d86a511ac51afbd6b79e0d1bbc35` | gitlink | Advanced; fetched 2026-08-21 |
| `mlc-llm` | `main` | `9fa644f54b04983adea4d0168f49fc6af4a893ba` | gitlink | Fetch succeeded; no branch advance |
| `mlx` | `main` | `27fec909a3df9e572f5195607a453e273e7d80d0` | gitlink | Fetch succeeded; no branch advance |
| `gemma4metal` | `main` | `0f09466b7fde772a4876bf7bee3ccdeb34313304` | local-only | Already current |
| `vllm-mlx` | `main` | `8c814e30f54ee2a8e06acf768713cf0f24e22850` | gitlink | Advanced; fetched 2026-08-21 |
| `mlx-lm` | `main` | `d06c5374a12e1f9384aad5fece583d7be9d2619d` | gitlink | Fetch succeeded; no branch advance |
| `mlx-swift-lm` | `main` | `3c5805a1ecf31cf41ee7c34d5a3858439706cb2c` | gitlink | Advanced; fetched 2026-08-21 |
| `SpecForge` | `main` | `2590f48e3a93f69a1e9e63caa23e9f2f9e07c84a` | gitlink | Fetch succeeded; no branch advance |

## Evaluation order

1. Read the Aster runtime and benchmark baseline first.
2. Compare DFlash cache ownership, rollback, draft/verify scheduling, and
   MLX/Metal execution boundaries against the corresponding Aster modules.
3. Reproduce any proposed gain on this Mac with deterministic output parity,
   memory/swap measurements, and a rollback path before integrating behavior.

The local-only clones were created with shallow history. Refresh all remote
references and commit pins explicitly before relying on newer upstream behavior.
