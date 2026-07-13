# Iteration 025: Roll Back Memory-Aware Prefill Microbatching

- Iteration ID: `ITER-20260714-025-prefill-microbatch-rollback`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `955ec1a`
- End commit: `955ec1a` (no code commit; experiment rolled back)
- Evidence: `iterations/artifacts/ITER-20260714-025-prefill-microbatch-rollback/current/`

## Problem And Hypothesis

Iteration 024 showed that naive batch prefill at 1024-token chunks was too
memory-heavy. The follow-up hypothesis was that cache-only evaluation plus a
small-chunk guard (`chunk <= 256`) could capture microbatch launch savings
without materializing the full vocabulary logits or exceeding unified-memory
pressure.

## Matrix Evidence

The standalone 0.8B fresh-cache probe used the existing MLX-LM cache merge and
extract methods and evaluated cache state only. The measured batch/elapsed
seconds were:

| Chunk | Batch 1 | Batch 2 | Batch 4 |
| ---: | ---: | ---: | ---: |
| 128 | `0.076s` | `0.127s` | `0.176s` |
| 256 | `0.120s` | `0.176s` | `0.339s` |
| 512 | `0.165s` | `0.343s` | `0.687s` |
| 1024 | `0.333s` | `0.706s` | `1.331s` |

The corresponding allocator peaks stayed between `0.66` and `2.15 GB` in
the isolated probe. Small chunks looked promising relative to repeating the
batch-1 call, but isolated kernel timing was not sufficient to establish
serving correctness.

## End-To-End Evidence

Workload: 0.8B Qwen3.5, manual runtime, no prefix cache, four concurrent
requests, 8,373 prompt tokens each, 128 completion tokens, greedy sampling,
`prefill_token_budget=256`.

| Path | Elapsed median | Completion tok/s median | Peak MLX memory | Swap |
| --- | ---: | ---: | ---: | ---: |
| Serial prefill | `22.818s` | `22.439` | `1.662 GB` | `0` |
| Batch=4 prefill | `20.182s` | `25.369` | `1.931 GB` | `0` |

This is an apparent `11.5%` latency improvement and `13.1%` throughput
improvement for the explicitly small-chunk configuration. A one-shot 9B
512-word probe also improved `25.355s -> 24.358s`, with throughput
`20.193 -> 21.020 tok/s` and peak memory `5.997 -> 6.622 GB`.

## Correctness Failure And Rollback

Serial-vs-batched 0.8B greedy parity was false: three of four response hashes
matched, but one prompt produced a different text SHA. Completion token counts
and finish reasons remained equal, so the failure is a deterministic content
divergence rather than a lifecycle crash. The hybrid `ArraysCache + KVCache`
state cannot be treated as equivalent merely because first-token argmaxes
match.

An earlier variant explicitly evaluated the full logits tensor and reached
`12.886 GB` peak plus `0.93 GiB` swap at batch=4; removing that error lowered
memory but did not fix parity. All temporary source and tests were restored to
`955ec1a`.

## Verification And Next Priority

```text
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests
.venv/bin/pip check
uv lock --check
git diff --check
```

After rollback: `418 passed, 9 skipped, 1 warning` across 427 tests. Power data
remains unavailable because `powermetrics` requires superuser privileges.

Next, audit `mlx_lm.BatchGenerator` as the model-native alternative for
prefill/decode interleaving and cache ownership. Do not reimplement hybrid
batched prefill in the manual runner until deterministic token/text parity is
understood.
