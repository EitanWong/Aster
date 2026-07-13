# Iteration 018: Lazy Paged KV Pool Promotion

- Iteration ID: `ITER-20260714-018-lazy-paged-pool`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `8c88173`
- Code end commit: `6772425`
- Evidence: `iterations/artifacts/ITER-20260714-018-lazy-paged-pool/current/`

## Problem And Hypothesis

The opt-in paged path had reached timing parity after contiguous fallback
reuse, but it still created a persistent block pool and a contiguous cache for
the native MLX-LM attention path. The hypothesis was that serving should keep
the contiguous cache as the default storage for this boundary and only promote
it into the physical pool when a block-indexed consumer explicitly asks for
`PagedAttentionView.block_pool()`.

## Changes

- Added a storage-only mode for `PagedKVCacheLayer` and selected it for the
  opt-in model-runner path.
- Added lazy promotion from the materialized cache into the persistent pool.
- Added per-layer written-token accounting so hybrid full-attention layers do
  not share the manager's global block token count.
- Preserved COW behavior for storage-only append, fork, and explicit pool
  promotion; forked materialized buffers are independent.
- Added promotion and multi-layer storage tests, including fork isolation.

## Correctness And Tests

- Same greedy request text, prompt tokens, completion tokens, and finish reason
  matched between native and storage-only paged paths.
- Storage-only Qwen3.5-0.8B manual runtime completed 128-token generations at
  both 2,229 and 8,373 prompt tokens with zero failed requests and zero swap
  growth.
- Pool lifecycle still reclaimed `2,097,152` pool bytes and all manager blocks
  after the last bundle release.

```text
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests scripts/dev
.venv/bin/pip check
git diff --check
```

Result: `411 passed, 9 skipped, 1 warning` across 420 collected tests.

## Benchmark

The workload used Qwen3.5-0.8B 4-bit, greedy sampling, one request, 128 output
tokens, manual runtime, and process-level model load plus warmup. Single probes
were run at 2,229 and 8,373 prompt tokens. The randomized 8K order was
`paged, native, native, paged, paged, native`.

| Path | Prompt | Elapsed | Completion tok/s | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: |
| Native | 2,229 | `2.749s` | `46.55` | `1.677 GB` |
| Storage-only paged | 2,229 | `2.989s` | `42.82` | `1.654 GB` |
| Native | 8,373 | `5.452s` | `23.48` | `2.297 GB` |
| Storage-only paged | 8,373 | `5.477s` | `23.37` | `2.374 GB` |

In the randomized 3×3 8K run, native elapsed median was `5.4541s` and paged
was `5.4526s` (`-0.03%`); throughput medians were `23.468` and `23.475`
completion tok/s. This is below the 3% gate and is not a performance win.
Paged peak memory was `3.38%` higher in that run, but the prior persistent
pool-plus-contiguous duplication was removed from the normal storage path.

All six randomized requests completed 128 tokens with zero swap delta. Raw
records, parity output, lifecycle output, and the computed summary are in the
artifact directory above.

## Dependency Audit

`pip list --outdated` found unrelated transitive updates, but the Aster runtime
set is current within its compatibility bounds: `mlx 0.32.0`, `mlx-lm 0.31.3`,
`mlx-audio 0.4.5`, `fastapi 0.139.0`, and `pydantic 2.13.4`. `transformers`
remains `5.12.1` because `pyproject.toml` and `requirements.txt` require
`<5.13.0` for the current MLX-Audio compatibility boundary; PyPI `5.13.1` is
not an eligible upgrade. `pip check` passed, so no dependency change was
justified.

## Decision And Next Priority

Keep the storage-only lazy-promotion boundary opt-in. It preserves native
attention compatibility, parity, lifecycle reclamation, and removes the
normal duplicate pool allocation, but it does not clear the performance gate
or justify a default change. Next, evaluate direct attention over the pool or
another memory-reducing path with the same randomized end-to-end gate.
