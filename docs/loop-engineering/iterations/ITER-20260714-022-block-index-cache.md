# Iteration 022: Cache Direct Paged-Attention Block Indices

- Iteration ID: `ITER-20260714-022-block-index-cache`
- Date: 2026-07-14
- Machine: macOS `27.0`, Apple M5, 10 cores, 24 GB unified memory
- Python: `3.14.5`
- Start commit: `80e8bd1`
- Code commit: `be48448`
- Dependency commit: `86ed15c`
- Evidence: `iterations/artifacts/ITER-20260714-022-block-index-cache/current/`

## Problem And Hypothesis

The direct bridge recreated the logical block-index tensor on each attention
dispatch. This is a small fixed cost in a decode loop. Caching the tensor while
invalidating it on every block-table topology change should reduce allocator
and host-side dispatch overhead without changing kernel inputs or ownership.

## Change

- `PagedKVCacheLayer` now stores the last `uint32` block-index tensor and the
  corresponding immutable tuple of physical block IDs.
- The tensor is invalidated on table growth, copy-on-write remapping, trim,
  reset, and pool promotion changes.
- A regression assertion verifies that repeated `block_pool()` calls reuse the
  same index tensor while pool storage remains persistent.
- Dependency maintenance refreshed `uv.lock` to the latest versions allowed
  by the project constraints and declared `pydub` plus `python-multipart` in
  `pyproject.toml`, keeping locked sync behavior aligned with `requirements.txt`.

## Correctness And Benchmark

The fresh randomized order was `direct, native, native, direct, direct,
native`, with 128 completion tokens per request and zero swap delta.

| Path | Elapsed median | Completion tok/s median | Peak MLX memory |
| --- | ---: | ---: | ---: |
| Native | `5.4306s` | `23.570` | `2.297 GB` |
| Direct paged | `5.4597s` | `23.445` | `2.286 GB` |

Relative to native, direct paged attention was `+0.54%` elapsed and `-0.53%`
completion throughput. The difference is below the 3% decision gate, so this
is retained as a low-risk allocation optimization and not reported as a
performance win. All six requests completed successfully.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python -m compileall -q aster tests
.venv/bin/pip check
uv lock --check
git diff --check
```

Result: `417 passed, 9 skipped, 1 warning` across 426 collected tests. The
focused paged-attention tests passed `23/23`; Ruff checks passed for the changed
logic with the repository's existing import-gating exceptions ignored.

## Decision And Next Priority

Keep the cached block-index tensor inside the existing opt-in direct bridge.
Do not enable direct paged attention by default: the bridge remains functional
and memory-neutral, but the cache reuse does not clear the speed gate. Next,
measure decode-only bridge overhead or evaluate broader model/batch support
with the same parity, lifecycle, memory, and randomized A/B requirements.
