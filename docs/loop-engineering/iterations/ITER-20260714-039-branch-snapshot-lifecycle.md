# Iteration 039: Avoid Branch-Only Full Prompt Snapshots

## Scope

- Starting code commit: `95239aa`
- Source commit: `07bd566`
- Machine: macOS 27.0, Apple Silicon arm64, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`

## Problem and hypothesis

The sustained Agent probe created twelve unique recent branches from an
80-turn conversation. Each prefix-hit branch still created a full prompt KV
snapshot at decode activation, although the branch had already reused an
ancestor and was unlikely to be requested exactly again. The store grew from
`0.628 GB` / 13 entries after the cold base request to `1.344 GB` / 26
entries.

The candidate skips the full-prompt checkpoint only when the request has a
non-exact prefix hit. Cold requests still store a full prompt checkpoint, and
exact hits still retain/refresh that checkpoint. The decision is centralized
across both prefill-end `_maybe_checkpoint` and decode activation, with
`engine.snapshot_skip_full_prompt_on_prefix_hit: true` as the default and an
explicit `false` rollback switch.

## Reference design and implementation

The design preserves Aster's existing `PrefixStore` ownership and longest-
prefix lookup contract in `aster/inference/prefix_store.py`. It is informed by
the branch-aware radix-cache workload in
`examples/sglang/test/manual/core/test_session_radix_cache.py`, but copies no
reference code. Only the checkpoint admission decision changed; cache clone,
pin, lookup, cancellation, and decode state ownership are unchanged.

## TDD and verification

Tests cover branch prefix-hit skipping in decode activation and prefill-end
checkpointing, exact-hit retention, and the configuration default. Final
checks:

```text
.venv/bin/pytest -q
447 passed, 9 skipped, 1 warning

.venv/bin/ruff check aster/core/config.py aster/inference/engine.py \
  tests/test_config.py tests/test_engine_runtime.py
All checks passed!

.venv/bin/python -m compileall -q aster tests
/opt/homebrew/bin/uv lock --check
.venv/bin/pip check
git diff --check
```

## Performance and resource A/B

Three fresh-process runs per arm used Qwen3.5-0.8B, prefix caching, an
80-turn Agent base conversation, append/recent-branch follow-ups, twelve
unique branches, and recovery. The old arm set
`snapshot_skip_full_prompt_on_prefix_hit: false`; the candidate used the
default `true`.

| Scenario | Old median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| cold | `2.2711s` | `2.3041s` | `+1.5%` |
| exact hot | `0.1483s` | `0.1516s` | `+2.2%` |
| append-only turn | `0.4491s` | `0.4682s` | `+4.3%` |
| branch 1 | `0.4336s` | `0.4575s` | `+5.5%` |
| branch 6 | `0.4161s` | `0.4181s` | `+0.5%` |
| branch 12 | `0.4134s` | `0.4269s` | `+3.3%` |
| recovery | `0.4221s` | `0.4294s` | `+1.7%` |

The candidate reduced post-recovery snapshot memory from `1.511 GB` / 29
entries to `0.739 GB` / 15 entries (`-51.1%`). At branch 12 it was
`1.455 GB` / 28 entries versus `0.684 GB` / 14 entries. All 3×3 arms had
identical hashes, prompt/completion counts, cache-hit flags, and saved-token
counts for every scenario; no swap growth was observed. The small latency
tradeoff is accepted as a bounded-memory lifecycle optimization and should
be rechecked with randomized ordering in future sustained runs.

The final cancellation probe accepted cancellation, returned the expected
`AsterError`, left `pinned_entries=0` and `pinned_bytes=0`, and the following
recovery request hit the prefix cache.

## Decision

**KEEP, with a measured tradeoff.** The default prevents unique branch
prompts from accumulating redundant full snapshots while preserving exact
reuse and branch correctness. This is not a claim of a universal latency
improvement; append/branch latency is slightly higher in the grouped A/B.

Rollback: `git revert 07bd566` or set
`engine.snapshot_skip_full_prompt_on_prefix_hit: false`.

## Next priority

Run randomized sustained branch/cancellation/recovery workloads and inspect
whether the remaining latency tradeoff is measurement noise or checkpoint
selection overhead. Then return to model-native fixed-shape state isolation
for safe multi-lane batching.
