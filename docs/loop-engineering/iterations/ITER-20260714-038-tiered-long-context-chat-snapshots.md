# Iteration 038: Tiered Long-Context Chat Snapshots

## Scope

- Starting code commit: `392b311`
- Source commit: `0e13e8f`
- Machine: macOS 27.0, Apple Silicon arm64, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`

## Problem and hypothesis

Iteration 037 bounded chat snapshots to the most recent eight boundaries,
which controlled memory but discarded useful mid-history and early-history
branches in an 80-turn Agent conversation. Unlimited snapshots preserved all
branches but grew to more than `3 GB` of retained KV state.

The candidate keeps the recent eight boundaries, then retains up to four
older boundaries with exponentially increasing distance and one earliest
boundary at or above `snapshot_min_prefix_tokens`. Sparse retention activates
only when the full prompt reaches the default `2048` tokens. Shorter chats
continue using the recent-boundary policy. Setting
`snapshot_chat_reuse_sparse_points: 0` disables the sparse tier; setting
`snapshot_max_chat_reuse_points: 0` preserves unlimited historical points.

## Reference design and implementation

The design follows the bounded-history principle used by radix/prefix cache
implementations in `examples/sglang/python/sglang/srt` and the snapshot-indexed
lookup boundary in Aster's `aster/inference/prefix_store.py`; no reference
code was copied. The selection remains in `ModelRunner._chat_reuse_points`,
so `InferenceEngine` and `PrefixStore` ownership and lookup semantics are
unchanged.

## TDD and verification

Tests cover recent-only selection, sparse older-boundary retention, the
long-context activation threshold, default capacities, and validation of all
new non-negative settings. Final checks:

```text
.venv/bin/pytest -q
444 passed, 9 skipped, 1 warning

.venv/bin/ruff check aster/core/config.py aster/inference/model_runner.py \
  tests/test_config.py tests/test_model_runner.py
All checks passed!

.venv/bin/python -m compileall -q aster tests
/opt/homebrew/bin/uv lock --check
.venv/bin/pip check
git diff --check
```

## Performance and resource A/B

Three fresh-process runs per arm used Qwen3.5-0.8B, prefix caching, an
80-turn Agent conversation, and exact/append/recent-branch/mid-branch/
old-branch follow-ups. Baseline used unlimited points (`0`); candidate used
recent `8`, sparse `4`, and sparse activation at `2048` prompt tokens.

| Scenario | Baseline median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| cold | `3.6749s` | `2.2592s` | `-38.5%` |
| exact hot | `0.1441s` | `0.1428s` | `-0.9%` |
| append-only turn | `0.4588s` | `0.4464s` | `-2.7%` |
| recent branch | `0.4335s` | `0.4323s` | `-0.3%` |
| mid-history branch | `0.2377s` | `0.4793s` | `+101.6%` |
| early-history branch | `0.1785s` | `0.3398s` | `+90.4%` |

The older branch latencies are higher because the candidate deliberately
reuses a shorter sparse prefix rather than retaining every historical
boundary. Both branches still hit the prefix cache, and all six scenario
hashes, prompt token counts, completion counts, and finish behavior matched
the baseline. Saved tokens were `2950/5872/8720/10162/10494` for baseline
and `2950/5872/8720/9348/9414` for candidate after the cold request.

Initial snapshot memory fell from `3.092 GB` / 81 entries to `0.628 GB` /
13 entries (`-79.7%`). After all follow-ups, memory was `3.436 GB` / 89
entries for baseline versus `1.411 GB` / 36 entries for candidate. All six
requests completed in every run with zero observed swap growth.

A separate three-run 40-turn regression probe confirmed that sparse retention
is disabled below the threshold: the candidate kept `0.326 GB` / 9 initial
entries, with exact/append/branch medians of `0.1755s`, `0.2759s`, and
`0.2730s`.

## Decision

**KEEP.** The tiered policy is a long-context memory and cold-latency
optimization, not a claim that every historical branch is as fast as an
unbounded snapshot store. It retains a bounded, recent-heavy history and
keeps old branches correct while reducing their reuse depth gracefully.

Rollback: `git revert 0e13e8f`; set
`engine.snapshot_chat_reuse_sparse_points: 0` for recent-only behavior; or
set `engine.snapshot_max_chat_reuse_points: 0` for unlimited historical
points.

## Next priority

Measure sustained 80+ turn conversations with repeated branch creation,
cancellation, and cache recovery; then return to model-native fixed-shape
state isolation for safe multi-lane batching. Do not increase sparse retention
until branch-depth data shows a stable benefit under the same memory gate.
