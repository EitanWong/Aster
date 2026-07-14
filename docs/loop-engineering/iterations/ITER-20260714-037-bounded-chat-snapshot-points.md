# Iteration 037: Bound Chat Snapshot Reuse Points

## Scope

- Starting code commit: `27d6e33`
- Source commit: `a8913e6`
- Machine: macOS 27.0, Apple Silicon arm64, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`

## Problem and hypothesis

For a 40-turn Agent conversation, automatic prefix snapshots were created at
nearly every chat boundary. The full KV snapshots grew to 39 entries and
about `1.192 GB` in the mixed exact/append/branch scenario, even though the
workload's useful follow-up branches were close to the end of the conversation.

The candidate adds `engine.snapshot_max_chat_reuse_points`, default `8`.
Positive values retain only the most recent chat reuse boundaries; `0`
preserves the previous unlimited behavior. This bounds retained KV snapshot
growth without changing tokenization, cache lookup semantics, or decode state.

## TDD and verification

The new tests cover most-recent-boundary selection, the default capacity, and
non-negative configuration validation. Final checks:

```text
.venv/bin/pytest -q
442 passed, 9 skipped, 1 warning

.venv/bin/ruff check aster/core/config.py aster/inference/model_runner.py \
  tests/test_config.py tests/test_model_runner.py
All checks passed!

.venv/bin/python -m compileall -q aster tests
/opt/homebrew/bin/uv lock --check
.venv/bin/pip check
git diff --check
```

## Performance and resource A/B

Three fresh-process runs per arm used Qwen3.5-0.8B, prefix caching, a
40-turn Agent conversation, and exact/append/branch follow-ups. The baseline
used `snapshot_max_chat_reuse_points: 0`; the candidate used `8`.

| Scenario | Baseline median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| cold | `2.5375s` | `1.8549s` | `-26.9%` |
| exact hot | `0.1833s` | `0.1815s` | `-1.0%` |
| append-only turn | `0.2849s` | `0.2865s` | `+0.6%` |
| branch | `0.3008s` | `0.2933s` | `-2.5%` |

All 12 requests completed with identical per-scenario response hashes,
identical prefix-hit flags, and identical tokens saved: `1470` exact,
`2912` append, and `4280` branch. Snapshot memory after the cold request fell
from `1.192 GB` / 39 entries to `0.326 GB` / 9 entries (`-72.7%`). After the
append and branch requests, candidate memory was `0.401 GB` and `0.475 GB`,
with zero swap growth in the probes.

## Decision

**KEEP.** This is a long-context Agent memory-bound and cold-latency
improvement. It preserves the recent exact/append/branch reuse paths measured
here, while keeping an explicit `0` escape hatch for workloads that prefer
unlimited historical snapshots.

Rollback: `git revert a8913e6` or set
`engine.snapshot_max_chat_reuse_points: 0`.

## Next priority

Continue the model-native fixed-shape/state-isolation investigation for safe
multi-lane batching, and expand the bounded-snapshot probe to longer,
branchier conversations, cancellation, and sustained runs before changing the
default again.
