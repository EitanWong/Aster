# Iteration 035: Skip Disabled Chat Reuse Analysis

## Scope

- Starting code commit: `a268e69`
- Source commit: `acf785f`
- Machine: macOS 27.0, Apple Silicon arm64, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`

## Problem and hypothesis

`ModelRunner.encode_request()` computed all Agent chat reuse points even when
`engine.prefix_cache_enabled` was false. The engine never consumes those
points in that mode, so the tokenizer repeatedly rendered every chat boundary
for no functional benefit.

The minimal change keeps `reuse_points=()` when prefix caching is disabled and
retains the existing analysis when it is enabled. No model, cache, sampling,
or API behavior changes.

## TDD and correctness

The disabled-path test was first run red because reuse analysis was still
called. It then passed after the guard was added. A second test verifies that
the enabled path still returns reuse points. Final verification:

```text
.venv/bin/pytest -q
439 passed, 9 skipped, 1 warning

.venv/bin/ruff check aster/inference/model_runner.py tests/test_model_runner.py
All checks passed!

.venv/bin/python -m compileall -q aster scripts tests
/opt/homebrew/bin/uv lock --check
.venv/bin/pip check
git diff --check
```

## Performance evidence

On the current Mac with the real Qwen3.5-0.8B tokenizer, a 40-turn Agent
conversation produced 1,718 prompt tokens. Seven encode-only measurements
changed median time from `73.136 ms` to `1.787 ms` (`-97.6%`), with reuse
analysis changing from 41 points to zero as intended.

Five end-to-end manual-runtime runs used the same 1,718-token conversation,
greedy sampling, 16 generated tokens, and prefix cache disabled. Baseline
median elapsed was `2.7199s`; current median was `1.8380s` (`-32.4%`). A
separate baseline/current parity probe returned the same text SHA-256,
`1718` prompt tokens, `16` completion tokens, and `length` finish reason.

## Decision

**KEEP.** This is a default-path, scenario-specific preprocessing improvement;
it does not claim a global decode speedup. Prefix-enabled behavior remains
covered and unchanged. Rollback is `git revert acf785f`.

## Next priority

Continue the fixed-shape/model-native state-isolation investigation for safe
multi-lane batching, while expanding the Agent workload matrix to include
prefix-cache enabled runs and longer contexts.
