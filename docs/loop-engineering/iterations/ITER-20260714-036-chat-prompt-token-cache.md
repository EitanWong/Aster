# Iteration 036: Bounded Chat Prompt Token Cache

## Scope

- Starting code commit: `1bdc7b4`
- Source commit: `b82599b`
- Machine: macOS 27.0, Apple Silicon arm64, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`

## Problem and hypothesis

With prefix caching enabled, every repeated Agent request re-rendered the
entire chat template and recomputed all reuse points. A 40-turn prompt took
about 74 ms in preprocessing even when the same request had just been
processed. The candidate added a per-`ModelRunner` bounded LRU keyed by the
messages, `enable_thinking`, and chat-template kwargs. It stores only token IDs
and reuse-point metadata; KV state remains owned by the existing prefix cache.

Default capacity is 32 entries and `0` disables the cache. Entries are copied
on return, evicted LRU, and cleared by `clear_runtime_caches()`.

## TDD and verification

Tests cover cache hit, bounded eviction, disabled-prefix reuse skipping,
prefix-enabled reuse retention, and configuration validation. Final checks:

```text
.venv/bin/pytest -q
441 passed, 9 skipped, 1 warning

.venv/bin/ruff check aster/core/config.py aster/inference/model_runner.py \
  tests/test_config.py tests/test_model_runner.py
All checks passed!

.venv/bin/python -m compileall -q aster scripts tests
/opt/homebrew/bin/uv lock --check
.venv/bin/pip check
git diff --check
```

## Performance A/B

The encode-only 40-turn Qwen3.5-0.8B prompt (1,718 tokens, prefix enabled)
changed from `74.082 ms` median to `0.028 ms` on repeated requests.

Three fresh-process prefix-enabled end-to-end runs measured:

| Scenario | Baseline median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| cold | `2.5516s` | `2.5119s` | `-1.6%` |
| exact hot | `0.2548s` | `0.1788s` | `-29.8%` |
| append-only turn | `0.3032s` | `0.2846s` | `-6.1%` |

All requests completed with 16 completion tokens, `length` finish reasons,
valid prefix hits for hot/append cases, unchanged `1.495 GB`-class memory
behavior, and zero swap growth. The exact-hot and append response hashes
matched their baselines in the parity probes.

## Decision

**KEEP.** This is a measured Agent/chat preprocessing and cache-hit latency
improvement, not a claim of global decode throughput improvement. The cache is
bounded, opt-out, model-runner scoped, and does not alter KV ownership.

Rollback: `git revert b82599b` or set
`engine.chat_prompt_cache_max_entries: 0`.

## Next priority

Continue the model-native fixed-shape/state-isolation investigation for safe
multi-lane batching, and expand prefix-enabled Agent tests to longer contexts,
branching conversations, cancellation, and sustained runs.
