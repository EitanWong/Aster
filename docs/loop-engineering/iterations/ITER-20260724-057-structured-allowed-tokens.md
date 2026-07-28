# Iteration 057: Reuse Structured Allowed-Token Lists

- **Date:** 2026-07-24
- **Reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`; Iterations
  055-056 and unrelated shared-worktree changes remain uncommitted
- **Hardware:** Apple M5, 10 CPU cores, 24 GB unified memory
- **Software:** macOS 27.0 (`26A5388g`), Python 3.14.5, MLX 0.32.0,
  MLX-LM 0.31.3, lm-format-enforcer 0.11.3, Apple clang 21.0.0
- **Model:** Qwen3.5-0.8B 4-bit

## Problem and Gates

Iteration 056 left Aster-owned structured and thinking processors as the main
unprofiled eager-row cost. The first hypothesis was that sending Python token
history directly to those processors would avoid an MLX array round trip.
Profiling instead found a larger Aster-local cost inside
`JSONSchemaLogitsProcessor._allowed_tokens`: lm-format-enforcer (LMFE) had
already cached a native `list[int]`, but Aster copied and reconverted about
246,884 token IDs on every structured decode row, then scanned that full list
to find one of a few EOS IDs.

Admission required:

- a `+3%` lower bound for balanced, baseline-first, and production-first
  assignment strata;
- 18 fresh processes and nine odd/even physical-runner-balanced replicates per
  short and long cell;
- exact token, text, and cache parity, no swap growth, and all but at most one
  stable replicate;
- schema-valid stop-aware output with dynamic active-membership shrink;
- the current production source and model inputs bound into every payload.

## Code and Dependency Analysis

CodeGraph traced `ModelRunner._apply_logits_processors` through
`JSONSchemaLogitsProcessor.__call__`, `_allowed_tokens`, and `_mask`. The
installed LMFE `TokenEnforcer.get_allowed_tokens` stores parser results in
`OutputTensorState` and returns `TokenList.allowed_tokens` as a cached Python
integer list when bitmasks are disabled. Aster did not mutate this list, but
still rebuilt it with:

```python
[int(token_id) for token_id in allowed]
```

The EOS test then iterated the large allowed set instead of the small tokenizer
EOS set. This is an adapter ownership issue, so no reference-project code was
copied. The production change follows the existing LMFE return contract and
retains conversion for non-native containers.

## Profiling and Candidate Reduction

The instrumented B4 screen attributed a mean `15.32 ms` per row to the JSON
processor, including `8.10 ms` in `_allowed_tokens` and `6.04 ms` in `_mask`.
Allowed-set cardinality had a median, p95, and maximum of 246,884 IDs. Across
320 mask calls, only 19 token-tuple contents were distinct, but a benchmark-only
identity cache hit 0/320 times. Constructing 246k-entry tuple keys also changed
the measured path, so that cache was rejected.

The direct Python-token delivery candidate remained below the 3% early gate in
its isolated screen and was not promoted. A combined benchmark then measured
`+26.84%` at short B4 and `+24.24%` at long B2, but its mask cache had zero
hits. An isolated `_allowed_tokens` benchmark reproduced `+26.89%` and
`+23.83%`, proving that one method owned the gain.

## Implementation

`JSONSchemaLogitsProcessor._allowed_tokens` now:

1. borrows LMFE's native empty or Python-integer list;
2. retains the prior integer-conversion fallback for other containers;
3. applies key-context and incomplete-JSON EOS filters into new lists, so the
   LMFE cache is never mutated;
4. checks EOS membership by iterating the small EOS set.

No ModelRunner token contract, parser order, mask construction, sampler, RNG,
cache extraction, or host materialization behavior changed.

Tests were written first. The native-list identity and complete-JSON identity
assertions failed under the old implementation, while non-native conversion
and EOS filtering already passed. All four behaviors pass after the change.

## Formal Benchmark Design

The strict harness keeps two independent KV states on one loaded model. Its
baseline explicitly replays the archived list copy and vocabulary-driven EOS
scan; production calls the current source. Every measured step alternates
A/B then B/A order. Odd/even processes swap policy-to-runner assignment and
form one balanced replicate. Each payload validates source/model hashes,
settings, timing order, physical assignment, and exact output before the
matrix advances.

Short command:

```bash
.venv/bin/python docs/loop-engineering/artifacts/ITER-20260724-057-structured-python-tokens/strict_matrix.py \
  --profile short --cell structured:4 --runs 18 --context-words 128 \
  --steps 256 --pair-warmup-steps 32 --block-size 16 --seed 20260724 \
  --output-dir docs/loop-engineering/artifacts/ITER-20260724-057-structured-python-tokens/results/strict-short-b4-r18
```

Long command:

```bash
.venv/bin/python docs/loop-engineering/artifacts/ITER-20260724-057-structured-python-tokens/strict_matrix.py \
  --profile long --cell structured:2 --runs 18 --context-words 8192 \
  --steps 128 --pair-warmup-steps 16 --block-size 8 --seed 20260724 \
  --output-dir docs/loop-engineering/artifacts/ITER-20260724-057-structured-python-tokens/results/strict-long-b2-r18
```

## Performance Results

All-step latency statistics include 4,608 short and 2,304 long paired policy
observations. Process throughput statistics contain 18 fresh processes per
cell.

| Cell | Prompt | Metric | Archived baseline | Production | Change |
| --- | ---: | --- | ---: | ---: | ---: |
| Structured B4 | 409 tokens | decode median | `79.962 ms` | `59.125 ms` | `-26.06%` |
| Structured B4 | 409 tokens | decode p95 | `111.803 ms` | `81.577 ms` | `-27.04%` |
| Structured B4 | 409 tokens | process throughput median | `46.649 tok/s` | `61.565 tok/s` | `+31.98%` |
| Structured B2 | 24,601 tokens | decode median | `51.023 ms` | `40.972 ms` | `-19.70%` |
| Structured B2 | 24,601 tokens | decode p95 | `58.216 ms` | `45.359 ms` | `-22.08%` |
| Structured B2 | 24,601 tokens | process throughput median | `37.534 tok/s` | `47.621 tok/s` | `+26.87%` |

| Cell | Balanced 96.09% interval | Baseline-first interval | Production-first interval | Stable |
| --- | ---: | ---: | ---: | ---: |
| Structured B4 | `[+31.33%, +33.40%]` | `[+25.19%, +27.09%]` | `[+37.29%, +39.90%]` | 9/9 |
| Structured B2 long | `[+24.96%, +27.63%]` | `[+28.30%, +32.86%]` | `[+20.96%, +23.22%]` | 9/9 |

Short per-process speedups ranged from `+26.38%` to `+37.88%`; long results
ranged from `+23.26%` to `+29.97%`. No record was removed. Median dual-state
MLX peaks were 758,705,476 bytes for short B4 and 2,054,056,378 bytes for long
B2. Every process had non-growing swap; one short process observed 8 MiB of
system swap reclamation.

TTFT, prefill throughput, prefix-cache hit rate, and power were not measured:
the changed method runs only during structured decode, and `powermetrics`
requires privileges unavailable to this loop.

## Correctness and Verification

- formal A/B: 36/36 fresh processes retained exact token, text, and cache
  output; both 11-gate aggregates passed;
- stop-aware real-model B4: 4/4 schema-valid JSON results, all stopped in
  17-58 tokens, and active membership shrank `4 -> 3 -> 1`;
- affected tests: `66 passed`;
- artifact assertions: `3 passed` across both matrices, hashes, validation,
  and composite admission;
- Ruff, `compileall`, and `git diff --check`: passed;
- full suite: `486 passed, 9 skipped, 1 failed, 1 warning`. The retained
  failure is the pre-existing missing
  `InferenceEngine._snapshot_budget_for_state` method in unrelated shared
  worktree changes.

The composite `final-admission.json` passes all 11 gates and binds the current
production SHA-256, both model signatures, both formal matrices, and retained
failed candidates.

## Decision and Rollback

**Retain the native allowed-token list fast path.** It removes a redundant
large Python allocation and reverses the membership scan without changing any
public or cross-processor contract. The gain clears the 3% floor in short,
long, balanced, and both order strata.

Rollback is local: restore unconditional integer-list conversion and the
allowed-list-driven EOS generator in `_allowed_tokens`. No configuration,
serialized state, cache key, or migration is involved.

## Next Priority

`_mask` remains about `6 ms` per structured row. The next iteration should
look for a cheap LMFE parser/token-list semantic key and measure a bounded
mask cache. It must not hash or tuple-convert the 246k-entry allowed set, must
define capacity and invalidation, and must preserve key-context/EOS filtering,
dynamic membership, and unknown processor behavior.

## Fixed Loop Output

LOOP ITERATION: 057
ROOT CAUSE: Aster recopied LMFE's cached 246k-entry Python token list and scanned it to find a tiny EOS set on every structured row.
CHANGES: Borrow native integer lists, retain conversion/filter fallbacks, and drive EOS membership from EOS IDs.
RESULT: Short B4 throughput median improved 31.98% and long B2 improved 26.87%; all formal and correctness gates passed.
NEXT: Find a bounded, cheap semantic key for structured mask reuse and measure it before changing production.
