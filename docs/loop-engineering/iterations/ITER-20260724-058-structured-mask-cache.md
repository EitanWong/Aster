# Iteration 058: Reuse Consecutive Structured Masks

- **Date:** 2026-07-24
- **Reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`; Iterations
  055-057 and unrelated shared-worktree changes remain uncommitted
- **Hardware:** Apple M5, 10 CPU cores, 24 GB unified memory
- **Software:** macOS 27.0 (`26A5388g`), Python 3.14.5, MLX 0.32.0,
  MLX-LM 0.31.3, lm-format-enforcer 0.11.3, Apple clang 21.0.0
- **Model:** Qwen3.5-0.8B 4-bit

## Problem and Gates

After Iteration 057 removed Aster's redundant allowed-list conversion,
`JSONSchemaLogitsProcessor._mask` still occupied about `6 ms` of the profiled
structured row. It rebuilt a full float32 vocabulary mask and copied it into
MLX even when consecutive free-text parser states allowed exactly the same
tokens.

Admission retained Iteration 057's strict requirements: a `+3%` lower bound in
balanced and both order strata, 18 fresh processes and nine physical-runner-
balanced replicates per short/long cell, exact token/text/cache parity, no swap
growth, stable blocks, stop-aware valid JSON, and source/model hash binding.
The dual-runner MLX peak delta was additionally capped at 16 MiB for short B4
and 8 MiB for long B2 against the same-shape Iteration 057 matrices.

## Root Cause and Design Reduction

CodeGraph traced `_mask` only to the JSON processor's eager-row call. The
installed LMFE source explained why the Iteration 057 identity-cache probe hit
0/320 times: `JsonSchemaParser` does not implement `cache_key()`, so
`TokenEnforcer` creates a new `TokenList` for each suffix even when its contents
repeat. A tuple/content hash would need to traverse about 246,884 IDs before
every lookup and was rejected.

The benchmark candidate instead used:

1. an O(1) key containing logits shape, allowed length, and first/middle/last
   IDs;
2. full list equality before every hit, making fingerprint collisions a safe
   miss;
3. a request-local LRU for immutable MLX masks.

A capacity-8 B4 screen hit 1,076/1,152 calls (`93.4%`) with no collisions and
measured `+83.16%`, but retained up to eight masks per lane. Capacity 1 retained
the same 1,076 hits, measured `+90.61%`, and reduced the dual-state MLX peak by
about 28 MiB versus capacity 8. Long B2 capacity 1 hit 270/288 (`93.8%`) and
measured `+64.16%`.

The result shows that useful reuse is consecutive, not general LRU reuse. The
production design therefore stores exactly one prior allowed snapshot and one
mask per processor.

## Correctness-First Implementation

`JSONSchemaLogitsProcessor` now owns three lifecycle-bound fields: the prior
fingerprint, an allowed-list snapshot, and the immutable MLX mask. `_mask`:

- computes the constant-time fingerprint;
- verifies full allowed-list equality and returns the prior mask only on an
  exact match;
- rebuilds through the unchanged clamp, NumPy mask, and MLX conversion path on
  every miss;
- shallow-copies the allowed list on a miss before replacing the single entry.

The shallow copy is required even though LMFE currently treats its lists as
immutable. A test demonstrated that retaining a mutable caller reference could
return a stale mask if a non-fingerprint element changed in place. The snapshot
makes that case a verified miss. Shape is part of the key, so one-dimensional
and `[1,V]` logits cannot share an incorrectly shaped mask.

The cache is per request because the processor is per request. It has no global
capacity, eviction, persistence, configuration, or cross-thread ownership.
Parser semantics, key/EOS filtering, arbitrary processors, sampler/RNG order,
and cache state are unchanged.

## Formal Benchmark Design

The formal baseline runs the current source but clears its one-entry cache
before every row. Production runs the current source directly. This isolates
only mask reuse while keeping mask construction, fingerprinting, list snapshot,
model graph, and all other code identical. The baseline forced exactly 1,152
short B4 and 288 long B2 misses in every process, including paired warmups.

Both matrices used two independent KV states, adjacent alternating A/B calls,
the same per-step seed, odd/even policy-to-runner swaps, source/model signatures,
and one fresh process per record.

## Performance Results

| Cell | Prompt | Metric | Forced-miss baseline | Production | Change |
| --- | ---: | --- | ---: | ---: | ---: |
| Structured B4 | 409 tokens | decode median | `60.927 ms` | `25.394 ms` | `-58.32%` |
| Structured B4 | 409 tokens | decode p95 | `86.617 ms` | `36.927 ms` | `-57.37%` |
| Structured B4 | 409 tokens | process throughput median | `59.297 tok/s` | `125.939 tok/s` | `+112.39%` |
| Structured B2 | 24,601 tokens | decode median | `41.801 ms` | `25.064 ms` | `-40.04%` |
| Structured B2 | 24,601 tokens | decode p95 | `50.067 ms` | `28.230 ms` | `-43.61%` |
| Structured B2 | 24,601 tokens | process throughput median | `45.566 tok/s` | `75.613 tok/s` | `+65.94%` |

| Cell | Balanced 96.09% interval | Baseline-first interval | Production-first interval | Stable |
| --- | ---: | ---: | ---: | ---: |
| Structured B4 | `[+114.79%, +119.26%]` | `[+100.97%, +107.33%]` | `[+126.49%, +135.45%]` | 9/9 |
| Structured B2 long | `[+64.81%, +68.91%]` | `[+71.21%, +76.17%]` | `[+56.10%, +61.79%]` | 9/9 |

Short process speedups ranged from `+100.16%` to `+138.82%`; long results
ranged from `+58.45%` to `+73.98%`. No record was removed.

Against Iteration 057's same-shape dual-runner matrices, median MLX peak grew
7,979,008 bytes at short B4 and 3,989,504 bytes at long B2. Both remain below
their predeclared bounds. The formal process contains both policies, and each
baseline runner retains its final forced-miss mask, so these are conservative
dual-state deltas rather than a claim about a single serving request. RSS was
noisy and did not establish a separate memory reduction. Swap did not grow.

TTFT, prefill throughput, prefix-cache hit rate, and power were not measured
because this is a structured decode-only change and privileged power telemetry
remains unavailable.

## Correctness and Verification

- formal A/B: 36/36 fresh processes retained exact token, text, and cache
  output; both 11-gate strict aggregates passed;
- stop-aware B4: 4/4 schema-valid JSON results, all stopped in 17-58 tokens,
  active membership `4 -> 3 -> 1`;
- unit coverage includes exact reuse, fingerprint collision, in-place mutation,
  and logits-shape invalidation;
- affected tests: `84 passed`;
- artifact assertions: `3 passed`; composite admission: 13/13 gates;
- Ruff and `git diff --check`: passed;
- full suite: `490 passed, 9 skipped, 1 failed, 1 warning`. The sole failure
  remains the unrelated shared-worktree absence of
  `InferenceEngine._snapshot_budget_for_state`.

## Decision and Rollback

**Retain the one-entry structured mask cache.** It has exact collision
verification, request-local lifetime, constant capacity, bounded memory, and a
large short/long gain under both order strata.

Rollback is local: remove the three processor fields plus the fingerprint,
equality, snapshot, and assignment block in `_mask`. The existing mask
construction then executes on every row. No configuration, serialization, or
migration is involved.

## Next Priority

Reprofile the post-Iteration-058 structured row. The remaining candidates are
LMFE's repeated `TokenList` construction/traversal, full-history MLX-to-Python
token conversion, and model/sampling time. Do not infer their new shares from
the pre-cache profile; measure them on the retained path before another change.

## Fixed Loop Output

LOOP ITERATION: 058
ROOT CAUSE: Consecutive JSON free-text states rebuilt identical full-vocabulary MLX masks because LMFE emits new list objects without a JSON parser cache key.
CHANGES: Added a request-local one-entry mask cache with O(1) fingerprinting, exact list verification, mutable-input snapshotting, and shape invalidation.
RESULT: Short B4 throughput median improved 112.39% and long B2 improved 65.94%; all performance, correctness, swap, and memory gates passed.
NEXT: Reprofile the retained structured row before targeting LMFE list construction or token-history conversion.
