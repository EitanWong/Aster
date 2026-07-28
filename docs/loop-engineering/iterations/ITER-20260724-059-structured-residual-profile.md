# Iteration 059: Reuse Structured EOS Membership

- **Date:** 2026-07-24
- **Reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`; later
  iterations and unrelated shared-worktree changes remain uncommitted
- **Hardware:** Apple M5, 10 CPU cores, 24 GB unified memory
- **Software:** macOS 27.0 (`26A5388g`), Python 3.14.5, MLX 0.32.0,
  MLX-LM 0.31.3, lm-format-enforcer 0.11.3
- **Model:** Qwen3.5-0.8B 4-bit

## Problem and Hypothesis

The post-Iteration-058 profile showed that mask construction was no longer the
dominant structured-output cost. `JSONSchemaLogitsProcessor._allowed_tokens`
still occupied about 51-55% of short B4 decode and 28% of 24,601-token B2
decode. Most consecutive states produced the same allowed list, but Aster
rescanned that approximately 246,884-entry list to determine whether it
contained EOS before returning the already cached mask.

The falsifiable hypothesis was that reusing EOS presence only when the existing
mask snapshot matched exactly would improve short B4 and long B2 structured
decode by at least 3%, without changing token, text, cache, stop, schema, or
dynamic-membership behavior.

## Design

The processor stores one additional boolean beside its request-local one-entry
mask snapshot. Reuse requires the same length, first/middle/last probes, and
full list equality. A mismatch performs the original EOS membership scan. The
boolean is updated only when `_mask` accepts the exact allowed-list object from
the current `_allowed_tokens` call.

Incomplete JSON still performs `_is_complete_json` on every call when EOS is
present. EOS filtering creates a new list, and LMFE's native list is never
mutated. The cache remains capacity one, request-local, and lifecycle-bound.

Tests were written first for repeated membership, in-place mutation, and
completion rechecking. The first test failed with two membership scans before
the production change and passed with one afterward. Existing collision,
shape, mutation, and EOS-filter tests remained active.

## Benchmark Control

The formal baseline uses the current production source but clears only
`_mask_cache_contains_eos` before each row. Production uses the same source
without that forced miss. Model execution, LMFE advancement, allowed tokens,
key/EOS filtering, masks, sampler order, RNG, cache extraction, and runner
assignment are otherwise identical.

Each formal cell used 18 fresh processes and nine odd/even runner-balanced
replicates. Adjacent calls alternate AB/BA order, and the aggregate requires a
distribution-free 95% median interval to clear 3% for the balanced result and
both order strata. No observation was discarded.

## Results

Production screens retained exact token/text/cache parity and zero swap growth:

| Cell | Baseline | Production | Change |
| --- | ---: | ---: | ---: |
| Structured B4, 409-token prompt | `102.712 tok/s` | `163.794 tok/s` | `+59.47%` |
| Structured B2, 24,601-token prompt | `70.612 tok/s` | `82.511 tok/s` | `+16.85%` |

Formal 96.09%-coverage intervals:

| Cell | Balanced | Baseline first | Production first | Stable |
| --- | ---: | ---: | ---: | ---: |
| Structured B4 | `[+38.01%, +47.33%]` | `[+21.03%, +30.27%]` | `[+55.08%, +68.23%]` | 9/9 |
| Structured B2 long | `[+21.09%, +24.47%]` | `[+32.59%, +37.68%]` | `[+10.80%, +13.48%]` | 9/9 |

All 36 formal processes preserved exact token, text, and cache state. Every
baseline forced exactly 1,152 short or 288 long membership misses; production
forced zero. All formal records had zero swap growth.

Stop-aware B4 produced schema-valid JSON in all four lanes, stopped each lane
before the 256-token limit, and shrank active membership from 4 to 3 to 1.

## Memory Protocol Correction

`CURRENT.json` originally required predeclared RSS and MLX limits but omitted
their numeric values. The first formal matrix therefore counts only as
discovery evidence for memory. The omission is retained as a protocol
deviation rather than repaired retrospectively.

Before a fresh confirmation, the limits were recorded as 4/2 GiB maximum RSS
growth for short/long and 16/8 MiB median MLX peak growth versus the same-shape
Iteration 058 records. Two new independent processes per cell then measured:

- short RSS growth `3.72/2.82 GB`; median MLX delta `-1,065,177` bytes;
- long RSS growth `1.33/1.37 GB`; median MLX delta `-24,576` bytes;
- exact output and zero swap growth in all four confirmations.

These confirmations close the numeric memory gate. They do not claim that the
large LMFE RSS footprint is solved.

## Verification and Evidence

- affected and artifact tests: `142 passed`;
- full worktree suite: `503 passed, 9 skipped, 1 warning`;
- Ruff and `git diff --check`: passed;
- checker-specific Pyright: zero errors; the JSON processor retains existing
  third-party stub and Unknown-type reports;
- composite admission: 12/12 gates pass;
- compact evidence: 50 logical files in one 237,686-byte archive, SHA-256
  `783eebf302d0a95aa415b6372565f91a25737d0d6c222683bd09f95d7c9aa340`.

The fixed-length candidate and production screens did not always contain a
complete JSON document. They are retained as throughput screens, not schema
evidence; the separate stop-aware run supplies that correctness gate.

## Decision and Rollback

**Admit exact EOS-membership reuse.** It removes a repeated full-list scan,
uses the already verified snapshot identity, preserves completion checks, and
clears every short/long performance, exactness, stop, memory, and swap gate.

Rollback is local: remove `_mask_cache_contains_eos` and the two pending
membership fields, call the EOS membership scan directly in `_allowed_tokens`,
and remove the boolean handoff in `_mask`. No configuration, migration, or
serialized state is involved.

## Next Priority

Profile ownership and lifetime of LMFE's retained 246,884-entry `TokenList`
objects. Iteration 059 observed multi-gigabyte RSS growth but did not establish
which prefix states must remain live. The next iteration must measure active
versus accumulated state before testing a bounded lifetime change.

## Fixed Loop Output

LOOP ITERATION: 059
ROOT CAUSE: Exact consecutive allowed lists reused their MLX mask but rescanned the full list for EOS presence on every row.
CHANGES: Stored EOS presence beside the exact request-local mask snapshot and preserved per-call completion checks.
RESULT: Formal short/long interval lower bounds were 38.01%/21.09%; all 36 outputs, stop semantics, memory confirmations, and swap gates passed.
NEXT: Attribute LMFE TokenList ownership and reduce structured RSS with an explicit bounded-lifetime design.
