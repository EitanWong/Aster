# Iteration 060: Reuse LMFE Freetext TokenList Ownership

- **Date:** 2026-07-28
- **Reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`; unrelated shared-worktree
  changes remain uncommitted
- **Hardware:** Apple M5, 10 CPU cores, 24 GB unified memory
- **Software:** macOS 27.0 (`26A5388g`), Python 3.14.5, MLX 0.32.0,
  MLX-LM 0.31.3, lm-format-enforcer 0.11.3
- **Model:** Qwen3.5-0.8B 4-bit

## Problem and Hypothesis

Iteration 059 removed repeated EOS scans but its ownership profile still showed
large structured-output RSS growth. Native LMFE retained one list-backed
`TokenList` for every prefix state. In JSON freetext, most of each list was the
same 246,881-token static allowlist, copied again for each generated token.

The falsifiable hypothesis was that one request-local working copy per static
freetext allowlist, paired with retention of only the active sequential prefix
state, would reduce median RSS growth by at least 25% at structured B4 and
24,601-token B2 while preserving exact output and clearing a 3% balanced
throughput lower bound.

## Ownership Profile

The source-bound native profile measured strictly append-one sequences: 1,020
append transitions for 1,032 short B4 calls and 254 append transitions for 260
long B2 calls. Native prefix state retained 1,024 short or 256 long request
lists, with 2,115,467,520 and 533,310,912 bytes of list backing respectively.

The candidate subclasses LMFE's `TokenEnforcer` only for list-backed JSON
freetext states. It creates a working `TokenList` once per static cached
allowlist per request, trims only the dynamic tail before the next sequential
state, and never inserts that mutable working list into LMFE's
`allowed_token_cache`. Aster retains the active state after each append. A
non-monotonic suffix rebuilds from the root parser, preserving the existing
fallback behavior for non-sequential callers.

## Rejected Screens

- Releasing every predecessor state without list reuse sharply reduced memory
  but was about 3% slower on short B4 due to repeated large list allocation.
- A 32-state prefix window preserved exact output but regressed short B4 by
  22.9% because it released large lists in bursts.
- A Python composite static/dynamic token sequence was exact but regressed by
  59%; a vectorized mask variant still regressed by about 15%.

These alternatives are retained only in ignored scratch records and are not
part of the production path.

## Formal Results

Each formal cell used 18 fresh processes and nine runner-balanced AB/BA
replicates. Every record preserved exact token, text, and cache state; every
record had zero swap growth. The distribution-free 96.09%-coverage intervals
all clear the predeclared 3% floor.

| Cell | Balanced interval | Baseline-first | Production-first |
| --- | ---: | ---: | ---: |
| Structured B4, 409-token prompt | `[+16.01%, +18.00%]` | `[+6.64%, +7.30%]` | `[+25.98%, +29.98%]` |
| Structured B2, 24,601-token prompt | `[+12.52%, +13.79%]` | `[+11.67%, +13.55%]` | `[+12.37%, +13.92%]` |

The formal runner observed native maximum prefix-state counts of 272/144
(short/long, including paired warmup) and a candidate maximum of one per
processor. Stop-aware B4 produced 4/4 schema-valid JSON results, each stopped
before the limit, while active membership shrank from 4 to 3 to 1.

Two independent source-bound native/candidate ownership pairs per cell gave
these RSS-growth medians:

| Cell | Native median | Candidate median | Reduction |
| --- | ---: | ---: | ---: |
| Structured B4 | `1,901,150,208 B` | `25,059,328 B` | `98.68%` |
| Structured B2 long | `488,308,736 B` | `11,657,216 B` | `97.61%` |

The native/candidate profiles ended with `1,024/4` short and `256/2` long
prefix states. Candidate profiles retained one working freetext list per lane,
and every recorded request `TokenList` was absent after lane release.

## Verification and Evidence

- focused constrained-decoding, runner, and archive tests: `61 passed`;
- complete worktree suite: `510 passed, 9 skipped, 1 warning`;
- Ruff, `py_compile`, and `git diff --check`: passed;
- compact evidence: 49 logical files in one 218,232-byte archive, SHA-256
  `51b4475dbd3c332c85cd34ac9533fee7e30cce6f830901cee1495cf63095c603`;
- composite admission: 12/12 gates pass and both strict aggregates recompute
  exactly from the compact archive.

The fixed-length performance screens are not schema claims. The separate
stop-aware run supplies schema, finish-reason, dynamic-membership, and swap
evidence.

## Decision and Rollback

**Admit request-local freetext list reuse with active-state retention.** It
removes the measured repeated ownership while preserving LMFE parser
transitions, allowed-token membership, key/EOS filtering, masks, sampling,
cache extraction, and runner assignment.

Rollback is local: construct LMFE's native `TokenEnforcer` directly, remove
the active-prefix helpers and optional decode-step hook, and remove their
focused tests. No serialized format, configuration value, or migration is
involved.

## Remaining Limits

The implementation intentionally relies on LMFE 0.11.3 private traversal
behavior, so an LMFE upgrade requires a source-diff review and a fresh parity
matrix. The formal workload covers sequential JSON freetext; broader schemas,
tool calls, non-monotonic structured callers, concurrency/cancellation
pressure, energy, and thermal stability remain unmeasured.

## Next Priority

Iteration 061 establishes a fair local cross-engine baseline before any new
production candidate: same model, tokenizer/chat template, greedy sampling,
prompt/output shapes, cache policy, and warmup discipline for Aster and the
installed reference runtime. It must report equivalence limitations rather
than treating incomparable API paths as a speed ranking.

## Fixed Loop Output

LOOP ITERATION: 060
ROOT CAUSE: LMFE copied the same large JSON-freetext allowlist into every retained prefix state.
CHANGES: Reused one request-local working list per static allowlist and retained only the active sequential state.
RESULT: Formal short/long lower bounds were 16.01%/12.52%; RSS medians fell 98.68%/97.61%; exactness, schema, stop, release, and swap gates passed.
NEXT: Build an apples-to-apples local cross-engine baseline before claiming comparative engine performance.
