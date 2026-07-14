# Iteration 045 — Length-indexed prefix lookup

Date: 2026-07-15
Baseline commit: `2ad78dc`

## Objective

Remove the high-cardinality branch-miss scan in `PrefixStore.lookup()` without
changing snapshot ownership, cache contents, or serving behavior. Iteration
044 showed that cloning one 8,372-token exact hit cost only 9.545 ms, leaving
the flat prefix index as the narrower candidate.

## Reference comparison

- Rapid-MLX walks a token trie and combines it with LRU, pinning, and a deep
  copy on fetch.
- SGLang uses a radix owner with page alignment, reference counts, eviction,
  and optimized common-prefix matching.
- LM Studio selects a nearest snapshot, deep-copies it, and trims exact or
  near hits before reuse.

Those structures scale beyond Aster's default 256 snapshots, but importing
them would also import node/page ownership and memory overhead. Aster already
maintained a sorted index of distinct snapshot lengths, so the minimal
candidate probes those lengths longest-first and performs a direct dictionary
lookup for each possible prompt prefix.

## TDD boundary

A regression test stores one reusable short prefix plus 32 same-length
branches, then wraps the sorted token index in a list that rejects slicing.
The old implementation failed because it copied and scanned
`sorted_keys[:index]`; the candidate passes and returns the same longest
prefix.

## A/B microbenchmark

Each case retains branch snapshots in one namespace and measures exact,
ordinary prefix, and divergent worst-branch lookup. Values are medians in
milliseconds; the same process and workload were used before and after the
candidate.

| Entries × prompt tokens | Path | Baseline | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| 32 × 2,048 | divergent branch | 0.161 | 0.051 | -68.3% |
| 256 × 2,048 | divergent branch | 0.943 | 0.057 | -93.9% |
| 32 × 8,192 | divergent branch | 0.808 | 0.207 | -74.4% |
| 256 × 8,192 | divergent branch | 4.984 | 0.228 | -95.4% (`21.8x`) |
| 256 × 8,192 | exact | 0.029 | 0.032 | +0.003 ms |
| 256 × 8,192 | ordinary prefix | 0.098 | 0.108 | +0.010 ms |

The same 256 × 8,192 synthetic shape measured Rapid-MLX trie medians of
`0.418 ms` exact, `0.421 ms` ordinary prefix, and `0.247 ms` divergent branch.
Aster's candidate was `0.032/0.108/0.228 ms`. This cross-check supports the
bounded index decision, but it is not an overall engine comparison: Rapid-MLX
has different metadata and deep-copy semantics.

## Verification

- Prefix/cache suites: `12 passed`.
- Full suite excluding one unrelated in-progress runtime test: `458 passed,
  9 skipped, 1 deselected`.
- Full unfiltered suite: `458 passed, 9 skipped, 1 failed`; the failure is
  `test_long_context_snapshot_budget_is_capped_for_clone_headroom`, which
  targets uncommitted engine work outside this iteration.
- Ruff for both changed Python files, `compileall`, and `git diff --check`:
  passed.

## Decision

Keep the candidate. It removes entry-count scaling from the measured branch
miss and preserves the existing ownership model. Do not introduce a trie or
radix tree until sustained real traces demonstrate that the distinct-length
bound is itself a bottleneck.

## Next priority

Return to the production manual runtime: profile KV ownership and fixed-shape
state handling under real mixed/staggered model load. In parallel, maintain a
frontier-paper radar, but move only one paper-derived mechanism at a time into
a controlled reproduction branch.
