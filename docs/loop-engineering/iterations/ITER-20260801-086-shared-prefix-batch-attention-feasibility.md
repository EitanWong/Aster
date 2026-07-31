# Iteration 086: Shared-Prefix Batch Attention Feasibility

- Date: 2026-08-01
- Phase: planned
- Baseline: completed I085 at working-tree head
  `cefe75359e6408aa5fd95f9662d1fc0108e4cfd2`
- Scope: benchmark-only attention ownership experiment; no default path,
  cache budget, persistence format, or eviction policy is pre-authorized

## Objective

Determine whether Qwen3.5 B>1 decode can consume the identical full-attention
prefix from shared storage without constructing a batch-contiguous copy for
each request, while retaining independent request-owned linear-attention state.

## Hypothesis

I085 attributes 86.80% of the 390,103,040-byte per-request merged state to eight
full-attention `BatchKVCache` layers and only 13.20% to 24 `ArraysCache` layers.
A split batch representation that keeps `ArraysCache.merge` unchanged but gives
the attention bridge a shared immutable prefix plus private suffix/block table
can remove at least 75% of B8 merge growth. Python wrapper changes alone cannot.

## Predeclared Work

1. Trace Aster's paged bundle, Qwen3.5 attention bridge, decode batch-state
   reuse, extraction, cancellation, and release contracts. Freeze the smallest
   benchmark-only interface that can carry shared prefix plus private suffix.
2. Add failing tests for B2/B4/B8 block-table aliasing, refcounts, first suffix
   write, unequal lengths, batch membership changes, extraction, cancellation,
   release, and native fallback. Keep all 24 auxiliary `ArraysCache` states
   request-owned.
3. Build a small-model attention-only screen. Reject the candidate if it
   materializes an equivalent B-by-prefix tensor before SDPA, changes logits,
   leaks a block reference, or invokes the native full-prefix merge.
4. Only after those gates pass, run the locked 9B QMSUM B2/B4/B8 fanout plans
   against the unchanged native merge control in fresh, rotated processes.

## Selection Gate

- Primary metric: B8 full-attention materialized bytes attributable to batch
  construction. The candidate must reduce total B8 merge growth by at least
  75% and full-attention growth by at least 90%.
- Exact token/text/finish identity, logits within the existing numerical
  contract, unchanged retained-snapshot digest, sibling independence, zero
  dangling references, and clean cancellation/release are hard gates.
- Replay p95 latency must not regress by 3%; peak MLX memory must not regress;
  no new swap pressure or decode fallback is allowed.
- Unknown models, non-Qwen3.5 cache layouts, unsupported attention shapes, and
  disabled experimental settings retain the native merge path.
- The production default, 8 GiB snapshot budget, reservation trace,
  persistence schema, eviction policy, and rollback switch remain unchanged
  throughout I086.
