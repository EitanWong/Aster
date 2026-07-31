# Iteration 085: Exact-Hit Shared-State Feasibility

- Date: 2026-07-31
- Phase: planned
- Baseline: completed I084 fanout screen at working-tree head `5e32228f2014`;
  the I081-I084 candidate remains uncommitted
- Scope: cache-type mutation proof and opt-in experiment; no default-path,
  budget, persistence-format, or eviction-policy change is pre-authorized

## Objective

Determine whether exact-prefix hits can fork a retained prompt cache by sharing
immutable history and detaching only mutable request-owned state. Replace no
behavior unless the candidate proves isolation for every cache layer used by
Qwen3.5 and materially reduces the B8 ownership signal measured in I084.

## Hypothesis

`ModelRunner.clone_cache` currently applies `copy.deepcopy` to the complete
cache. I084 observed one 390,397,952-byte active estimate per exact replay,
linear live-state growth, a 2.330 GB B8 MLX-peak increase over B4, and material
tail-latency growth. Some MLX cache layers may safely share evaluated history
through separate Python wrappers because updates assign new arrays, but hybrid,
rotating, or state-space layers may mutate nested state and require an eager
copy or a type-specific detach operation.

## Predeclared Work

1. Inventory the concrete Qwen3.5 prompt-cache layer types and trace each
   update/trim/merge/extract path. Classify wrapper fields as immutable-shared,
   copy-on-write, or eagerly isolated; reject unknown types by default.
2. Add tests that fail before implementation for base-snapshot immutability,
   sibling-fork independence, exact decode, strict-prefix append, trim,
   merge/extract, cancellation, and release. Include array identity and value
   digests so a shallow wrapper copy cannot pass on output alone.
3. Implement the smallest opt-in cache-fork primitive supported by those
   proofs. Keep `clone_cache` and the engine default unchanged until the
   experimental path clears unit and real-model gates.
4. Benchmark clone/fork construction and first-write materialization on the
   local small model, then run fresh 9B B2/B4/B8 exact fanout only if the
   mutation gates pass. Use I084 plans, sampler, rotated order, and output
   contract unchanged.

## Selection Gate

- Exact token/text/finish identity, unchanged retained-snapshot digest, and
  independent sibling state are hard gates across exact, append, decode,
  cancellation, persistence, and batch membership changes.
- Unknown or in-place-mutating cache layers fall back to eager clone; no
  generic `copy.copy` or untyped aliasing is admitted.
- A production A/B advances only if three fresh B8 repetitions reduce the
  clone-correlated MLX/live-state signal by at least 25% and replay p95 latency
  does not regress by 3%, with zero new swap pressure and clean terminal state.
- Keep the configured 8 GiB budget, reservation trace, store eviction policy,
  persistence schema, and `snapshot_skip_full_prompt_on_prefix_hit` rollback
  semantics unchanged.
