# Iteration 085: Exact-Hit Shared-State Feasibility

- Planned: 2026-07-31
- Completed: 2026-08-01
- Phase: complete
- Baseline: `cefe75359e6408aa5fd95f9662d1fc0108e4cfd2`
- Decision: reject a typed prompt-cache fork; retain the production
  `copy.deepcopy` path
- Scope: measurement tooling and tests only; no runtime, cache-budget,
  persistence, eviction, or default-setting change

## Objective

Determine whether exact-prefix hits can materially reduce B8 ownership by
replacing the complete prompt-cache `deepcopy` with a type-specific shared-state
fork while preserving retained-base and sibling isolation.

## Frozen Boundary

- Public workload: `cross-engine-core`, record
  `longbench:qmsum:fdd371de2668a6f1e7914fe9a67aef33927ecc392fdc2606`
- Source lock SHA-256: `d6d0877b452ed5627bf0fd39ebc1e59ccad6284cdb4eace27a954603a5211c16`
- Prompt: 41,672 characters / 10,334 tokens; only prompt and token-ID hashes
  are retained
- Model: local Qwen3.5-9B 4-bit, MLX-LM `0.31.3`
- Cache boundary: prefill 10,333 tokens in 2,048-token chunks
- Fanout: B2/B4/B8, three left-rotated repetitions in one loaded-model process

## Mutation Inventory

Qwen3.5-9B constructs 32 cache layers: 24 `ArraysCache` layers for linear
attention and 8 `KVCache` layers for full attention. Linear-attention forward
rebinds the two array slots and advances wrapper metadata. `KVCache` grows by
allocating when necessary, then writes the appended slice through MLX array
assignment. `ArraysCache.merge` and `BatchKVCache.merge` allocate batch-sized
state and copy every row.

MLX `copy.deepcopy` creates distinct Python/MLX array descriptors without
allocating another physical array. MLX slice assignment changes the target
descriptor to a copy-on-write result, leaving the retained base and siblings
unchanged. The proposed typed wrapper therefore duplicates behavior already
provided by the current production clone.

## TDD Contract

The initial focused run recorded three expected failures because the probe
module did not exist, while the independent MLX hybrid-cache isolation test
already passed. The completed seven-test contract covers:

- normalized and rejected fanout inputs;
- prompt-token admission before any prefill work;
- rejection of unknown cache-layer types;
- per-type byte attribution;
- retained base, two siblings, append, trim, merge/extract, and first-write
  value/offset isolation for `ArraysCache + KVCache`;
- physical-owner summary gates; and
- recomputation of the retained prompt-free artifact.

## 9B Results

Deep-copy construction produced zero active-memory growth in all nine rows.
Median construction time was 0.231 ms at B2, 0.367 ms at B4, and 0.741 ms at
B8. Batch merge, in contrast, produced median active-memory deltas of
780,402,816, 1,560,543,488, and 3,120,824,832 bytes. Those deltas match the
materialized merged state within 0.03%; every release returned exactly to the
5,492,206,090-byte loaded-model/prompt-cache baseline.

The logical merged state is 390,103,040 bytes per request:

- full-attention `BatchKVCache`: 338,591,744 bytes / 86.80%;
- linear-attention `ArraysCache`: 51,511,296 bytes / 13.20%.

At B8, batch merge materializes 2,708,733,952 full-attention bytes and
412,090,368 linear-attention bytes. The original cache has 454,164,480 allocated
bytes because native `KVCache` capacity grows in steps, while merge compacts to
the 390,103,040-byte logical length.

## Reference Boundary

- MLX-LM's native `BatchKVCache.merge` and `ArraysCache.merge` are the measured
  allocation owner.
- SGLang's local MLX backend has a shared slot pool and a pool-backed prefix
  adapter, but currently converts it to per-request contiguous attention caches
  and concatenates rows before batched SDPA. It is useful ownership scaffolding,
  not evidence that current MLX batched decode avoids materialization.
- vLLM assigns shared prefix block IDs to requests, increments block references,
  copy-on-writes partial writable blocks, and lets paged attention consume block
  tables directly. This is the architectural boundary that avoids full-prefix
  row copies.
- Aster already has an experimental reference-counted paged bundle, but its
  model gate currently forbids prefix caching and B>1 decode, and its batch path
  falls back to native contiguous merge.

## Decision

Reject the typed-fork candidate before production implementation. Its measured
physical saving is zero, so it cannot clear the predeclared 25% B8 reduction
gate; a production latency A/B would test an implementation with no plausible
memory effect. Keep eager `deepcopy`, the 8 GiB budget, reservation/eviction
policy, persistence schema, and exact-hit rollback behavior unchanged.

I084's active-cache estimate is retained as conservative logical ownership, not
as proof of clone-time physical allocation. I086 targets the actual owner:
direct shared-prefix consumption for full-attention layers during B>1 decode,
while preserving per-request linear-attention state.

## Verification

- Focused probe contract: `7 passed`
- Full suite: `568 passed, 9 skipped, 1 warning`
- Touched-file Ruff: passed
- Retained artifact:
  `docs/loop-engineering/artifacts/ITER-20260731-085-exact-hit-shared-state-feasibility/cache-ownership-probe.json`
