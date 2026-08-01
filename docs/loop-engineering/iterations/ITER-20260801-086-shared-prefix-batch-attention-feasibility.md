# Iteration 086: Shared-Prefix Batch Attention Feasibility

- Planned: 2026-08-01
- Completed: 2026-08-01
- Phase: complete
- Baseline: `4daeae3e8b9fe0bf352e55aa266543c285db6052`
- Decision: reject the current SIMD-group shared-pool kernel before a 9B
  model-runner A/B
- Scope: benchmark-only attention ownership experiment; no model-runner,
  attention-bridge, configuration, cache budget, persistence, eviction, or
  production-default change

## Objective

Determine whether Qwen3.5 B>1 decode can consume an identical full-attention
prefix from shared physical storage without constructing a batch-contiguous
copy for every request, while retaining independent request-owned
linear-attention state.

I085 measured 390,103,040 logical merged bytes per request: 338,591,744 bytes
(86.80%) in eight full-attention layers and 51,511,296 bytes (13.20%) in 24
`ArraysCache` layers. I086 therefore tests the actual ownership boundary rather
than another Python cache-clone wrapper.

## Traced Boundary

- `PagedKVCacheLayer.merge()` converts every paged layer to native `KVCache`
  and calls `BatchKVCache.merge`, which allocates `[B,Hkv,K,D]` keys and values.
- Exact-prefix `PagedKVCacheBundle.fork()` already shares block tables and one
  refcounted per-layer pool. A first write to a shared partial block performs
  CoW; full prefix blocks remain shared.
- The existing Metal entry point accepts only one-dimensional block indices and
  requires the pool batch dimension to equal the query batch. It therefore
  cannot consume B request tables over one singleton pool.
- Qwen3.5 calls `cache.update_and_fetch()` before the patched attention
  function. A production batch representation would consequently need model
  cache update/extract/filter/release integration in addition to an attention
  kernel.
- The production paged-cache gate still requires prefix caching off and
  `max_decode_batch=1`. Native batch merge, all 24 `ArraysCache` layers, stable
  batch membership, extraction, cancellation, and fallback paths are unchanged.

## Current Reference Refresh

The configured search endpoint returned HTTP 404, so the refresh used read-only
official GitHub APIs/Git refs and the arXiv API.

- MLX main was `2ad0d4d311f54de855b06cd21ca85d3b628c1012` (2026-07-31).
  Its public SDPA remains batch-contiguous; the latest relevant Metal SDPA
  commit was `8462ad9fd210518341f36f42e997cfafb7c528db`.
- vllm-metal main was
  `b6e35b6c642162dbf6f31009b81635426a91b64a`. Its Qwen3.5 path uses one
  global page pool, two-dimensional per-request block tables, sequence lengths,
  packed query offsets, and a lazy MLX Primitive. Commit
  `32cc5fd7d08f9bb4c896254d1384037cc3518435` moved invariant paged metadata
  construction from every layer to once per forward.
- The local vllm-metal mirror at `d4afdd6a062f647e1e4f4be2cab99e6377dc94a8`
  contains the same ownership and Qwen3.5 integration boundary used here.
- [PackInfer](https://arxiv.org/abs/2602.06072) packs heterogeneous query/KV
  regions and co-locates shared-prefix requests, but assumes a new optimized
  attention kernel and group-contiguous cache layout.
- [Requests of a Feather Must Flock Together](https://arxiv.org/abs/2605.06046)
  reports that prefix-homogeneous scheduling can outperform prefix-aware
  kernels, and that the useful batch-size/prefix-homogeneity tradeoff is
  workload-dependent.
- [RadixMLP](https://arxiv.org/abs/2601.15013) deduplicates position-wise
  prefill work inside a forward pass; it does not remove decode-time KV reads
  and is a separate candidate class.

## TDD Contract

The first focused run failed at collection because `PagedBatchAttentionView`
did not exist. The completed contract contains ten new tests:

- B2/B4/B8 shared-prefix attention matches row-wise native MLX SDPA;
- monkeypatched `materialize()` and native `merge()` fail if the candidate
  invokes either path;
- unequal sequence lengths use independent table rows and sequence lengths;
- first suffix writes preserve common full blocks and CoW the shared partial
  block;
- independent physical pools are rejected;
- batch metadata borrows rather than retains pool ownership, and final release
  leaves zero allocated blocks and zero pool bytes; and
- the benchmark parser and five predeclared selection gates recompute from
  synthetic pass/fail fixtures; and
- the retained summary records all six scratch-result hashes, recomputes the
  selection decision, and verifies all five source hashes from the current
  workspace.

Production membership changes, extraction, cancellation, and native fallback
did not receive a candidate implementation: the attention-only screen failed
before model-runner integration. Their existing native-path tests remain green.

## Benchmark-Only Implementation

`PagedBatchAttentionView` accepts multiple `PagedAttentionView` instances only
when they share the same physical pool, layer index, and block size. It builds
one padded `[B,max_blocks]` metadata array plus `[B]` sequence lengths and does
not retain pool arrays or block references.

`paged_batch_block_attention` extends Aster's existing 32-SIMD-group online
softmax math to:

- read one `[P,1,Hkv,block,D]` physical pool;
- choose blocks from the query row's two-dimensional table;
- derive each query offset from its sequence length; and
- emit `[B,Hq,Q,Dv]` without a B-by-prefix K/V tensor.

No production call site imports this class or function. The existing native
merge and Qwen attention bridge are unchanged.

## Primary Screen

The locked Qwen3.5 full-attention shape was FP16 `Hq=16`, `Hkv=4`, `D=256`,
`Q=1`, block size 64. Each scenario built a `K-1` shared prefix, forked B
request tables, and appended one private suffix token. Native control K/V was
materialized once before timing. Five warmups and 21 balanced/interleaved
measurements per method covered 2K/8K/10,334 tokens at B2/B4/B8.

| KV tokens | Batch | Candidate/native p95 | Max absolute error |
| ---: | ---: | ---: | ---: |
| 2,048 | 2 | 1.119x | 0 |
| 2,048 | 4 | 0.986x | 0 |
| 2,048 | 8 | 0.983x | 0 |
| 8,192 | 2 | 0.843x | 3.05e-05 |
| 8,192 | 4 | 1.137x | 6.10e-05 |
| 8,192 | 8 | 1.300x | 6.10e-05 |
| 10,334 | 2 | 1.168x | 3.05e-05 |
| 10,334 | 4 | 1.165x | 3.05e-05 |
| 10,334 | 8 | 1.574x | 3.05e-05 |

All numerical, non-materialization, and release gates passed. The locked B8
table has 161 common blocks and eight private tail blocks. Its candidate
metadata is 5,216 bytes versus 338,624,512 bytes for one native dense layer.
Conservatively multiplying metadata across eight full-attention layers while
leaving `ArraysCache` unchanged estimates:

- full-attention batch construction: 2,708,996,096 -> 41,728 bytes
  (`-99.9985%`);
- total merge growth: 3,121,126,344 -> 412,171,976 bytes (`-86.7941%`).

Both memory gates pass. The pool itself is existing request-owned storage, not
batch-construction growth; its geometric capacity is 67,108,864 bytes in this
single-layer probe.

## Five-Process Confirmation

The failed 10,334/B8 cell was repeated in five fresh processes with 30 warmups
and 200 balanced/interleaved measurements per method per process.

- median of process medians: native 4.271 ms, candidate 5.434 ms, ratio 1.272;
- median of process p95: native 8.944 ms, candidate 9.671 ms, ratio 1.177;
- process p95 ratios: 1.078 to 1.394; all five exceed the 1.03 ceiling;
- maximum absolute error: 6.10e-05; and
- every process releases to zero allocated blocks and zero pool bytes.

I046's earlier long-context kernel advantage used `Hq=16/Hkv=8/D=128`.
The locked 9B shape uses `Hkv=4/D=256`, doubling per-thread value reduction
work and synchronization iterations in this kernel. The older result is not a
contradiction and cannot override the current model-shaped confirmation.

## Decision

Reject this SIMD-group kernel shape before model-runner integration. It proves
that shared-prefix batch ownership can clear the memory target, but fails the
hard p95 no-regression gate in every confirmation process. Per the predeclared
stop rule, no locked 9B model A/B, cache membership/extraction implementation,
or production routing was run.

Retain the tested two-dimensional block-table boundary as experimental
scaffolding for a genuinely different kernel/Primitive or prefix-homogeneous
scheduler experiment. Do not optimize the rejected Python/Metal wrapper by
micro-tuning launch metadata; current vllm-metal evidence says metadata should
be built once per forward, and I047 already rejected a private Primitive
boundary without a stable kernel win.

The production native merge, eager clone, 8 GiB budget, reservation/eviction
policy, persistence schema, rollback switch, and configuration defaults remain
unchanged.

## Verification

- New focused contracts: `10 passed`
- Affected paged/Metal/harness suite: `33 passed`
- Full suite: `578 passed, 9 skipped, 1 warning`
- Touched-file Ruff and formatting: passed
- Retained artifact:
  `docs/loop-engineering/artifacts/ITER-20260801-086-shared-prefix-batch-attention-feasibility/attention-screen-summary.json`
