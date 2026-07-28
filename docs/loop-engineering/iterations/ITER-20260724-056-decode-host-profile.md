# Iteration 056: Profile Residual Decode Sampling Costs

- **Date:** 2026-07-24
- **Reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`; Iteration 055
  production changes remain uncommitted in the shared worktree
- **Hardware:** Apple M5, 10 CPU cores, 24 GB unified memory
- **Software:** macOS 27.0 (`26A5388g`), Python 3.14.5, MLX 0.32.0,
  MLX-LM 0.31.3
- **Model:** Qwen3.5-0.8B 4-bit

## Problem and Gates

Iteration 055 bounded built-in penalty history and left three plausible costs
in grouped asynchronous decode: host token materialization, per-row active
penalty graphs, and per-row full-vocabulary normalization. Iterations 052-054
already rejected raw-logit and neutral-processor changes in isolation, so this
iteration measured the remaining costs before changing another production
contract.

The early gate remained a `+3%` end-to-end median gain with exact token, text,
and cache output and no swap growth. A candidate below that gate across
B2/B4/B8 would stop before dynamic-membership and independent-process matrices.

## Code and Reference Analysis

CodeGraph traced `ModelRunner._decode_batch`, grouped `mx.async_eval`/`mx.eval`,
`_materialize_sampled_token`, and `_decode_result`. It also located MLX-LM
0.31.3's built-in processors. Repetition and presence use indexed replacement,
so duplicate token IDs are adjusted once; frequency uses scatter subtraction,
so duplicates accumulate. The benchmark-only tensorized path preserved those
semantics, processor order, per-row sampler calls, and RNG order.

A separate candidate replaced B per-row `logsumexp` reductions with one
`[B,V]` reduction for processor-free top-p rows. It did not batch samplers.

## Host Profile

The current Iteration 055 path was instrumented around whole-batch decode,
explicit/async evaluation, processor construction, sampled-token
materialization, and `DecodeResult` construction.

| Batch | Token materialization | Materialization + result construction |
| --- | ---: | ---: |
| B2 | `0.063%` | `0.274%` |
| B4 | `0.060%` | `0.294%` |
| B8 | `0.045%` | `0.225%` |

The pure `.item()` path never reached `0.1%` of batch time. Even a zero-cost
host post-eval path could not approach the admission gate, so bulk `.tolist()`
was not pursued. This agrees with Iteration 051's rejected sampled-scalar
concatenation evidence.

## Batched Penalty Screen

Both policies used the same 20-token work items. The candidate applied all
three penalties over a `[B,20]` token matrix and then invoked existing samplers
in row order.

| Batch | Steps | Median gain | Vectorized rows | Exact | Swap growth |
| --- | ---: | ---: | ---: | :---: | ---: |
| B2 | 128 | `+0.452%` | 288 | yes | 0 |
| B4 | 64 | `+0.009%` | 320 | yes | 0 |
| B8 | 128 | `+0.660%` | 1,152 | yes | 0 |

Every candidate batch hit the vectorized path. The maximum gain was `+0.660%`,
too small to justify new processor metadata and runtime branches.

## Batched Normalization Screen

Processor-free top-p rows retained normalized log-probabilities and per-row
sampler/RNG calls. Only the B reductions became one batch-shaped reduction.

| Batch | Steps | Median gain | Batched rows | Exact | Swap growth |
| --- | ---: | ---: | ---: | :---: | ---: |
| B2 | 128 | `-1.161%` | 288 | yes | 0 |
| B4 | 64 | `+1.782%` | 320 | yes | 0 |
| B8 | 128 | `+0.110%` | 1,152 | yes | 0 |

The result was neither monotonic with batch size nor consistently positive.
Its best cell remained below 3%, and B2 exceeded the 1% regression allowance.

## Decision

**Reject all three candidates and keep production unchanged.** Host token
materialization has insufficient addressable share. Batched penalties and
batched normalization preserve output but do not produce a stable end-to-end
gain at Aster's B2-B8 target.

The screens stop before long-context, dynamic-membership, and independent-
process confirmation because their first-stage effect sizes fail admission.
Raw payloads, source/model hashes, the aggregate, and the negative decision are
retained under
`docs/loop-engineering/artifacts/ITER-20260724-056-decode-host-profile`.

## Verification

- nine paired real-model payloads retained exact token/text/cache parity;
- all nine payloads recorded zero swap growth;
- artifact assertions: `4 passed`;
- artifact Ruff and `compileall`: passed;
- no production source file changed in this iteration.

## Next Priority

Aster-owned structured and thinking processors still receive full token history
through an MLX array and immediately convert it back to a Python list on eager
rows. The next iteration should measure a processor-declared Python-token path
while preserving arbitrary/custom processor input, parser state, structured
stop behavior, and dynamic membership.

## Fixed Loop Output

LOOP ITERATION: 056
ROOT CAUSE: Residual grouped-decode sampling costs were not separately bounded.
CHANGES: Added benchmark-only host, batched-penalty, and batched-normalization probes.
RESULT: All outputs matched, but maximum candidate gain was `+1.782%`; no production change admitted.
NEXT: Profile Python-token delivery for Aster-owned structured/thinking processors.
