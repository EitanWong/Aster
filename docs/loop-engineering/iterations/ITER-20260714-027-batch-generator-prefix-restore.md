# Iteration 027: Restore Prefix Caches in BatchGenerator

Date: 2026-07-14

## Scope and hypothesis

Starting commit: `91f612c`.

The previous audit showed that `BatchedEngine` detected prefix entries but
discarded them before `BatchGenerator.insert()`. The hypothesis was that
capturing the cache at the prompt boundary and inserting a private clone would
remove repeated prefill work while preserving deterministic output.

Success gates were exact greedy token/text parity, correct stop reasons,
successful cancellation cleanup, no pinned-cache leak, no swap regression, and
at least a 3% improvement for repeated-prefix workloads. The default manual
runtime and the unavailable `BatchGeneratorRuntimeKernel` boundary were not
changed.

## Design

`mlx-lm 0.31.3` exposes the reusable cache before the final prompt token is
fed through `BatchGenerator.next()`: the preceding prompt response has
`end_of_segment=true`, `end_of_prompt=false`, and progress `(N-1, N)`. The
engine now consumes `next()` instead of `next_generated()`, snapshots that
cache boundary, and inserts the first uncached token plus suffix with
`caches=[...]`.

The stored cache is cloned before insertion because BatchGenerator mutates its
cache objects. Prefix entries are pinned while a request is live and unpinned
on completion, cancellation, scheduling failure, and engine shutdown. Cache
hit flags now reflect an actually restored cache. `all_tokens` contains only
the already cached prefix, preventing duplicate logits-processor history.

Snapshots whose cache offset cannot be safely rewound are rejected. This
preserves the existing safety boundary for divergent LCP matches and hybrid
cache types.

## Changes

- `aster/inference/batched_engine.py`
  - Added prompt-boundary snapshot extraction and cache insertion.
  - Added cloned cache ownership and offset validation.
  - Added prefix pin release on every terminal path.
  - Removed per-step cache extraction and the incorrect post-generation
    prompt snapshot.
  - Corrected response cache-hit flags and token-history initialization.
- `tests/test_batched_engine.py`
  - Added failing-first tests for insertion offsets and prompt-boundary store.

## Verification

Commands:

```text
pytest -q
ruff check aster/inference/batched_engine.py tests/test_batched_engine.py
python -m compileall -q aster tests
uv lock --check
pip check
```

Results: `420 passed, 9 skipped, 1 warning`; Ruff, compileall, lock, package
compatibility, and diff checks passed.

## Benchmark evidence

Machine: macOS 27.0 arm64, Python 3.14.5, MLX 0.32.0, mlx-lm 0.31.3,
24 GB unified memory. All requests used greedy sampling and the local
Qwen3.5 4-bit models.

The 0.8B four-request repeated-prompt matrix measured no-prefix median
`1.255s / 204.0 completion tok/s` versus prefix-hot median
`0.979s / 261.6 completion tok/s`: `-22.0%` elapsed and `+28.2%`
throughput. Cold prefix elapsed and peak MLX memory were within the no-prefix
control range; all 12 response hashes matched.

Long-context exact reuse showed the cache's intended value:

| Model / prompt | Cold | Hot | Peak MLX | Swap delta | Cache |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B / 12,295 tokens | 5.725s | 0.484s | 1.679 GB | 0 | 170.6 MB |
| 9B / 12,295 tokens | 35.755s | 3.375s | 6.852 GB | 0 | 454.4 MB |

Both pairs returned 32 completion tokens with identical SHA-256 text hashes.
An append-only Agent-style prompt reused 481 cached tokens and also matched
the no-cache output hash. A cancellation probe completed three 128-token
requests, cancelled one request with `request_cancelled`, left zero running
requests and zero pinned entries, and accepted a follow-up request. A stream
probe emitted 12 chunks including one terminal chunk and left zero pins.

## Decision and risks

Keep the change in the experimental `engine_type=batched` path. It clears the
performance and correctness gates for exact and strict-prefix reuse, but does
not make the model-native runtime globally production-ready. Divergent LCP
reuse remains unsafe for the hybrid `ArraysCache + KVCache` shape, and the
separate `BatchGeneratorRuntimeKernel` adapter remains unavailable.

Rollback: revert `68b0a2b`; the manual production runtime is unaffected.

## Next priority

Run the full BatchedEngine workload matrix at concurrency 2/4/8, including
mixed, reuse, staggered, structured output, and long-running cancellation.
Use those results to decide whether to expose BatchedEngine as an eligible
runtime strategy, while separately investigating safe append-only handling for
hybrid cache LCP matches.
