# Iteration 044 — Prefix clone baseline

Date: 2026-07-15

## Objective

Measure the exact-prefix hit path before changing Aster's full-snapshot
ownership model. The reference matrix identified `PrefixStore.lookup()` and
`ModelRunner.clone_cache()` as possible long-Agent costs; this probe tests the
most direct exact-hit case first.

## Reference and current path

SGLang and Rapid-MLX use structural prefix indexes, pinning, and cache-sharing
rules. Aster currently stores full snapshots, uses a sorted token index in
`aster/inference/prefix_store.py`, then calls `copy.deepcopy()` and cache trim
in `aster/inference/model_runner.py` before using a hit.

## Probe

One process, Qwen3.5-9B 4-bit, prefix cache enabled, greedy sampling, 8,373
prompt tokens, and 32 completion tokens. A cold request was followed by the
same request, so the second request used one exact snapshot with 8,372 reused
tokens and a 325,844,992-byte cache entry.

| Metric | Cold | Exact hot |
| --- | ---: | ---: |
| Total latency | 22.177 s | 2.521 s |
| Admission preparation | 2.162 s | 9.545 ms |
| Model prefill | 17.527 s | 0.000 s |
| Prefill steps | 6 | 0 |
| Prefix reused tokens | 0 | 8,372 |
| Output SHA | `d1ca7425...a8084` | identical |

## Decision

Do not change snapshot clone ownership yet. At this cache size the exact-hit
admission path is only 9.545 ms; decode dominates the 2.521 s hot request.
The evidence does not support a complex structural-sharing or clone rewrite.

## Next priority

Measure `PrefixStore.lookup()` with many retained Agent snapshots and long
branch prompts. That isolates the remaining high-cardinality index question
without conflating it with model decode time.

