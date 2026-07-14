# Iteration 041 — Core reference matrix

Date: 2026-07-14

## Objective

Stop guessing about core inference behavior. Compare Aster with the mature
engines already cloned under `examples/`, then select one bounded improvement
that can be proven with tests and an A/B benchmark.

## Evidence reviewed

- `mlx-lm`: thread-local MLX stream ownership, wired memory limits, and the
  continuous `BatchGenerator` prompt/decode lifecycle.
- `vLLM-MLX`: stable active decode batches, cache extraction/filtering,
  generator cleanup, stream-safety tests, deterministic batching tests, and
  memory-stability tests.
- `Rapid-MLX`: waiting/running scheduling, trie/LRU prefix cache, pinning, and
  pressure-eviction coverage.
- `SGLang`: radix matching, page alignment, reference-protected nodes, and
  explicit eviction accounting.
- `OMLX`: chunked-prefill controls and a transient unfused-SDPA memory
  estimator used by a prefill guard.
- `z-lab/dflash`, `bstnxbt/dflash-mlx`, and `Aryagm/dflash-mlx`: draft/verify
  and rollback designs, retained as reference-only at this stage.

## Result

The comparison is recorded in
`docs/loop-engineering/CORE_REFERENCE_MATRIX.md`. Aster is already at the
reference level for the basic single-owner runtime boundary, explicit request
lifecycle, and stable manual decode batching. It is behind the references in
transient-aware prefill admission and in structural prefix-cache ownership.

## Decision

The next core implementation candidate is transient-aware prefill memory
admission. It will not change decode batching, cache representation, or MLX
thread ownership. The candidate must first reproduce the current baseline,
then add a failing unit test, implement the smallest route-aware estimator,
and pass deterministic correctness plus resource-aware A/B gates.

DFlash remains a later optimization family. It is not integrated until the
manual runtime foundation, cache lifecycle, and rollback/correctness gates are
strong enough to distinguish a real speculative-decoding gain from a cache or
scheduling regression.

