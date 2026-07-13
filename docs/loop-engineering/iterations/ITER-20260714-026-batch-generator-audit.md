# Iteration 026: BatchGenerator and Compatible Package Audit

Date: 2026-07-14

## Goal

Audit the model-native `mlx_lm.BatchGenerator` boundary after the prefill
microbatch rollback, while keeping the runtime package set at the newest
versions compatible with the checked-in declarations.

## Package audit

`uv lock --upgrade` resolved 72 packages and produced no lockfile changes.
The relevant installed versions are already the newest compatible set:

- `mlx==0.32.0`
- `mlx-lm==0.31.3`
- `mlx-audio==0.4.5`
- `fastapi==0.139.0`
- `pydantic==2.13.4`
- `transformers==5.12.1`

The newer `transformers==5.13.1` cannot be selected because
`mlx-audio==0.4.5` declares `transformers<5.13.0`. Its compatible
`tokenizers` range also excludes the current PyPI latest `0.23.1`, so the
resolved `tokenizers==0.22.2` is intentional. `pydantic==2.13.4` pins
`pydantic-core==2.46.4`; upgrading that transitive package independently
would violate the installed Pydantic pin.

## BatchGenerator audit

The installed API is:

- `BatchGenerator.insert(prompts, max_tokens, caches, all_tokens, samplers,
  logits_processors, state_machines)`
- `next_generated()` returns generated responses only
- `remove(uids, return_prompt_caches=False)` removes active sequences
- `extract_cache(uids)` returns cache plus token state

The existing `BatchedEngine` completed a real 0.8B smoke with four concurrent
requests, 64 completion tokens, and clean request cancellation. It remains an
experimental engine strategy selected by `engine.engine_type=batched`; the
default manual engine was unchanged.

The prefix-cache smoke exposed a migration blocker: the engine finds and pins
a cached prefix but deliberately inserts the request with `prompts` instead of
passing `caches` to `BatchGenerator`. Two sequential identical requests both
reported `prefill_cache_hit=false`, and the second request recomputed its full
196-token prompt. The response path also hardcodes cache-hit flags to false.

## Decision

Do not enable or broaden the BatchGenerator serving path in this iteration.
No source change was justified by the package audit. The next implementation
must first define cache ownership and restore/store parity, then add
deterministic serial-vs-batched response tests, cancellation tests, and
multi-trial mixed/reuse benchmarks before changing defaults.

## Evidence

Raw audit output is in `artifacts/ITER-20260714-026-batch-generator-audit/`.
