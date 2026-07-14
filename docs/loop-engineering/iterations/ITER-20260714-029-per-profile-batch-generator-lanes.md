# Iteration 029: Per-Profile BatchGenerator Lanes

Date: 2026-07-14

## Scope

- Starting commit: `5b750b9`
- Design/plan commit: `297a557`
- Source commit: `d791253`
- Machine: macOS 27.0, Apple Silicon arm64, Python 3.14.5, 24 GB unified
  memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`, mlx-audio `0.4.5`,
  transformers `5.12.1`

Iteration 028 proved that heterogeneous prompt/cache profiles must not share
one active BatchGenerator batch. The remaining bottleneck was that the guard
also serialized profiles that could theoretically make progress independently.
The hypothesis was that bounded per-profile generators would improve mixed
workloads without changing cache ownership or MLX thread ownership.

## Design and implementation

Added `engine.batch_generator_max_lanes` to `EngineSettings`, default `1`.
`BatchedEngine` now owns `_BatchLane` objects containing a profile key,
generator, request IDs, and lane-local UID mappings. Prefix extraction,
response routing, finish, cancellation, and generator cleanup use the owning
lane. The engine loop still calls every lane sequentially on the same worker;
there is no concurrent MLX call or new dependency.

Empty lanes are closed and recycled when a new profile arrives. This fixes a
real lifecycle issue discovered during the first benchmark: without recycling,
an old empty profile permanently consumed the lane limit and later requests
could wait forever.

The benchmark harness accepts `--max-lanes` and records the selected value in
its JSON output. The default benchmark behavior remains one lane.

## Reference designs consulted

- `examples/mlx-lm/mlx_lm/generate.py:1663`: `BatchGenerator.insert()` and
  the generator-owned UID/cache lifecycle.
- `examples/vllm-mlx/vllm_mlx/mllm_batch_generator.py:734`: request insertion
  and UID ownership in a serving batch generator.
- `examples/Rapid-MLX/vllm_mlx/mllm_batch_generator.py:516`: comparable
  insertion/removal boundary.

The selected design borrows the ownership boundary, not reference code. Aster
keeps one engine-loop owner because concurrent MLX streams were not proven
safe on this machine.

## Tests and commands

Focused TDD cycle:

```text
pytest -q tests/test_config.py::test_batch_generator_lane_limit_defaults_to_one_and_is_bounded
pytest -q tests/test_batched_engine.py::test_batched_engine_creates_bounded_lanes_for_incompatible_profiles
pytest -q tests/test_batched_engine.py::test_batched_engine_abort_removes_request_from_its_lane
pytest -q tests/test_batched_engine.py::test_batched_engine_extracts_prompt_cache_from_the_owning_lane
pytest -q tests/test_benchmark_batched_engine.py::test_benchmark_overrides_lane_limit_without_mutating_settings
```

Full verification:

```text
pytest -q                         # 432 passed, 9 skipped, 1 warning
ruff check ...                    # All checks passed
python -m compileall -q aster scripts tests
uv lock --check                   # passed
pip check                         # No broken requirements found
git diff --check                  # passed
```

## Benchmark A/B

The matched matrix used `/tmp/aster-qwen08-native.yaml`, Qwen3.5-0.8B 4-bit,
temperature `0.0`, prefix cache off, workloads `mixed,staggered`, concurrency
`2,4`, two rounds, and the existing request definitions from
`scripts/dev/benchmark_live.py`. Commands:

```text
python scripts/dev/benchmark_batched_engine.py \
  --config /tmp/aster-qwen08-native.yaml \
  --workloads mixed,staggered --concurrency-levels 2,4 --rounds 2 \
  --long-prompt-words 64 --prefix-cache off --max-lanes 1 \
  --output artifacts/.../lanes-1.json

python scripts/dev/benchmark_batched_engine.py \
  --config /tmp/aster-qwen08-native.yaml \
  --workloads mixed,staggered --concurrency-levels 2,4 --rounds 2 \
  --long-prompt-words 64 --prefix-cache off --max-lanes 2 \
  --output artifacts/.../lanes-2.json
```

Both lane limits completed all 8 records with zero request errors, zero swap
delta, and approximately `1.495 GB` MLX peak. For `mixed`, all four records
had exact response-hash parity and lane 2 reduced elapsed time by `2.90%` to
`5.78%`. For `staggered`, all four lane-2 records had response-hash drift and
were `3.46%` to `5.80%` slower.

The mismatch is reproducible and has a concrete cause: the first short
request creates a new profile lane before later staggered requests arrive;
those later requests join after the first `BatchGenerator.next()` step. The
resulting batch-membership change changes greedy output. When the same
staggered requests are submitted simultaneously, lane 2 matches lane 1. A
sequential single-request probe produces a third, stable hash, confirming
that this is a BatchGenerator batch-shape/membership sensitivity rather than
random sampling.

Additional lane-2 probes:

- Structured output at concurrency 2/4: 6/6 valid JSON responses, all
  `stop`, zero errors, zero swap delta.
- Prefix reuse at concurrency 2, two rounds: cache hits `1` then `2`, zero
  errors, exact repeated-response hashes.
- Cancellation at concurrency 4: 3 completed, 1 cancelled, follow-up
  completion length 8, zero running requests and zero pinned entries.

Raw artifacts:

- `artifacts/ITER-20260714-029-per-profile-batch-generator-lanes/lanes-1.json`
- `artifacts/ITER-20260714-029-per-profile-batch-generator-lanes/lanes-2.json`
- `artifacts/ITER-20260714-029-per-profile-batch-generator-lanes/structured-lanes-2.json`
- `artifacts/ITER-20260714-029-per-profile-batch-generator-lanes/reuse-lanes-2.json`
- `artifacts/ITER-20260714-029-per-profile-batch-generator-lanes/sequential-staggered-hashes.json`
- `artifacts/ITER-20260714-029-per-profile-batch-generator-lanes/summary.json`

## Decision

Retain the lane implementation as an explicit experimental capability, but
keep `batch_generator_max_lanes=1` by default. Do not claim a global
performance improvement and do not change the manual production runtime.
The lane-2 candidate does not pass the staggered token-parity gate. Rollback
of the implementation is `git revert d791253`; operational rollback is simply
setting the lane limit back to `1`.

## Next priority

Design a deterministic admission/cohort policy or a BatchGenerator-compatible
per-request state isolation mechanism that makes lane membership independent
of arrival timing. It must be tested against simultaneous, staggered,
streaming, cancellation, and long-context workloads before lane 2 can be
promoted.
