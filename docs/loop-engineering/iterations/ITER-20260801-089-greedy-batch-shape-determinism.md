# Iteration 089: Greedy Batch-Shape Determinism

- Date: 2026-08-20
- Phase: completed diagnostic
- Baseline commit: `f1038fa50b7679c0b378de96583e6d1b5b1e13d4`
- Model: Qwen3.5-9B 4-bit, MLX 0.32.0, MLX-LM 0.31.3
- Scope: correctness attribution only; no production or performance candidate

## Objective

Classify I088's completion-index-6 greedy drift before considering a scheduler,
sampler, precision, or determinism change. The experiment must separate cache
merge/extract integrity, current cohort shape, accumulated decode history, and
the closest installed model-native MLX-LM boundary.

## Frozen Inputs

The public target is `mt-bench:84:turn-1`; its I088 companion for completion
indices 0-5 is `mt-bench:83:turn-1`. Model and tokenizer hashes are
`d77667c1...38b2a` and `94b66525...59e`. The target has 39 prompt tokens and
the companion has 57. Both Aster and MLX-LM reproduce the target prefix
`271,12646,25,357,2526,2923` and companion prefix
`271,332,3723,25,11819,220` before the divergent step.

The four diagnostic processes use balanced `single-first` / `batch-first`
orders. Forced candidate-logit evaluation makes every timing observation
invalid; the experiment retains no performance claim.

`measurement_status: invalidated` is intentional. I089 has no valid timing
delta (`baseline_delta: null`) because forced evaluation changes the execution
boundary. Its linked valid baseline is I088's source-bound frontier; I090 then
supplies the next valid 32-row cross-engine performance ledger. I089 therefore
classifies ownership only and cannot be read as a speedup or slowdown.

## TDD Contract

The initial red boundary was a missing classifier. The first green classifier
required four source-identical independent processes, balanced probe order,
immutable cache hashes, stable selected tokens, merge/extract parity, and
Aster/MLX-LM agreement. A real probe then falsified the initial assumption that
batch size two alone causes the drift, so the contract was expanded before the
next implementation to freeze serial and continuously paired histories plus
the exact heterogeneous companion cohort.

Five retained tests now cover the shared-reference classification,
merge/extract-sensitive rejection, source/state drift rejection, process/order
requirements, and artifact/source-hash recomputation.

## Experiment

Each process constructs three target states:

1. Serial Aster history through the six shared output tokens.
2. Aster history with the target continuously batched after the 57-token
   companion, matching I088 completion indices 0-5.
3. Native MLX-LM `GenerationBatch` history with the same row order, prompts,
   selected prefixes, and cache-layer merge/extract operations.

At the next token, each frozen target cache is tested as a single row, after
merge then extract back to one row, as a duplicated two-row cohort, and with
the original heterogeneous companion. The native comparison uses installed
MLX-LM `GenerationBatch`, not a reimplemented sampler loop.

## Results

All 22 classification gates pass in four independent AB/BA processes.

- Serial target cache SHA-256 is `491b3b82...40c4b`. Single, merge/extract,
  Aster duplicate batch, and native duplicate batch all select token 364.
- Continuously paired Aster and native MLX-LM target caches are byte-identical
  at `3d1f3322...c14bf`; they differ from the serial cache as expected.
- On that paired-history cache, single and merge/extract select token 8574.
- Duplicating that target state into two identical rows selects token 364 in
  both Aster and native MLX-LM.
- Reattaching the original companion in its original row order selects token
  421 in both Aster and native MLX-LM, reproducing I088 exactly.
- Candidate logits remain a BF16 near tie: the target leaders move among
  tokens 364, 421, and 8574 by 0.125 or an exact tie depending on history and
  cohort shape.
- Every canonical cache hash is unchanged after all probes. Merge then extract
  always matches a direct single-row prediction from the same state.

The retained 88,375-byte artifact contains the four full records, raw hashes,
source hashes, frozen cache metadata, and recomputed classification. Its
SHA-256 is `d4bfa13255ace4db107ea70b3925964de7f41423fd12dfdf50f2513c21b3dd45`.

## Decision

Classify the issue as `reference-shared-batched-history-cohort-arithmetic`.
The evidence falsifies Aster-specific cache corruption: Aster and native
MLX-LM create byte-identical paired-history caches and select the same token at
every controlled boundary. Ordinary BF16 model execution has cohort-dependent
argmax behavior on this exact near tie.

Keep production scheduling, cache merge/extract, precision, and greedy argmax
semantics unchanged. Epsilon tie-breaking, forced single-row decode, or a
higher-precision exception would create new semantics and need broad quality
plus serving-performance evidence; I089 provides no basis to admit one.

## Verification

- Focused I088/I089 tests: 11 passed.
- Full suite: 593 passed, 9 skipped, 1 dependency deprecation warning.
- Touched Ruff lint and formatting pass.
- Artifact JSON, raw/source hashes, model/tokenizer fingerprints, independent
  PIDs, balanced order, cache immutability, and all 22 gates recompute.

## Next

I090 returns to the foundation: build a source-bound Qwen3.5-9B Aster versus
direct/model-native MLX-LM parity matrix for B1 and concurrent public workloads.
Measure TTFT, prefill/decode throughput, end-to-end/tail latency, memory, and
swap before selecting another production hypothesis. MTP remains deferred.
