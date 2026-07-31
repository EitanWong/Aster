# Iteration 066: Public Cross-Engine Foundation

- **Date:** 2026-07-28
- **Phase:** admitted foundation
- **Scope:** benchmark infrastructure plus one complete scoped public
  Aster/direct-MLX-LM matrix. The unstarted isolated short-decode attribution
  plan is superseded before any production change.

## Objective

Make public, pinned test data and complete cross-engine comparability a hard
precondition for future Aster performance decisions. This prevents a local
single-scenario improvement from being mistaken for a general engine result.

## Primary Metric

The first primary metric is comparison completeness: the percentage of selected
public workload records covered exactly once by every required compatible engine
with equal model/Tokenizer fingerprints, equal generation settings, equal prompt
hashes, deterministic token parity, and complete latency/resource metrics.

## Hypothesis

MT-Bench covers interactive short-context behavior while LongBench covers
published long-context tasks. A version-pinned source lock, local engine
inventory, and executable result gate will reveal whether a performance gap is
real, scenario-scoped, or merely an input/runtime availability difference.

## Predeclared Gates

1. Downloaded sources must match locked revision, SHA-256, byte size, and
   structural record count before a workload is generated.
2. MT-Bench prompts are verbatim public first turns. LongBench prompts use only
   the project-provided official templates and official output limits.
3. Every locally available compatible engine is required. An unavailable or
   incompatible engine has an explicit inventory reason rather than a silent
   omission.
4. `validate-results` rejects a matrix with missing record coverage, model or
   Tokenizer drift, prompt drift, deterministic output drift, or absent
   TTFT/prefill/decode/end-to-end/RSS/swap fields.
5. `cross-engine-core` supports only a scoped bottleneck screen. A complete
   statement within the declared MT-Bench plus LongBench-primary scope requires
   `full-public` without per-stratum limiting.
6. No Aster production code changes occur in this iteration.

## Implementation

- `PublicWorkloadResolver` reconstructs each prompt from the pinned raw source
  at execution time. It verifies MT-Bench rows and LongBench JSONL/template
  hashes before yielding token input; result files contain hashes and counts,
  not prompt text.
- `public_engine_matrix.py` runs one engine/task shard per fresh process,
  alternates engine-first order across its six shards, and atomically persists
  each completed shard for resume.
- The Aster and direct-MLX-LM adapters use the same greedy contract, public
  source-rendered token IDs, official LongBench head/tail truncation, 32,768
  token input ceiling, 8-token process warmup, and explicit 2,048-token
  prefill step. The explicit chunk parity removed an earlier long-prompt output
  drift caused by Aster using a 1,024-token step while direct MLX-LM used 2,048.
- Each record captures input/output token hashes, TTFT, prefill/decode/end to
  end timing, sampled peak RSS, and swap delta. Aster scopes its decode-cache
  maintenance counter to the public request so warmup state cannot alter a
  later response's maintenance boundary.

## Matrix Result

- `cross-engine-core` completed all 1,380 public records on both required local
  engines, producing 2,760 engine-records. The ignored raw bundle is
  `run/loop-engineering/ITER-20260728-066-public-cross-engine-foundation/core-matrix`.
- Source lock SHA-256: `d6d0877b...a5211c16`; workload SHA-256:
  `d6c7fa00...a9444e46`; model SHA-256: `54c8e234...235fa0cf`; Tokenizer
  SHA-256: `50e396ca...681f2cf0`.
- All eight comparison gates passed: complete coverage, required engines,
  public prompt identity, effective input token identity, execution contract,
  model/Tokenizer fingerprints, deterministic output token parity, and metric
  completeness. Both engines had zero maximum swap delta.
- Across the scoped set, paired median Aster/direct-MLX-LM ratios were
  +15.762% decode throughput, -8.509% prefill throughput, -5.112% TTFT,
  +0.975% end-to-end time, and -3.304% peak RSS. Positive latency deltas are
  slower for Aster; throughput deltas are faster for Aster.
- The screen is length-dependent: Aster prefill was lower in the three bins
  below 8,192 input tokens and higher by 5.897% in the 8,192-32,768 bin. Decode
  and end-to-end directions also vary by workload. The compact, hash-bound
  summary is
  `docs/loop-engineering/artifacts/ITER-20260728-066-public-cross-engine-foundation/core-matrix-summary.json`.

## Decision

**Admit the public adapter and comparable core-matrix foundation. Reject a
production performance selection and any global engine ranking.** This is one
order-alternated matrix, not a crossed replication: each engine/task shard ran
once, and the profile is explicitly scoped. I067 will repeat the whole core
matrix with every shard order reversed before attributing a bottleneck.

## Bounded Artifact Set

- `docs/loop-engineering/benchmarks/public-dataset-lock.json`: tracked public
  source provenance and integrity lock.
- `scripts/dev/public_benchmark.py`: download, verify, inventory, workload, and
  result-comparability CLI.
- `run/loop-engineering/public-benchmarks/`: ignored downloaded data, install
  manifest, workload manifests, inventory, and future result records.
- `tests/test_public_benchmark.py` and `tests/test_public_engine_matrix.py`:
  focused offline verification of source resolution, token-result gates,
  ordering, truncation, and aggregate restoration.
- `docs/loop-engineering/artifacts/ITER-20260728-066-public-cross-engine-foundation/core-matrix-summary.json`:
  compact source/result hashes and descriptive scoped statistics.

## Deferred Work

I067 owns the crossed reverse-order confirmation. I061 remains historical local
evidence only because its prompt was locally constructed; it must not choose
the next production optimization.
