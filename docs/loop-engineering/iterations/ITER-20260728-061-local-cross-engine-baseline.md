# Iteration 061: Local Cross-Engine Baseline

- **Date:** 2026-07-28
- **Phase:** complete
- **Scope:** benchmark protocol and baseline evidence only; no production
  performance candidate is authorized in this iteration.

## Objective

Establish whether Aster's manual runtime and an installed local reference
runtime can execute an equivalent greedy workload with the same model files,
tokenizer/chat template, prompt token IDs, completion cap, stop behavior,
cache policy, warmup, and measurement boundaries.

## Primary Metric

When equivalence is demonstrated, report fixed-length completion tokens per
second and elapsed time for the same single-request workload. Do not compute a
relative speed ranking when model input, tokenizer, template, sampling,
stopping, cache semantics, or output token sequence differs.

## Initial Protocol

1. Inventory available local reference runtimes and versions without installing
   or updating dependencies.
2. Freeze a 0.8B model signature, raw prompt tokens, greedy sampling settings,
   output length, warmup, cache policy, and process-isolation protocol.
3. Run a one-process equivalence preflight. Record token/text/finish behavior,
   model input hashes, and every mismatch.
4. Only after a passing preflight, run a bounded paired matrix with alternating
   order and independent processes. If no compatible local reference exists,
   record the blocked comparison and retain no synthetic ranking.

## Preflight Result

The installed reference is `mlx-lm 0.31.3` over `mlx 0.32.0`. Aster's manual
runtime and direct `mlx_lm.stream_generate` loaded the same local Qwen3.5-0.8B
4-bit files and tokenizer. For a raw 128-word prompt, 64 greedy completion
tokens, and an 8-token in-process warmup, both paths produced identical prompt
IDs, all 64 completion IDs, text hash, and `length` finish reason. Neither
process increased swap. The unreplicated observation was 176.966 tok/s for
Aster and 158.489 tok/s for direct MLX-LM, which admits a paired matrix but is
not a performance conclusion.

## Formal Matrix Protocol

- Scenarios: 128-word prompt / 256-token completion and 2,048-word prompt /
  64-token completion.
- Six independent AB/BA pairs per scenario, with three Aster-first and three
  MLX-LM-first pairs.
- Each child process reloads the same model, executes the same local warmup,
  and records source/model hashes, prompt/output IDs, text hash, finish reason,
  completion throughput, elapsed timings, RSS, and swap observations.
- A pair is excluded only by a predeclared hard failure: model/settings/prompt/
  token/text/finish mismatch, swap growth, missing records, or source drift.
  The matrix records every completed process and rejects the comparison if any
  gate fails.

## Success Gate

The iteration succeeds as a baseline only when its artifact records either a
reproducible equivalent comparison or a precise, evidence-backed incompatibility
reason. It does not change Aster production code.

## Formal Result

The formal matrix admitted the bounded comparison. Each of the 12 pairs used
two fresh model processes, with three Aster-first and three MLX-LM-first pairs
per scenario. Every pair matched model-file hashes, settings, raw prompt IDs,
completion IDs, text hash, and `length` finish reason. All 24 process records
had non-growing swap and stable per-engine source/environment fingerprints.

| Scenario | Aster p50 | MLX-LM p50 | Paired Aster change | 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| 128-word prompt, 256-token decode | 175.889 tok/s | 188.482 tok/s | -5.743% | [-9.228%, -4.499%] |
| 2,048-word prompt, 64-token decode | 156.087 tok/s | 141.361 tok/s | +10.431% | [+5.749%, +13.394%] |

The short-decode result is a reproducible local regression against direct
MLX-LM for this configuration. The long-prompt result is a reproducible local
advantage for the same bounded completion metric. The opposite signs mean no
overall ranking is justified. Post-load RSS growth was recorded for diagnosis,
not treated as peak unified-memory or energy evidence.

## Evidence And Verification

- `formal-evidence.tar.gz`: 38 members / 31,783 bytes, containing the manifest,
  all 24 raw child records, all 12 pair comparisons, and the aggregate.
- `final-admission.json`: all six evidence gates pass; archive SHA-256 is
  `e554a596ff923b524ae567b038b46e042dfb9cbf8fda67105a0751a311613d0a`.
- `test_artifacts.py`: archive-only aggregate recomputation and source/archive
  hash checks passed (`2 passed`).
- `ruff check` and Python byte compilation passed for every I061 artifact tool.

## Conclusion And Next Priority

Decision: **admit** the comparison infrastructure and its scenario-scoped
baseline; retain no production code change. The highest-value next question is
why short-context fixed-length decode is slower than direct MLX-LM while the
long-context case is faster. The next iteration will profile per-token manual
runtime overhead with the frozen I061 model, prompt, sampler, and exact-output
gates before proposing a production candidate.
