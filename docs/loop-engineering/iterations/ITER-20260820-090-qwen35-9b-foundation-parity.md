# ITER-20260820-090: Qwen3.5-9B Foundation Parity

## Objective

Establish a current source-bound baseline between Aster and direct/model-native
MLX-LM before selecting another Aster production optimization. The matrix
covers B1 short, B1 long, B4 simultaneous-short, and B4 mixed long/short public
cohorts.

## Baseline Contract

- Baseline/reference: direct/model-native MLX-LM on the same Qwen3.5-9B model
  and tokenizer files.
- Measured path: Aster under the same public records, greedy settings,
  prompt-chunking contract, output limits, cache state, and metric definitions.
- Matrix: 4 cells x 2 engines x 4 independent repetitions = 32 rows, with
  balanced first-engine order in every cell.
- Primary metric: `decode_driver_tps` (higher is better). Secondary metrics:
  aggregate generation throughput, prefill throughput, TTFT p95, end-to-end
  p95, peak MLX memory, peak RSS, and swap delta.
- Positive deficit means Aster is worse than the reference for the metric;
  negative means Aster is better. Values below are paired median ratios from
  the retained artifact, not pooled single-process observations.

## Performance Change

| Cell | Aggregate throughput | Decode-driver TPS | TTFT p95 | End-to-end p95 | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 short | +11.504% deficit | -8.132% | +21.306% | +11.504% | +1.049% |
| B1 long | +1.640% deficit | +5.978% | +1.612% | +1.640% | +11.445% |
| B4 short | +110.030% deficit | +36.322% | +183.324% | +110.023% | -0.921% |
| B4 mixed | -31.323% deficit | +24.484% | -31.674% | -31.323% | -32.763% |

The primary decode-driver gap is reproducible in the two B4 cells: Aster is
`36.322%` slower in B4-short and `24.484%` slower in B4-mixed. B1-long is not
order-stable (`+12.774%` versus `-2.731%` by first-engine stratum), while
B1-short favors Aster on decode-driver TPS (`-8.132%`). This is a workload
profile, not a global engine ranking.

## Correctness And Attribution

Source, input, execution-contract, order-balance, terminal, and resource gates
pass for all 32 rows. Cross-engine output identity has two declared divergences:
`b4-short/short-3` and `b4-mixed/short-0`; terminal identity remains exact.
Therefore the matrix selects a bounded decode-driver attribution profile but
does not authorize a production change or a global performance claim.

The retained summary names the next owner as `aster-manual-decode-driver` and
selects I091. I091 must align the B4 output contract and measure the common
decode-driver boundary with fresh independent A/B processes before changing
runtime code.

## Verification

- Retained evidence: `docs/loop-engineering/artifacts/ITER-20260820-090-qwen35-9b-foundation-parity/foundation-parity-evidence.json`.
- Focused tests: 16 passed (`test_foundation_parity_benchmark.py` and
  `test_greedy_batch_shape_diagnostic.py`).
- Full-suite verification at consolidation: 604 passed, 9 skipped, 1 warning.
- Production decision: no change; native merge, scheduler defaults, precision,
  and greedy semantics remain unchanged.
- Delivery: commit `9bf3fcf199b14caf3ec308e750f3ad22f726246c` was pushed to
  `origin/main`; I091 is anchored to that pushed baseline.

## Next Iteration

I091 measures the Aster-owned manual decode-driver profile on the B4-short and
B4-mixed cells. It must retain the same source/model/input locks, add a valid
timing boundary with no forced evaluation, preserve exact output/finish
semantics, and report baseline/candidate absolute values and percentage deltas
before any implementation is admitted.
