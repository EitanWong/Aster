# Iteration 053: Shift-Invariant Sampler Input Screen

- **Date:** 2026-07-23
- **Start/end reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`; no production source change
- **Model/runtime:** Qwen3.5-0.8B 4-bit, MLX 0.32.0, MLX-LM 0.31.3

## Hypothesis

Iteration 052 only passed raw logits to `temperature == 0` argmax. MLX-LM's
`min_p`, `top_k`, temperature scaling, and categorical sampling are also
invariant to adding one constant to all logits. `top_p` is not: it exponentiates
the supplied logprobs and compares their cumulative mass with an absolute
threshold. The candidate therefore retained normalization exactly when
`temperature > 0` and `0 < top_p < 1`.

## Controlled Screen

The benchmark reused Iteration 052's same-process adjacent pairing and
runner-assignment alternation. In the mixed workload, the candidate expanded
raw-logit rows from one of four to two of four; each measured B4 process
recorded `576` direct and `576` normalized rows, and each B8 process recorded
`1152` of each.

| Cell | Fresh processes | Speed change | Exact | Swap growth |
| --- | ---: | ---: | :---: | ---: |
| mixed B4 | 2 | `+1.22%` median (`+0.98%` to `+1.46%`) | yes | 0 |
| mixed B8 | 2 | `+0.05%` median (`-0.21%` to `+0.30%`) | yes | 0 |

A 32-step B4 smoke measured `+1.54%` and confirmed an 80/80 direct/normalized
split. It is retained as smoke evidence, not part of the screen summary.

## Decision

**Reject for production.** The broader mathematical contract remained exact,
but the incremental opportunity did not approach the 3% end-to-end gate and
disappeared at B8. Iteration 052 already covers the shared greedy and penalty
behavior, so repeating those unchanged cells would not alter this decision.
No sampler metadata or runtime branch was added.

The next profile target is a concrete processor-construction defect: Aster
passes the neutral default `repetition_penalty=1.0` to MLX-LM, whose current
factory creates a real gather/where/scatter processor for every default request.
That processor performs multiply/divide by one and should first be measured as
an isolated benchmark candidate.

## Verification

- Artifact tests: `3 passed`.
- Ruff: passed.
- Four measured payloads: exact token/text/cache parity and zero swap growth.
- Rollback: remove the Iteration 053 artifact directory; serving code is unchanged.
