# Decisions

## 2026-07-13: Admit Before Prefill Continuation

- Decision: retain the manual scheduler and process waiting admissions after decode but before prefill.
- Reason: request timelines and existing tests showed short requests could wait behind a long prefill even when decode was idle.
- Alternative rejected: enabling `batch_generator` immediately. Its adapter is still marked unavailable and would bypass required compatibility evidence.
- Tradeoff: newly admitted prompts can preempt an existing prefill continuation, increasing fairness and short-request responsiveness while delaying the long prompt by one scheduler turn.
- Rollback: revert commit `32addf1`.
- Scope: mixed scheduling and admission latency; no claim about single-request decode kernel speed or global throughput.
- Superseded by the randomized A/B re-evaluation and rollback decision below.

## 2026-07-13: Keep Paged KV as an Experimental Materialization Boundary

- Decision: retain `39502be` as a lossless block ownership/COW adapter, but do
  not enable it in either production runtime.
- Evidence: Qwen3.5-0.8B 2K chunked prefill matched native logits exactly; the
  materializing adapter was `1.29%` slower at 2K and statistically flat at 8K,
  so it did not pass the 3% performance gate or reduce retained KV memory.
- Reason: current MLX-LM attention consumes contiguous K/V and cannot consume a
  block table directly. Enabling the adapter now would add copies without
  proving a user-visible gain.
- Next experiment: validate a block-indexed MLX/Metal kernel against
  `PagedAttentionView`, then add hybrid-cache bundle lifecycle and batch merge
  support.
- Rollback: remove `39502be` and the associated experimental tests/docs; the
  default native cache path is unchanged.

## 2026-07-13: Randomized A/B Re-evaluation

- Result: do not accept `32addf1` as a default performance profile yet.
- Evidence: seven interleaved baseline/current trials with greedy sampling gave current elapsed median `+2.86%` and completion throughput `-2.78%`; 95% bootstrap intervals included zero.
- Interpretation: the earlier grouped `-13.6%` result was affected by execution order or workload shape. The scheduler behavior may still help only when short requests arrive during an active long prefill.
- Next experiment: use a staggered arrival workload that submits the long prompt first and short requests after prefill has begun.

## 2026-07-13: Roll Back Admission Scheduling Candidate

- Decision: revert `32addf1` with `5f2b952`.
- Evidence: seven interleaved staggered trials per side with identical 272-token outputs showed short-request p95 `2.4447s -> 2.7454s` (`+12.3%`), aggregate elapsed `4.2352s -> 4.1885s` (`-1.1%`), and completion throughput `64.223 -> 64.940 tok/s` (`+1.1%`). All bootstrap intervals crossed zero.
- Reason: aggregate throughput did not improve materially and the protected short-request metric regressed in the measured workload. Unit-level fairness behavior is insufficient to retain a runtime policy without a stable end-to-end gain.
- Retained work: keep the deterministic/resource-aware/staggered benchmark harness and all raw artifacts for future scheduler candidates.
