# Iteration 083: Exact-Hit Commit And Sustained Lifecycle Loop

- Date: 2026-07-31
- Phase: completed
- Baseline: I082 production-budget admission artifact; the I081/I082 source
  change is still uncommitted in the working tree
- Scope: package the admitted exact-hit predicate and run a short repeated
  lifecycle loop; no cache budget or eviction-policy change.

## Objective

Convert the source-bound I081/I082 candidate into a small reviewable runtime
change, then exercise repeated exact hits, strict-prefix appends,
cancellation, and persistence boundaries after the commit. Keep the explicit
rollback switch and the 8 GiB production default visible in the evidence.

## Predeclared Work

1. Review the final diff and stage only the engine predicate, focused runtime
   tests, loop-engineering state, and compact I081/I082 artifacts required for
   the decision.
2. Create a conventional commit only after the user requests the push/commit
   action or the next execution turn confirms that authorization.
3. Run a short repeated exact/strict-prefix/cancel lifecycle loop in fresh
   processes, checking output hashes, cache counters, pin cleanup, and trace
   bounds on every cycle.
4. Keep the persistence compatibility switch and
   `snapshot_skip_full_prompt_on_prefix_hit=false` as rollback controls.

## Success Gates

- The final diff contains no budget, representation, or unrelated scheduler
  changes.
- Repeated cycles preserve exact token/text/finish identity, no duplicate
  exact-hit store, zero terminal active/pinned state, and bounded trace data.
- Focused and full verification remain green after packaging.
- Any failure leaves the admitted source uncommitted and restores the explicit
  refresh behavior.

## Current Progress

The runtime diff remains one predicate line. I083 also adds two measurement-only
arrival scenarios: a serial cold-plus-eight-exact chain and a source-derived
strict-prefix append followed by an identical repeat. Only those scenarios add
compact per-request terminal lifecycle snapshots. Existing plan serialization
omits the new optional suffix field, and the I080 window-6 plan remains byte
identical at SHA-256 `9612ccb4...e1e`.

Test-first evidence was preserved. Three sustained-exact tests failed because
the scenario/count/snapshots did not exist, then passed. Two strict-prefix tests
failed because the derived scenario did not exist, then passed. The complete
arrival harness is `22 passed`; touched Ruff and `git diff --check` pass.

## Real-Model Lifecycle Results

The production-8-GiB Qwen3.5-9B sustained exact process completed one cold
10,334-token request and eight serial exact replays in `20.094584s`. Every
replay used 10,334 cached tokens and zero prefill steps. Exact hits advanced
from zero through eight while stores, entries, and trace events stayed at one;
every terminal sample had zero active/pending/pinned state and zero dropped
events. All nine output token hashes (`9bfc5326...b9fe`), text hashes
(`6893ba97...cd70`), and length finishes match. Swap delta is zero.

The strict-prefix process extended the same public prompt from 10,334 to
10,342 tokens. Both derived requests reused 10,334 tokens, performed one
prefill step, produced identical token/text hashes, and advanced prefix hits
from zero to two. Stores, entries, and trace events stayed at one, terminal
pins stayed at zero, and swap delta stayed zero. This proves the append is a
real token-level strict prefix rather than a label inferred from prompt text.

A fresh cancellation process accepted the long-prefill cancellation, recorded
one cancelled request and one cancelled checkpoint, completed the deterministic
MT-Bench follow-up, and ended with zero requests/queues/active bytes/pins. Its
follow-up token/text/finish and 85,065,728-byte one-store snapshot match I082.

## Cross-Engine And Research Gates

Fresh two-record Aster/direct-MLX-LM processes match source lock, workload,
model/tokenizer, generation, prompt token IDs, output token IDs, text, and
finish, with zero swap in both engines. This is a compatibility smoke only.

The read-only upstream refresh pinned vLLM `82ae4164`, SGLang `5f9b0db1`, and
MLX-LM `e5baded8`; MLX-LM's current cache source is byte-identical to the local
example. SGLang's radix lock refs/evictable leaves, vLLM's block refcounts, and
Rapid-MLX's optional radix lookup index were compared with Aster. Aster already
has O(1) exact lookup, distinct-length prefix probes, and sorted-neighbor LCP;
a Python token trie is therefore not selected without a new measured lookup
bottleneck. Current papers on semantic/position-independent caching and
distributed reservation do not transfer directly to this local exact-cache
boundary. TraceLab and the shared-block implementations instead motivate a
concurrent exact-hit ownership screen.

## Decision

I083 is complete. The lifecycle predicate remains admitted and uncommitted;
the configured 8 GiB budget, eviction policy, snapshot representation, and
rollback switch are unchanged. Full verification is `559 passed, 9 skipped,
1 warning`. No commit or push was created in this iteration.
