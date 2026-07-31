# Public Benchmark Contract

This directory tracks the small, reviewable contract for Aster performance
inputs. Downloaded data and generated result files belong under ignored
`run/loop-engineering/public-benchmarks/`.

## Source Data

`public-dataset-lock.json` pins the source URL, immutable revision, byte size,
SHA-256, license note, and structural validation for each public source.

- MT-Bench supplies 80 public multi-turn questions. The deterministic workload
  uses the verbatim first turn rather than a locally composed chat prompt.
- LongBench v1's downloaded archive contains the 4,750-record / 21-task primary
  corpus and the 3,668-record / 13-task LongBench-E extension. The tool uses
  the primary corpus with its official prompt templates and output limits, both
  pinned in the source lock.

Run once to fetch and validate data, then run the offline check before a matrix:

```bash
uv run python scripts/dev/public_benchmark.py sync
uv run python scripts/dev/public_benchmark.py verify
```

## Workloads

```bash
uv run python scripts/dev/public_benchmark.py inventory \
  --output run/loop-engineering/public-benchmarks/engine-inventory.json
uv run python scripts/dev/public_benchmark.py build-workload \
  --profile cross-engine-core \
  --output run/loop-engineering/public-benchmarks/cross-engine-core.json
```

`cross-engine-core` is a scoped diagnostic workload. `full-public` includes all
locked MT-Bench records plus the complete LongBench v1 primary corpus and is the
only profile eligible for a complete comparison within that named scope.
`--limit-per-stratum` creates a screen-only workload even when its inputs come
from public data.

The workload manifest contains public record IDs, renderer identities, prompt
hashes, and output caps, not copied prompts. Engines retrieve the pinned source
record at execution time, preventing a local prompt rewrite from changing one
engine's input silently.

## Aster And Direct MLX-LM Matrix

`public_engine_matrix.py` is the source-bound adapter for the currently
available Aster and direct `mlx-lm` engines. It resolves each workload row from
the pinned sources, verifies the source-row/template/prompt hashes, tokenizes
the resolved text with the shared model tokenizer, and applies LongBench's
official half-head/half-tail policy only when the configured input window is
exceeded. Both adapters use the same explicit 2,048-token prefill chunk, which
matches direct MLX-LM's upstream default and avoids a chunk-boundary numerical
drift on long prompts. It records the effective input token hash separately
from the source prompt hash.

The matrix runs each engine/task shard in a fresh process, warms that process
once on its first public row, and alternates Aster/MLX-LM order by shard. This
keeps model loading and same-process state out of the cross-engine boundary
while retaining task-local sequential throughput measurements. A stopped run
can resume only when its workload hash, model fingerprint, and execution
contract match. Aster's decode allocator-maintenance counter is reset at each
public request boundary, so a preceding warmup cannot force a mid-response
allocator clear at a token position the direct reference never sees. The
adapter records this request-scoped condition in the execution contract; it
does not change production serving code.

```bash
uv run python scripts/dev/public_engine_matrix.py run-matrix \
  --workload run/loop-engineering/public-benchmarks/cross-engine-core.json \
  --run-dir run/loop-engineering/ITER-20260728-066-public-cross-engine-foundation/core-matrix
```

The final directory contains one partial record per engine/task shard,
aggregated `aster.json` and `mlx-lm.json` result files, `comparison.json`, the
matrix manifest, and the model fingerprint. Use `--resume` with the same
arguments after an interrupted run.

## Engine Results

Each engine writes one JSON object with this shape:

```json
{
  "engine": "aster",
  "engine_version": "...",
  "workload_sha256": "...",
  "generation": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": 0},
  "execution": {"input_truncation_policy": "official-longbench-half-head-half-tail", "max_input_tokens": 32768},
  "model_fingerprint": {"model_sha256": "...", "tokenizer_sha256": "..."},
  "records": [{
    "workload_id": "...",
    "prompt_sha256": "...",
    "prompt_token_ids_sha256": "...",
    "prompt_token_count": 0,
    "output_token_ids_sha256": "...",
    "metrics": {
      "ttft_seconds": 0.0,
      "end_to_end_seconds": 0.0,
      "prefill_tokens_per_second": 0.0,
      "decode_tokens_per_second": 0.0,
      "peak_rss_bytes": 0,
      "swap_delta_bytes": 0
    }
  }]
}
```

Validate every required compatible engine together before interpreting a gap:

```bash
uv run python scripts/dev/public_benchmark.py validate-results \
  --workload run/loop-engineering/public-benchmarks/cross-engine-core.json \
  --result run/loop-engineering/public-benchmarks/aster.json \
  --result run/loop-engineering/public-benchmarks/mlx-lm.json \
  --required-engine aster \
  --required-engine mlx-lm \
  --output run/loop-engineering/public-benchmarks/cross-engine-core-comparison.json
```

The validator rejects incomplete coverage, unequal model/tokenizer fingerprints,
source prompt drift, effective input-token drift, execution-contract drift,
deterministic output-token drift, and missing timing or resource metrics.
Unavailable engines stay visible in the inventory with their probe result; they
are not silently excluded.
