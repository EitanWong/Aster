from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_runtime_hashes(model: Path) -> dict[str, str]:
    runtime_suffixes = {".json", ".jinja", ".model", ".safetensors", ".txt"}
    return {
        path.relative_to(model).as_posix(): _sha256(path)
        for path in sorted(model.rglob("*"))
        if path.is_file()
        and path.suffix in runtime_suffixes
        and not any(part.startswith(".") for part in path.relative_to(model).parts)
    }


def validate_records(
    records: list[dict[str, object]], *, strict: bool = False
) -> None:
    seen: set[tuple[str, str, int, int]] = set()
    seen_pids: set[int] = set()
    reference_sources: dict[str, str] | None = None
    reference_environment: dict[str, object] | None = None
    settings_by_variant: dict[str, dict[str, object]] = {}
    for record in records:
        if record.get("schema_version") != 1:
            raise ValueError("unsupported schema version")
        mode = record.get("mode")
        variant = record.get("variant")
        run_id = record.get("run_id")
        context_words = record.get("context_words")
        if mode not in {"control", "profile"}:
            raise ValueError("invalid benchmark mode")
        if variant not in {"native", "direct"}:
            raise ValueError("invalid benchmark variant")
        if not isinstance(run_id, int) or run_id < 0:
            raise ValueError("invalid run id")
        if not isinstance(context_words, int) or context_words < 1:
            raise ValueError("invalid context size")
        cell = (mode, variant, run_id, context_words)
        if cell in seen:
            raise ValueError(f"duplicate process cell: {cell}")
        seen.add(cell)

        response = record.get("response")
        if not isinstance(response, dict):
            raise ValueError("missing response")
        elapsed = response.get("elapsed_seconds")
        completion_tokens = response.get("completion_tokens")
        token_ids = response.get("token_ids")
        text_sha256 = response.get("text_sha256")
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid elapsed time")
        if not isinstance(completion_tokens, int) or completion_tokens < 1:
            raise ValueError("invalid completion count")
        if not isinstance(token_ids, list) or len(token_ids) != completion_tokens:
            raise ValueError("token trace does not match completion count")
        if not isinstance(text_sha256, str) or len(text_sha256) != 64:
            raise ValueError("invalid response digest")

        if strict:
            pid = record.get("pid")
            if not isinstance(pid, int) or pid <= 0 or pid in seen_pids:
                raise ValueError("every benchmark cell must use a fresh process")
            seen_pids.add(pid)
            sources = record.get("source_sha256")
            if not isinstance(sources, dict) or not sources:
                raise ValueError("missing source provenance")
            normalized_sources = {str(name): str(digest) for name, digest in sources.items()}
            if any(len(digest) != 64 for digest in normalized_sources.values()):
                raise ValueError("invalid source provenance hash")
            if reference_sources is None:
                reference_sources = normalized_sources
            elif normalized_sources != reference_sources:
                raise ValueError("source provenance changed between processes")

            settings = record.get("settings")
            if not isinstance(settings, dict):
                raise ValueError("missing benchmark settings")
            normalized_settings = {str(key): value for key, value in settings.items()}
            variant_key = str(variant)
            if variant_key in settings_by_variant:
                if normalized_settings != settings_by_variant[variant_key]:
                    raise ValueError("benchmark settings changed within a variant")
            else:
                settings_by_variant[variant_key] = normalized_settings

            environment = record.get("environment")
            if not isinstance(environment, dict):
                raise ValueError("missing benchmark environment")
            stable_environment = {
                key: environment.get(key)
                for key in (
                    "platform",
                    "python",
                    "mlx",
                    "mlx_lm",
                    "numpy",
                    "psutil",
                    "git_commit",
                    "git_branch",
                )
            }
            if reference_environment is None:
                reference_environment = stable_environment
            elif stable_environment != reference_environment:
                raise ValueError("benchmark environment changed between processes")


def validate_manifest_membership(
    input_paths: list[Path],
    records: list[dict[str, object]],
    manifest_paths: list[Path],
) -> dict[str, object]:
    expected: dict[Path, dict[str, object]] = {}
    manifest_hashes: dict[str, str] = {}
    reference_inputs: dict[str, object] | None = None
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("invalid execution manifest")
        if manifest.get("benchmark_sha256") != _sha256(ARTIFACT_DIR / "benchmark.py"):
            raise ValueError("execution manifest benchmark hash is stale")
        if manifest.get("runner_sha256") != _sha256(ARTIFACT_DIR / "run_matrix.py"):
            raise ValueError("execution manifest runner hash is stale")
        config = Path(str(manifest.get("config", "")))
        model = Path(str(manifest.get("model", "")))
        if not config.is_file() or manifest.get("config_sha256") != _sha256(config):
            raise ValueError("execution manifest config provenance is stale")
        if not model.is_dir() or manifest.get("model_file_sha256") != _model_runtime_hashes(
            model
        ):
            raise ValueError("execution manifest model provenance is stale")
        input_provenance = {
            "config_sha256": manifest["config_sha256"],
            "model_file_sha256": manifest["model_file_sha256"],
        }
        if reference_inputs is None:
            reference_inputs = input_provenance
        elif input_provenance != reference_inputs:
            raise ValueError("execution manifest inputs changed between groups")
        cells = manifest.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("execution manifest has no cells")
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("exit_code") != 0:
                raise ValueError("execution manifest contains a failed cell")
            output = cell.get("output")
            output_digest = cell.get("output_sha256")
            if not isinstance(output, str) or not isinstance(output_digest, str):
                raise ValueError("execution manifest cell has no output digest")
            path = (manifest_path.parent / output).resolve()
            if not path.is_file() or output_digest != _sha256(path):
                raise ValueError("execution manifest output digest is stale")
            if path in expected:
                raise ValueError("duplicate output across execution manifests")
            expected[path] = cell
        manifest_hashes[str(manifest_path.resolve())] = _sha256(manifest_path)

    actual_paths = [path.resolve() for path in input_paths]
    if len(actual_paths) != len(set(actual_paths)) or set(actual_paths) != set(expected):
        raise ValueError("aggregate inputs do not match execution manifests")
    for path, record in zip(actual_paths, records, strict=True):
        cell = expected[path]
        for field in ("mode", "variant", "run_id", "context_words"):
            if record.get(field) != cell.get(field):
                raise ValueError(f"record does not match manifest field: {field}")
    return {
        "execution_manifest_sha256": manifest_hashes,
        "input_records": len(records),
        "fresh_processes": len({int(record["pid"]) for record in records}),
        "input_provenance": reference_inputs,
    }


def paired_percent_deltas(
    records: list[dict[str, object]], *, metric: str
) -> list[float]:
    pairs: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for record in records:
        response = record["response"]
        assert isinstance(response, dict)
        value = response.get(metric)
        if not isinstance(value, (float, int)):
            raise ValueError(f"missing numeric metric: {metric}")
        key = (
            str(record["mode"]),
            int(record["context_words"]),
            int(record["run_id"]),
        )
        pairs[key][str(record["variant"])] = float(value)

    deltas: list[float] = []
    for key in sorted(pairs):
        pair = pairs[key]
        if set(pair) != {"native", "direct"}:
            raise ValueError(f"unpaired process cell: {key}")
        baseline = pair["native"]
        if baseline == 0:
            raise ValueError("paired baseline must be non-zero")
        deltas.append((pair["direct"] / baseline - 1.0) * 100.0)
    return deltas


def _bootstrap_median_interval(values: list[float], *, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires paired values")
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(10_000)
    ]
    medians.sort()
    return medians[249], medians[9749]


def aggregate(
    records: list[dict[str, object]], *, strict: bool = False
) -> dict[str, object]:
    validate_records(records, strict=strict)
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[(str(record["mode"]), int(record["context_words"]))].append(record)

    output: dict[str, object] = {"schema_version": 1, "groups": {}}
    rendered_groups: dict[str, object] = {}
    for (mode, context_words), group in sorted(groups.items()):
        variants: dict[str, object] = {}
        for variant in ("native", "direct"):
            selected = [item for item in group if item["variant"] == variant]
            elapsed = [float(_response(item)["elapsed_seconds"]) for item in selected]
            generation = [float(_response(item)["generation_tps"]) for item in selected]
            peak = [float(_response(item)["mlx_peak_memory_bytes"]) for item in selected]
            variants[variant] = {
                "processes": len(selected),
                "elapsed_median_seconds": statistics.median(elapsed),
                "elapsed_process_p95_seconds": _percentile(elapsed, 0.95),
                "generation_tps_median": statistics.median(generation),
                "peak_memory_max_bytes": max(peak),
                "swap_delta_max_bytes": max(
                    float(_response(item).get("swap_after_bytes", 0))
                    - float(_response(item).get("swap_before_bytes", 0))
                    for item in selected
                ),
                "ttft_median_seconds": _timeline_median(selected, "ttft_s"),
                "prefill_wall_median_seconds": _timeline_median(
                    selected, "prefill_wall_s"
                ),
                "decode_duration_median_seconds": _timeline_median(
                    selected, "decode_duration_s"
                ),
            }

        elapsed_deltas = paired_percent_deltas(group, metric="elapsed_seconds")
        generation_deltas = paired_percent_deltas(group, metric="generation_tps")
        elapsed_interval = _bootstrap_median_interval(
            elapsed_deltas, seed=49_000 + context_words
        )
        generation_interval = _bootstrap_median_interval(
            generation_deltas, seed=49_100 + context_words
        )
        parity = _paired_parity(group)
        rendered_groups[f"{mode}:{context_words}"] = {
            "variants": variants,
            "paired_elapsed_delta_percent_median": statistics.median(elapsed_deltas),
            "paired_elapsed_deltas_percent": elapsed_deltas,
            "paired_elapsed_delta_percent_bootstrap95": list(elapsed_interval),
            "paired_generation_tps_delta_percent_median": statistics.median(
                generation_deltas
            ),
            "paired_generation_tps_deltas_percent": generation_deltas,
            "paired_generation_tps_delta_percent_bootstrap95": list(
                generation_interval
            ),
            "paired_token_and_text_parity": parity,
            "phase_totals_median_ms_by_variant": {
                variant: _phase_totals(
                    [item for item in group if item["variant"] == variant]
                )
                for variant in ("native", "direct")
            },
        }
    output["groups"] = rendered_groups
    return output


def _response(record: dict[str, object]) -> dict[str, object]:
    response = record["response"]
    assert isinstance(response, dict)
    return response


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    index = max(math.ceil(quantile * len(ordered)) - 1, 0)
    return ordered[index]


def _timeline_median(records: list[dict[str, object]], metric: str) -> float | None:
    values: list[float] = []
    for record in records:
        timelines = record.get("request_timeline")
        if not isinstance(timelines, list) or not timelines:
            continue
        timeline = timelines[-1]
        if not isinstance(timeline, dict):
            continue
        value = timeline.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return statistics.median(values) if values else None


def _paired_parity(records: list[dict[str, object]]) -> bool:
    pairs: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in records:
        pairs[int(record["run_id"])][str(record["variant"])] = _response(record)
    for pair in pairs.values():
        if set(pair) != {"native", "direct"}:
            return False
        if pair["native"]["token_ids"] != pair["direct"]["token_ids"]:
            return False
        if pair["native"]["text_sha256"] != pair["direct"]["text_sha256"]:
            return False
    return True


def _phase_totals(records: list[dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        timings = record.get("timings")
        if not isinstance(timings, dict):
            continue
        for name, summary in timings.items():
            if isinstance(summary, dict) and isinstance(summary.get("total_ms"), (int, float)):
                grouped[str(name)].append(float(summary["total_ms"]))
    return {name: statistics.median(values) for name, values in sorted(grouped.items())}


def _load(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        payload: Any = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"record must be an object: {path}")
        records.append(payload)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    args = parser.parse_args()

    records = _load(args.inputs)
    manifest_evidence = (
        validate_manifest_membership(args.inputs, records, args.manifest)
        if args.manifest
        else None
    )
    result = aggregate(records, strict=bool(args.manifest))
    if manifest_evidence is not None:
        result["evidence"] = {
            **manifest_evidence,
            "source_sha256": records[0]["source_sha256"],
            "aggregator_sha256": _sha256(Path(__file__)),
        }
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
