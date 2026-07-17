#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BENCHMARK = ARTIFACT_DIR / "benchmark.py"
PYTHON = PROJECT_ROOT / ".venv/bin/python"
POLICIES = ("baseline", "periodic-512")
CELLS = (
    {"cache_kind": "native", "batch_size": 1, "context_words": 128, "max_tokens": 10_000},
    {"cache_kind": "direct", "batch_size": 1, "context_words": 128, "max_tokens": 10_000},
    {"cache_kind": "native", "batch_size": 4, "context_words": 128, "max_tokens": 4_096},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _source_hashes(config: Path, model: Path) -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
        PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
        PROJECT_ROOT / "aster/inference/paged_attention_bridge.py",
        PROJECT_ROOT / "aster/inference/metal_paged_attention.py",
        config,
        model / "model.safetensors",
        model / "model.safetensors.index.json",
        model / "config.json",
        model / "tokenizer.json",
        model / "tokenizer_config.json",
        model / "chat_template.jinja",
        BENCHMARK,
        Path(__file__).resolve(),
    )
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in paths
        if path.is_file()
    }


def _name(cell: dict[str, Any], policy: str) -> str:
    return (
        f"{cell['cache_kind']}-b{cell['batch_size']}-{cell['context_words']}w-"
        f"{cell['max_tokens']}t-{policy}.json"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("*.json")) and not args.resume:
        raise FileExistsError(f"{args.output_dir} already contains JSON files")

    expected_hashes = _source_hashes(args.config, args.model)
    records: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for cell_index, cell in enumerate(CELLS):
        order = POLICIES if cell_index % 2 == 0 else tuple(reversed(POLICIES))
        for policy in order:
            output = args.output_dir / _name(cell, policy)
            command = [
                str(PYTHON),
                str(BENCHMARK),
                "--config",
                str(args.config),
                "--model",
                str(args.model),
                "--policy",
                policy,
                "--cache-kind",
                str(cell["cache_kind"]),
                "--batch-size",
                str(cell["batch_size"]),
                "--context-words",
                str(cell["context_words"]),
                "--max-tokens",
                str(cell["max_tokens"]),
                "--warmup-tokens",
                str(args.warmup_tokens),
                "--memory-sample-interval",
                str(args.memory_sample_interval),
                "--run-id",
                "1",
                "--output",
                str(output),
            ]
            if output.is_file() and args.resume:
                payload = json.loads(output.read_text())
                records.append(
                    {
                        "output": str(output.relative_to(ARTIFACT_DIR)),
                        "sha256": _sha256(output),
                        "pid": int(payload["pid"]),
                        "command": command,
                        "elapsed_seconds": None,
                        "resumed": True,
                    }
                )
                continue

            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark failed ({completed.returncode}): {' '.join(command)}\n"
                    f"{completed.stderr}"
                )
            payload = json.loads(output.read_text())
            observed = payload["source_sha256"] | payload["model_input_sha256"]
            if any(expected_hashes.get(path) != digest for path, digest in observed.items()):
                raise RuntimeError(f"source hash mismatch in {output}")
            records.append(
                {
                    "output": str(output.relative_to(ARTIFACT_DIR)),
                    "sha256": _sha256(output),
                    "pid": int(payload["pid"]),
                    "command": command,
                    "elapsed_seconds": elapsed,
                    "resumed": False,
                }
            )
            print(
                f"cache={cell['cache_kind']} batch={cell['batch_size']} "
                f"tokens={cell['max_tokens']} policy={policy} "
                f"tps={payload['decode']['tokens_per_second']:.3f} "
                f"rss={payload['memory']['rss_peak_bytes']}",
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "matrix": {
            "policies": list(POLICIES),
            "cells": list(CELLS),
            "warmup_tokens": args.warmup_tokens,
            "memory_sample_interval": args.memory_sample_interval,
            "fresh_processes": True,
            "order": "alternating adjacent A/B pairs",
        },
        "source_sha256": expected_hashes,
        "records": records,
        "wall_seconds": time.perf_counter() - started_all,
    }
    path = args.output_dir / "execution-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR / "results/long-stress",
    )
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--memory-sample-interval", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.warmup_tokens < 1 or args.memory_sample_interval < 1:
        raise ValueError("warmup tokens and memory sample interval must be positive")
    manifest = run(args)
    print(json.dumps({"records": len(manifest["records"]), "wall_seconds": manifest["wall_seconds"]}))


if __name__ == "__main__":
    main()
