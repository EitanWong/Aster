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

POLICIES = (
    "baseline",
    "skip-eval-clear-each",
    "periodic-256",
    "periodic-512",
    "periodic-2048",
    "skip-eval-no-clear",
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


def _rotated(values: tuple[str, ...], offset: int) -> tuple[str, ...]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def _output_name(*, context_words: int, run_id: int, policy: str) -> str:
    return f"native-b1-{context_words}w-run-{run_id}-{policy}.json"


def _display_path(path: Path) -> str:
    return str(path.relative_to(ARTIFACT_DIR)) if path.is_relative_to(ARTIFACT_DIR) else str(path)


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("*.json"))
    if existing and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} already contains JSON outputs; use --resume or a new directory"
        )

    expected_hashes = _source_hashes(args.config, args.model)
    records: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for run_id in range(1, args.runs + 1):
        for context_index, context_words in enumerate(args.contexts):
            order = _rotated(POLICIES, run_id + context_index)
            for policy in order:
                output = args.output_dir / _output_name(
                    context_words=context_words,
                    run_id=run_id,
                    policy=policy,
                )
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
                    "native",
                    "--batch-size",
                    "1",
                    "--context-words",
                    str(context_words),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--warmup-tokens",
                    str(args.warmup_tokens),
                    "--run-id",
                    str(run_id),
                    "--output",
                    str(output),
                ]
                if output.is_file() and args.resume:
                    payload = json.loads(output.read_text())
                    records.append(
                        {
                            "output": _display_path(output),
                            "sha256": _sha256(output),
                            "pid": int(payload["pid"]),
                            "elapsed_seconds": None,
                            "command": command,
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
                observed_hashes = payload["source_sha256"] | payload["model_input_sha256"]
                if any(expected_hashes.get(path) != digest for path, digest in observed_hashes.items()):
                    raise RuntimeError(f"source hash mismatch in {output}")
                records.append(
                    {
                        "output": _display_path(output),
                        "sha256": _sha256(output),
                        "pid": int(payload["pid"]),
                        "elapsed_seconds": elapsed,
                        "command": command,
                        "resumed": False,
                    }
                )
                print(
                    f"run={run_id} context={context_words} policy={policy} "
                    f"tps={payload['decode']['tokens_per_second']:.3f} "
                    f"rss={payload['memory']['rss_peak_bytes']}"
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
            "contexts": list(args.contexts),
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "cache_kind": "native",
            "batch_size": 1,
            "fresh_processes": True,
            "order": "rotated per run and context",
        },
        "source_sha256": expected_hashes,
        "records": records,
        "wall_seconds": time.perf_counter() - started_all,
    }
    manifest_path = args.output_dir / "execution-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
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
        default=ARTIFACT_DIR / "results/screen",
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=(128, 2048))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.runs < 1 or args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("runs and token counts must be positive")
    if any(context < 1 for context in args.contexts):
        raise ValueError("contexts must be positive")
    if not PYTHON.is_file() or not BENCHMARK.is_file():
        raise FileNotFoundError("benchmark runtime is incomplete")

    manifest = run(args)
    print(json.dumps({"records": len(manifest["records"]), "wall_seconds": manifest["wall_seconds"]}))


if __name__ == "__main__":
    main()
