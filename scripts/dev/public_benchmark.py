#!/usr/bin/env python3
"""Manage version-pinned public data for Aster cross-engine benchmarks.

The tool deliberately keeps downloaded benchmark data under ``run/``.  Git
tracks the small source lock while the data, derived workload manifests, and
engine results remain reproducible local inputs rather than repository bloat.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import statistics
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "run/loop-engineering/public-benchmarks"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "docs/loop-engineering/benchmarks/public-dataset-lock.json"
USER_AGENT = "Aster-public-benchmark/1.0"

PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "mt-bench": {"include_mt_bench": True, "longbench_tasks": ()},
    "cross-engine-core": {
        "include_mt_bench": True,
        "longbench_tasks": ("qasper", "2wikimqa", "qmsum", "gov_report", "lcc"),
    },
    "full-public": {"include_mt_bench": True, "longbench_tasks": None},
}

REQUIRED_METRICS = (
    "ttft_seconds",
    "end_to_end_seconds",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "peak_rss_bytes",
    "swap_delta_bytes",
)

ENGINE_PROBES: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "aster": ("aster", ()),
    "mlx-lm": ("mlx_lm", ()),
    "ollama": (None, ("ollama",)),
    "llama.cpp": (None, ("llama-cli", "llama-server")),
    "vllm": ("vllm", ("vllm",)),
    "sglang": ("sglang", ("sglang",)),
    "vllm-mlx": ("vllm_mlx", ()),
    "mlc-llm": ("mlc_llm", ("mlc_llm",)),
    "mistral.rs": ("mistralrs", ("mistralrs", "mistralrs-server")),
    "lmstudio-mlx-engine": ("mlx_engine", ("mlx-engine",)),
    "omlx": ("omlx", ("omlx",)),
    "exo": ("exo", ("exo",)),
}


class PublicBenchmarkError(ValueError):
    """Raised when a benchmark source, workload, or result is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PublicBenchmarkError(f"cannot load source lock {path}: {error}") from error
    if payload.get("schema_version") != 1:
        raise PublicBenchmarkError("public benchmark lock must use schema_version 1")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PublicBenchmarkError("public benchmark lock must declare sources")
    source_ids: set[str] = set()
    local_paths: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise PublicBenchmarkError("public benchmark source must be an object")
        for key in ("id", "url", "local_path", "sha256", "size_bytes", "validator"):
            if key not in source:
                raise PublicBenchmarkError(f"source is missing required key: {key}")
        source_id = source["id"]
        local_path = source["local_path"]
        if not isinstance(source_id, str) or source_id in source_ids:
            raise PublicBenchmarkError("source ids must be unique strings")
        if not isinstance(local_path, str) or local_path in local_paths:
            raise PublicBenchmarkError("source local paths must be unique strings")
        if not isinstance(source["sha256"], str) or len(source["sha256"]) != 64:
            raise PublicBenchmarkError(f"source {source_id} has an invalid sha256")
        if not isinstance(source["size_bytes"], int) or source["size_bytes"] < 1:
            raise PublicBenchmarkError(f"source {source_id} has an invalid size")
        source_ids.add(source_id)
        local_paths.add(local_path)
    return payload


def sources_by_id(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(source["id"]): source for source in lock["sources"]}


def source_path(data_root: Path, source: dict[str, Any]) -> Path:
    path = (data_root / str(source["local_path"])).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise PublicBenchmarkError(f"source path escapes data root: {source['id']}")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise PublicBenchmarkError(
                    f"invalid JSONL in {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise PublicBenchmarkError(f"JSONL row {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def validate_jsonl_source(source: dict[str, Any], path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    validator = source["validator"]
    expected_count = validator.get("record_count")
    if len(rows) != expected_count:
        raise PublicBenchmarkError(
            f"{source['id']} record count {len(rows)} does not match locked {expected_count}"
        )
    required_keys = tuple(validator.get("required_keys", ()))
    for row in rows:
        missing = [key for key in required_keys if key not in row]
        if missing:
            raise PublicBenchmarkError(
                f"{source['id']} record is missing keys: {', '.join(missing)}"
            )
    unique_key = validator.get("unique_key")
    if unique_key:
        values = [row.get(unique_key) for row in rows]
        if None in values or len(set(values)) != len(values):
            raise PublicBenchmarkError(f"{source['id']} has non-unique {unique_key} values")
    return {"record_count": len(rows)}


def validate_json_mapping_source(source: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PublicBenchmarkError(f"cannot parse {source['id']}: {error}") from error
    if not isinstance(payload, dict):
        raise PublicBenchmarkError(f"{source['id']} must be a JSON object")
    expected_count = source["validator"].get("key_count")
    if len(payload) != expected_count:
        raise PublicBenchmarkError(
            f"{source['id']} key count {len(payload)} does not match locked {expected_count}"
        )
    return {"key_count": len(payload)}


def longbench_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.endswith(".jsonl"):
            continue
        dataset = Path(info.filename).stem
        if dataset in members:
            raise PublicBenchmarkError(f"LongBench archive has duplicate dataset member: {dataset}")
        members[dataset] = info.filename
    return members


def validate_longbench_archive(source: dict[str, Any], path: Path) -> dict[str, Any]:
    validator = source["validator"]
    try:
        with zipfile.ZipFile(path) as archive:
            broken_member = archive.testzip()
            if broken_member is not None:
                raise PublicBenchmarkError(f"LongBench archive CRC failure: {broken_member}")
            members = longbench_members(archive)
            parsed_records = 0
            for member in members.values():
                with archive.open(member) as binary_handle:
                    for line_number, raw_line in enumerate(binary_handle, start=1):
                        if not raw_line.strip():
                            continue
                        try:
                            row = json.loads(raw_line)
                        except json.JSONDecodeError as error:
                            raise PublicBenchmarkError(
                                f"invalid LongBench JSONL in {member}:{line_number}: {error}"
                            ) from error
                        if not isinstance(row, dict):
                            raise PublicBenchmarkError(
                                f"LongBench JSONL row {member}:{line_number} must be an object"
                            )
                        parsed_records += 1
    except (OSError, zipfile.BadZipFile) as error:
        raise PublicBenchmarkError(f"cannot read LongBench archive {path}: {error}") from error
    expected_tasks = validator.get("task_count")
    expected_records = validator.get("record_count")
    if len(members) != expected_tasks:
        raise PublicBenchmarkError(
            f"LongBench task count {len(members)} does not match locked {expected_tasks}"
        )
    if parsed_records != expected_records:
        raise PublicBenchmarkError(
            f"LongBench record count {parsed_records} does not match locked {expected_records}"
        )
    primary_members = {name: member for name, member in members.items() if not name.endswith("_e")}
    primary_records = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for member in primary_members.values():
                with archive.open(member) as binary_handle:
                    primary_records += sum(1 for raw_line in binary_handle if raw_line.strip())
    except (OSError, zipfile.BadZipFile) as error:
        raise PublicBenchmarkError(f"cannot re-read LongBench archive {path}: {error}") from error
    expected_primary_tasks = validator.get("primary_task_count")
    expected_primary_records = validator.get("primary_record_count")
    if expected_primary_tasks is not None and len(primary_members) != expected_primary_tasks:
        raise PublicBenchmarkError(
            "LongBench primary task count "
            + f"{len(primary_members)} does not match locked {expected_primary_tasks}"
        )
    if expected_primary_records is not None and primary_records != expected_primary_records:
        raise PublicBenchmarkError(
            "LongBench primary record count "
            + f"{primary_records} does not match locked {expected_primary_records}"
        )
    return {
        "task_count": len(members),
        "record_count": parsed_records,
        "primary_task_count": len(primary_members),
        "primary_record_count": primary_records,
    }


def validate_source(source: dict[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PublicBenchmarkError(f"missing public source: {path}")
    size_bytes = path.stat().st_size
    if size_bytes != source["size_bytes"]:
        raise PublicBenchmarkError(
            f"{source['id']} size {size_bytes} does not match locked {source['size_bytes']}"
        )
    digest = sha256_file(path)
    if digest != source["sha256"]:
        raise PublicBenchmarkError(f"{source['id']} sha256 does not match the source lock")
    kind = source["validator"].get("kind")
    if kind == "jsonl":
        details = validate_jsonl_source(source, path)
    elif kind == "json_mapping":
        details = validate_json_mapping_source(source, path)
    elif kind == "longbench_zip":
        details = validate_longbench_archive(source, path)
    else:
        raise PublicBenchmarkError(f"unsupported validator kind for {source['id']}: {kind}")
    return {"path": str(path), "sha256": digest, "size_bytes": size_bytes, **details}


def verify_install(
    lock: dict[str, Any], data_root: Path, lock_path: Path = DEFAULT_LOCK_PATH
) -> dict[str, Any]:
    source_results = {
        str(source["id"]): validate_source(source, source_path(data_root, source))
        for source in lock["sources"]
    }
    return {
        "schema_version": 1,
        "decision": "verified",
        "data_root": str(data_root),
        "lock_sha256": sha256_file(lock_path),
        "sources": source_results,
    }


def download_source(source: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise PublicBenchmarkError(
            f"partial download exists at {partial}; inspect it before retrying sync"
        )
    request = Request(str(source["url"]), headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response, partial.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    except OSError as error:
        raise PublicBenchmarkError(f"download failed for {source['id']}: {error}") from error
    try:
        validate_source(source, partial)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)


def sync_install(lock: dict[str, Any], lock_path: Path, data_root: Path) -> dict[str, Any]:
    downloaded: list[str] = []
    for source in lock["sources"]:
        destination = source_path(data_root, source)
        if destination.exists():
            validate_source(source, destination)
            continue
        download_source(source, destination)
        downloaded.append(str(source["id"]))
    verification = {
        "schema_version": 1,
        "decision": "verified",
        "data_root": str(data_root),
        "lock_sha256": sha256_file(lock_path),
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "downloaded_sources": downloaded,
        "sources": {
            str(source["id"]): validate_source(source, source_path(data_root, source))
            for source in lock["sources"]
        },
    }
    write_json(data_root / "install-manifest.json", verification)
    return verification


def load_json_mapping(source: dict[str, Any], data_root: Path) -> dict[str, Any]:
    path = source_path(data_root, source)
    validate_source(source, path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise PublicBenchmarkError(f"{source['id']} is not a mapping")
    return payload


class PublicWorkloadResolver:
    """Resolve a manifest record back to its locked public prompt.

    Workload manifests intentionally store hashes instead of prompt text.  The
    resolver reopens the pinned source, checks the source-row and template
    hashes, then renders the exact public prompt at execution time.
    """

    def __init__(self, lock: dict[str, Any], data_root: Path) -> None:
        self._sources = sources_by_id(lock)
        self._data_root = data_root
        self._validated_sources: set[str] = set()
        self._jsonl_rows: dict[str, list[dict[str, Any]]] = {}
        self._mappings: dict[str, dict[str, Any]] = {}
        self._longbench_members: dict[str, str] | None = None
        self._longbench_rows: dict[str, dict[int, dict[str, Any]]] = {}

    def _source(self, source_id: str) -> dict[str, Any]:
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise PublicBenchmarkError(
                f"workload references an unpinned public source: {source_id}"
            ) from error

    def _validate(self, source_id: str) -> dict[str, Any]:
        source = self._source(source_id)
        if source_id not in self._validated_sources:
            validate_source(source, source_path(self._data_root, source))
            self._validated_sources.add(source_id)
        return source

    def _mapping(self, source_id: str) -> dict[str, Any]:
        cached = self._mappings.get(source_id)
        if cached is not None:
            return cached
        source = self._validate(source_id)
        path = source_path(self._data_root, source)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise PublicBenchmarkError(
                f"cannot parse source mapping {source_id}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise PublicBenchmarkError(f"source mapping {source_id} is not an object")
        self._mappings[source_id] = payload
        return payload

    def _mt_bench_row(self, source_id: str, question_id: int) -> dict[str, Any]:
        rows = self._jsonl_rows.get(source_id)
        if rows is None:
            source = self._validate(source_id)
            rows = read_jsonl(source_path(self._data_root, source))
            self._jsonl_rows[source_id] = rows
        for row in rows:
            if int(row["question_id"]) == question_id:
                return row
        raise PublicBenchmarkError(
            f"MT-Bench question is absent from the pinned source: {question_id}"
        )

    def _longbench_row(
        self, data_source_id: str, dataset: str, record_index: int
    ) -> dict[str, Any]:
        cached = self._longbench_rows.get(dataset)
        if cached is None:
            source = self._validate(data_source_id)
            path = source_path(self._data_root, source)
            try:
                with zipfile.ZipFile(path) as archive:
                    if self._longbench_members is None:
                        self._longbench_members = longbench_members(archive)
                    member = self._longbench_members.get(dataset)
                    if member is None:
                        raise PublicBenchmarkError(
                            f"LongBench dataset is absent from the pinned archive: {dataset}"
                        )
                    cached = {}
                    with archive.open(member) as binary_handle:
                        for index, raw_line in enumerate(binary_handle):
                            if not raw_line.strip():
                                continue
                            row = json.loads(raw_line)
                            if not isinstance(row, dict):
                                raise PublicBenchmarkError(
                                    f"LongBench row {dataset}:{index} is not an object"
                                )
                            cached[index] = row
            except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
                raise PublicBenchmarkError(
                    f"cannot read LongBench dataset {dataset}: {error}"
                ) from error
            self._longbench_rows[dataset] = cached
        try:
            return cached[record_index]
        except KeyError as error:
            raise PublicBenchmarkError(
                f"LongBench record is absent from the pinned source: {dataset}:{record_index}"
            ) from error

    @staticmethod
    def _expect_mapping(value: Any, field: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise PublicBenchmarkError(f"workload record has no {field} mapping")
        return value

    @staticmethod
    def _expect_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise PublicBenchmarkError(f"workload record has invalid {field}")
        return value

    def _verify_prompt(self, record: dict[str, Any], prompt: str) -> str:
        descriptor = self._expect_mapping(record.get("prompt"), "prompt")
        expected_sha256 = self._expect_string(descriptor.get("sha256"), "prompt sha256")
        if sha256_text(prompt) != expected_sha256:
            raise PublicBenchmarkError(
                f"public prompt hash does not match workload record {record.get('workload_id')}"
            )
        expected_characters = descriptor.get("characters")
        if expected_characters != len(prompt):
            raise PublicBenchmarkError(
                f"public prompt length does not match workload record {record.get('workload_id')}"
            )
        return prompt

    def resolve(self, record: dict[str, Any]) -> str:
        source_descriptor = self._expect_mapping(record.get("source"), "source")
        source_id = self._expect_string(source_descriptor.get("id"), "source id")
        prompt_descriptor = self._expect_mapping(record.get("prompt"), "prompt")
        renderer = self._expect_string(prompt_descriptor.get("renderer"), "prompt renderer")

        if source_id == "mt-bench-question":
            if renderer != "public-verbatim-mt-bench-turn":
                raise PublicBenchmarkError(f"unexpected MT-Bench renderer: {renderer}")
            question_id = source_descriptor.get("question_id")
            turn_index = source_descriptor.get("turn_index")
            if not isinstance(question_id, int) or not isinstance(turn_index, int):
                raise PublicBenchmarkError(
                    "MT-Bench workload record has invalid question or turn index"
                )
            row = self._mt_bench_row(source_id, question_id)
            if canonical_json_sha256(row) != source_descriptor.get("record_sha256"):
                raise PublicBenchmarkError(
                    f"MT-Bench source row drifted for question {question_id}"
                )
            turns = row.get("turns")
            if not isinstance(turns, list) or not 0 <= turn_index < len(turns):
                raise PublicBenchmarkError(
                    f"MT-Bench turn is unavailable: {question_id}:{turn_index}"
                )
            prompt = turns[turn_index]
            if not isinstance(prompt, str):
                raise PublicBenchmarkError(f"MT-Bench turn is not text: {question_id}:{turn_index}")
            return self._verify_prompt(record, prompt)

        if source_id == "longbench-v1-data":
            if renderer != "official-longbench-v1-template":
                raise PublicBenchmarkError(f"unexpected LongBench renderer: {renderer}")
            dataset = self._expect_string(source_descriptor.get("dataset"), "LongBench dataset")
            record_index = source_descriptor.get("record_index")
            if not isinstance(record_index, int) or record_index < 0:
                raise PublicBenchmarkError("LongBench workload record has invalid record index")
            template_source_id = self._expect_string(
                prompt_descriptor.get("template_source"), "LongBench template source"
            )
            template_sha256 = self._expect_string(
                prompt_descriptor.get("template_sha256"), "LongBench template sha256"
            )
            row = self._longbench_row(source_id, dataset, record_index)
            if canonical_json_sha256(row) != source_descriptor.get("record_sha256"):
                raise PublicBenchmarkError(
                    f"LongBench source row drifted for {dataset}:{record_index}"
                )
            templates = self._mapping(template_source_id)
            template = templates.get(dataset)
            if not isinstance(template, str):
                raise PublicBenchmarkError(f"LongBench template is unavailable: {dataset}")
            if sha256_text(template) != template_sha256:
                raise PublicBenchmarkError(f"LongBench template drifted for {dataset}")
            try:
                prompt = template.format(**row)
            except (KeyError, ValueError) as error:
                raise PublicBenchmarkError(
                    f"official LongBench template failed for {dataset}:{record_index}: {error}"
                ) from error
            return self._verify_prompt(record, prompt)

        raise PublicBenchmarkError(f"unsupported public workload source: {source_id}")


def resolve_workload_prompt(record: dict[str, Any], lock: dict[str, Any], data_root: Path) -> str:
    """Resolve one public workload record without copying its prompt into the manifest."""

    return PublicWorkloadResolver(lock, data_root).resolve(record)


def mt_bench_records(source: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    path = source_path(data_root, source)
    validate_source(source, path)
    records: list[dict[str, Any]] = []
    for row in sorted(read_jsonl(path), key=lambda item: int(item["question_id"])):
        turns = row["turns"]
        if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
            raise PublicBenchmarkError(
                f"MT-Bench question {row['question_id']} has no usable first turn"
            )
        prompt = turns[0]
        question_id = int(row["question_id"])
        records.append(
            {
                "workload_id": f"mt-bench:{question_id}:turn-1",
                "source": {
                    "id": source["id"],
                    "question_id": question_id,
                    "turn_index": 0,
                    "record_sha256": canonical_json_sha256(row),
                },
                "scenario": {"family": "interactive", "category": str(row["category"])},
                "prompt": {
                    "renderer": "public-verbatim-mt-bench-turn",
                    "sha256": sha256_text(prompt),
                    "characters": len(prompt),
                },
                "max_tokens": 256,
            }
        )
    return records


def longbench_records(
    data_source: dict[str, Any],
    prompt_source: dict[str, Any],
    max_output_source: dict[str, Any],
    data_root: Path,
    selected_tasks: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    data_path = source_path(data_root, data_source)
    validate_source(data_source, data_path)
    templates = load_json_mapping(prompt_source, data_root)
    max_outputs = load_json_mapping(max_output_source, data_root)
    if set(templates) != set(max_outputs):
        raise PublicBenchmarkError("LongBench prompt and maximum-output mappings differ")
    task_names = tuple(templates) if selected_tasks is None else selected_tasks
    unknown_tasks = sorted(set(task_names) - set(templates))
    if unknown_tasks:
        raise PublicBenchmarkError("unknown LongBench tasks: " + ", ".join(unknown_tasks))
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(data_path) as archive:
        members = longbench_members(archive)
        missing_members = sorted(set(task_names) - set(members))
        if missing_members:
            raise PublicBenchmarkError(
                "LongBench archive misses tasks: " + ", ".join(missing_members)
            )
        for dataset in task_names:
            template = templates[dataset]
            if not isinstance(template, str):
                raise PublicBenchmarkError(f"LongBench template for {dataset} is not text")
            try:
                max_tokens = int(max_outputs[dataset])
            except (TypeError, ValueError) as error:
                raise PublicBenchmarkError(
                    f"LongBench max output for {dataset} is invalid"
                ) from error
            template_sha256 = sha256_text(template)
            with archive.open(members[dataset]) as binary_handle:
                for record_index, raw_line in enumerate(binary_handle):
                    if not raw_line.strip():
                        continue
                    row = json.loads(raw_line)
                    if not isinstance(row, dict):
                        raise PublicBenchmarkError(
                            f"LongBench row {dataset}:{record_index} is not an object"
                        )
                    try:
                        prompt = template.format(**row)
                    except (KeyError, ValueError) as error:
                        raise PublicBenchmarkError(
                            f"official LongBench template failed for {dataset}:{record_index}: {error}"
                        ) from error
                    source_record_id = str(row.get("_id", record_index))
                    records.append(
                        {
                            "workload_id": f"longbench:{dataset}:{source_record_id}",
                            "source": {
                                "id": data_source["id"],
                                "dataset": dataset,
                                "record_index": record_index,
                                "record_sha256": canonical_json_sha256(row),
                            },
                            "scenario": {
                                "family": "long-context",
                                "dataset": dataset,
                                "language": row.get("language"),
                                "reported_length": row.get("length"),
                            },
                            "prompt": {
                                "renderer": "official-longbench-v1-template",
                                "template_source": prompt_source["id"],
                                "template_sha256": template_sha256,
                                "sha256": sha256_text(prompt),
                                "characters": len(prompt),
                            },
                            "max_tokens": max_tokens,
                        }
                    )
    return records


def limit_records(
    records: list[dict[str, Any]], limit_per_stratum: int | None
) -> list[dict[str, Any]]:
    if limit_per_stratum is None:
        return records
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for record in records:
        source = record["source"]
        stratum = str(source.get("dataset", source["id"]))
        if counts[stratum] >= limit_per_stratum:
            continue
        counts[stratum] += 1
        selected.append(record)
    return selected


def build_workload(
    lock: dict[str, Any],
    lock_path: Path,
    data_root: Path,
    profile: str,
    limit_per_stratum: int | None,
) -> dict[str, Any]:
    if profile not in PROFILE_DEFINITIONS:
        raise PublicBenchmarkError(f"unknown profile: {profile}")
    if limit_per_stratum is not None and limit_per_stratum < 1:
        raise PublicBenchmarkError("limit per stratum must be positive")
    sources = sources_by_id(lock)
    required_sources = {
        "mt-bench-question",
        "longbench-v1-data",
        "longbench-v1-prompts",
        "longbench-v1-max-output",
    }
    missing_sources = sorted(required_sources - set(sources))
    if missing_sources:
        raise PublicBenchmarkError(
            "source lock misses required sources: " + ", ".join(missing_sources)
        )
    definition = PROFILE_DEFINITIONS[profile]
    records: list[dict[str, Any]] = []
    if definition["include_mt_bench"]:
        records.extend(mt_bench_records(sources["mt-bench-question"], data_root))
    longbench_tasks = definition["longbench_tasks"]
    if longbench_tasks != ():
        records.extend(
            longbench_records(
                sources["longbench-v1-data"],
                sources["longbench-v1-prompts"],
                sources["longbench-v1-max-output"],
                data_root,
                longbench_tasks,
            )
        )
    records = limit_records(records, limit_per_stratum)
    workload_ids = [record["workload_id"] for record in records]
    if not records or len(set(workload_ids)) != len(workload_ids):
        raise PublicBenchmarkError("workload must contain unique public-source records")
    full_public = profile == "full-public" and limit_per_stratum is None
    return {
        "schema_version": 1,
        "kind": "public-cross-engine-workload",
        "created_utc": datetime.now(UTC).isoformat(),
        "lock_sha256": sha256_file(lock_path),
        "data_root": str(data_root),
        "profile": profile,
        "selection": {
            "origin": "public-dataset-only",
            "limit_per_stratum": limit_per_stratum,
            "global_cross_engine_claim_eligible": full_public,
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "seed": 0,
        },
        "records": records,
    }


def probe_engine(name: str) -> dict[str, Any]:
    try:
        module, commands = ENGINE_PROBES[name]
    except KeyError as error:
        raise PublicBenchmarkError(f"unknown engine probe: {name}") from error
    if module and importlib.util.find_spec(module) is not None:
        return {"status": "available", "kind": "python-module", "detail": module}
    command = next((candidate for candidate in commands if shutil.which(candidate)), None)
    if command:
        return {"status": "available", "kind": "command", "detail": command}
    probes = list(dict.fromkeys(([module] if module else []) + list(commands)))
    return {"status": "unavailable", "kind": "probe", "detail": ", ".join(probes)}


def inventory_engines() -> dict[str, Any]:
    engines = {name: probe_engine(name) for name in ENGINE_PROBES}
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "engines": engines,
        "available": sorted(
            name for name, result in engines.items() if result["status"] == "available"
        ),
        "unavailable": sorted(
            name for name, result in engines.items() if result["status"] != "available"
        ),
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise PublicBenchmarkError("cannot calculate a percentile of an empty series")
    index = max(0, min(len(values) - 1, int((len(values) - 1) * quantile + 0.999999)))
    return sorted(values)[index]


def validate_engine_results(
    workload_path: Path,
    result_paths: list[Path],
    required_engines: set[str],
) -> dict[str, Any]:
    workload = json.loads(workload_path.read_text())
    if workload.get("kind") != "public-cross-engine-workload":
        raise PublicBenchmarkError("result validation requires a public-cross-engine-workload")
    expected_records = {record["workload_id"]: record for record in workload["records"]}
    workload_digest = sha256_file(workload_path)
    errors: list[str] = []
    engines: dict[str, dict[str, Any]] = {}
    fingerprints: set[tuple[str, str]] = set()
    output_hashes: dict[str, set[str]] = {record_id: set() for record_id in expected_records}
    input_token_hashes: dict[str, set[str]] = {record_id: set() for record_id in expected_records}
    execution_contracts: set[str] = set()
    for path in result_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot load {path}: {error}")
            continue
        engine = payload.get("engine")
        if not isinstance(engine, str) or not engine:
            errors.append(f"{path} has no engine name")
            continue
        if engine in engines:
            errors.append(f"duplicate result file for engine {engine}")
            continue
        engines[engine] = payload
        if payload.get("workload_sha256") != workload_digest:
            errors.append(f"{engine} workload hash differs from {workload_path.name}")
        if payload.get("generation") != workload["generation"]:
            errors.append(f"{engine} generation settings differ from workload")
        execution = payload.get("execution")
        if not isinstance(execution, dict):
            errors.append(f"{engine} has no execution contract")
        else:
            execution_contracts.add(canonical_json_sha256(execution))
        fingerprint = payload.get("model_fingerprint")
        if not isinstance(fingerprint, dict):
            errors.append(f"{engine} has no model fingerprint")
        else:
            model_sha256 = fingerprint.get("model_sha256")
            tokenizer_sha256 = fingerprint.get("tokenizer_sha256")
            if not isinstance(model_sha256, str) or not isinstance(tokenizer_sha256, str):
                errors.append(f"{engine} has an incomplete model fingerprint")
            else:
                fingerprints.add((model_sha256, tokenizer_sha256))
        rows = payload.get("records")
        if not isinstance(rows, list):
            errors.append(f"{engine} has no result records")
            continue
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("workload_id"), str):
                errors.append(f"{engine} has a malformed result record")
                continue
            workload_id = row["workload_id"]
            if workload_id in rows_by_id:
                errors.append(f"{engine} has duplicate workload record {workload_id}")
                continue
            rows_by_id[workload_id] = row
        unexpected = sorted(set(rows_by_id) - set(expected_records))
        missing = sorted(set(expected_records) - set(rows_by_id))
        if unexpected:
            errors.append(f"{engine} has unexpected workload records: {', '.join(unexpected[:3])}")
        if missing:
            errors.append(f"{engine} misses {len(missing)} workload records")
        for workload_id, expected in expected_records.items():
            row = rows_by_id.get(workload_id)
            if row is None:
                continue
            if row.get("prompt_sha256") != expected["prompt"]["sha256"]:
                errors.append(f"{engine} prompt hash differs for {workload_id}")
            input_token_hash = row.get("prompt_token_ids_sha256")
            if not isinstance(input_token_hash, str) or len(input_token_hash) != 64:
                errors.append(f"{engine} input token hash is missing for {workload_id}")
            else:
                input_token_hashes[workload_id].add(input_token_hash)
            input_token_count = row.get("prompt_token_count")
            if not isinstance(input_token_count, int) or isinstance(input_token_count, bool):
                errors.append(f"{engine} input token count is invalid for {workload_id}")
            elif input_token_count < 1:
                errors.append(f"{engine} input token count is non-positive for {workload_id}")
            output_hash = row.get("output_token_ids_sha256")
            if not isinstance(output_hash, str) or len(output_hash) != 64:
                errors.append(f"{engine} output token hash is missing for {workload_id}")
            else:
                output_hashes[workload_id].add(output_hash)
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                errors.append(f"{engine} metrics are missing for {workload_id}")
                continue
            for metric_name in REQUIRED_METRICS:
                metric = metrics.get(metric_name)
                if isinstance(metric, bool) or not isinstance(metric, (int, float)) or metric < 0:
                    errors.append(f"{engine} metric {metric_name} is invalid for {workload_id}")
        metric_summary: dict[str, dict[str, float]] = {}
        for metric_name in REQUIRED_METRICS:
            values = [
                float(row["metrics"][metric_name])
                for row in rows_by_id.values()
                if isinstance(row.get("metrics"), dict) and metric_name in row["metrics"]
            ]
            if len(values) == len(expected_records):
                metric_summary[metric_name] = {
                    "p50": statistics.median(values),
                    "p95": percentile(values, 0.95),
                }
        payload["metric_summary"] = metric_summary
    absent = sorted(required_engines - set(engines))
    if absent:
        errors.append("required engines have no result: " + ", ".join(absent))
    if len(fingerprints) > 1:
        errors.append("engine results do not use the same model and tokenizer fingerprints")
    mismatched_outputs = [
        record_id for record_id, hashes in output_hashes.items() if len(hashes) > 1
    ]
    if mismatched_outputs:
        errors.append(
            "deterministic output token hashes differ for "
            + f"{len(mismatched_outputs)} workload records"
        )
    mismatched_inputs = [
        record_id for record_id, hashes in input_token_hashes.items() if len(hashes) > 1
    ]
    if mismatched_inputs:
        errors.append(
            "effective input token hashes differ for "
            + f"{len(mismatched_inputs)} workload records"
        )
    if len(execution_contracts) > 1:
        errors.append("engine results do not use the same execution contract")
    complete = not errors
    return {
        "schema_version": 1,
        "decision": "comparable" if complete else "incomplete",
        "workload": str(workload_path),
        "workload_sha256": workload_digest,
        "scope": "global"
        if workload["selection"]["global_cross_engine_claim_eligible"]
        else "scoped",
        "required_engines": sorted(required_engines),
        "gates": {
            "required_engines_present": not absent,
            "same_model_and_tokenizer": len(fingerprints) == 1,
            "complete_public_workload": not any("workload records" in error for error in errors),
            "same_public_prompts": not any("prompt hash" in error for error in errors),
            "same_effective_input_tokens": not mismatched_inputs
            and not any("input token hash" in error for error in errors),
            "same_execution_contract": len(execution_contracts) == 1,
            "deterministic_token_parity": not mismatched_outputs
            and not any("output token hash" in error for error in errors),
            "metrics_complete": not any(
                " metric " in error or "metrics are missing" in error for error in errors
            ),
        },
        "engine_summaries": {
            engine: {
                "engine_version": payload.get("engine_version"),
                "model_fingerprint": payload.get("model_fingerprint"),
                "metric_summary": payload.get("metric_summary", {}),
            }
            for engine, payload in sorted(engines.items())
        },
        "errors": errors,
    }


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    for name in ("sync", "verify"):
        subcommand = subparsers.add_parser(name)
        subcommand.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        subcommand.add_argument("--output", type=Path)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--output", type=Path)
    build = subparsers.add_parser("build-workload")
    build.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    build.add_argument("--profile", choices=sorted(PROFILE_DEFINITIONS), required=True)
    build.add_argument("--limit-per-stratum", type=int)
    build.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("validate-results")
    compare.add_argument("--workload", type=Path, required=True)
    compare.add_argument("--result", type=Path, action="append", required=True)
    compare.add_argument("--required-engine", action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return command_parser


def emit(payload: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_json(output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = parser().parse_args()
    if args.command == "inventory":
        emit(inventory_engines(), args.output)
        return
    lock_path = args.lock.resolve()
    lock = load_lock(lock_path)
    if args.command == "sync":
        emit(sync_install(lock, lock_path, args.data_root.resolve()), args.output)
        return
    if args.command == "verify":
        verification = verify_install(lock, args.data_root.resolve(), lock_path)
        emit(verification, args.output)
        return
    if args.command == "build-workload":
        workload = build_workload(
            lock,
            lock_path,
            args.data_root.resolve(),
            args.profile,
            args.limit_per_stratum,
        )
        emit(workload, args.output)
        return
    if args.command == "validate-results":
        result = validate_engine_results(
            args.workload.resolve(),
            [path.resolve() for path in args.result],
            set(args.required_engine),
        )
        emit(result, args.output)
        if result["decision"] != "comparable":
            raise SystemExit(1)
        return
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except PublicBenchmarkError as error:
        print(f"public benchmark error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
