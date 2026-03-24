#!/usr/bin/env python3
"""
Interactive and non-interactive model downloader for Aster.

Features:
- downloads LLM, audio, and embedding models into the local models/ directory
- supports MLX conversion for raw Hugging Face LLMs
- supports snapshot download for already-packaged MLX/audio/embedding repos
- prints config snippets after download
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

def ensure_project_python() -> None:
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return
    if not VENV_PYTHON.exists():
        return
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    ensure_project_python()
    from huggingface_hub import snapshot_download


DownloadStrategy = Literal["mlx_convert", "snapshot"]
Category = Literal["llm", "audio", "embeddings"]


@dataclass(frozen=True)
class ModelSpec:
    category: Category
    key: str
    name: str
    hf_id: str
    description: str
    size: str
    output_dir: str
    strategy: DownloadStrategy
    quantize: bool = False
    context: str | None = None
    recommended_for: str | None = None
    dimensions: int | None = None  # embedding output dimensions

    @property
    def full_key(self) -> str:
        return f"{self.category}:{self.key}"

    @property
    def destination(self) -> Path:
        return MODELS_DIR / self.output_dir


CATALOG: dict[Category, list[ModelSpec]] = {
    "llm": [
        # ── Qwen3.5 dense ──────────────────────────────────────────────────────
        ModelSpec(
            category="llm",
            key="qwen35_9b_r1",
            name="Qwen3.5-9B-Reasoning-Distilled-4bit",
            hf_id="Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit",
            description="Qwen3.5-9B distilled from Claude Opus for reasoning. Pre-packaged MLX 4-bit.",
            size="~5 GB",
            output_dir="qwen3.5-9b-reasoning-mlx",
            strategy="snapshot",
            context="32K",
            recommended_for="local reasoning",
        ),
        ModelSpec(
            category="llm",
            key="qwen35_9b",
            name="Qwen3.5-9B-4bit",
            hf_id="mlx-community/Qwen3.5-9B-4bit",
            description="Balanced 9B chat model. Pre-quantised MLX 4-bit snapshot.",
            size="~5 GB",
            output_dir="qwen3.5-9b-mlx",
            strategy="snapshot",
            context="32K",
            recommended_for="general local chat",
        ),
        ModelSpec(
            category="llm",
            key="qwen35_0.8b",
            name="Qwen3.5-0.8B-4bit",
            hf_id="mlx-community/Qwen3.5-0.8B-4bit",
            description="Tiny 0.8B model for speculative decoding. Pairs with any 9B/27B main model.",
            size="~0.6 GB",
            output_dir="qwen3.5-0.8b-mlx",
            strategy="snapshot",
            context="32K",
            recommended_for="draft / speculative decoding",
        ),
        ModelSpec(
            category="llm",
            key="qwen35_27b",
            name="Qwen3.5-27B-4bit",
            hf_id="mlx-community/Qwen3.5-27B-4bit",
            description="High-quality dense 27B model. Good upper-tier choice for Macs with 32 GB+ RAM.",
            size="~16 GB",
            output_dir="qwen3.5-27b-mlx",
            strategy="snapshot",
            context="32K",
            recommended_for="high-quality chat, 32 GB+ RAM",
        ),
        # ── Qwen3.5 MoE (sparse, small active params) ─────────────────────────
        ModelSpec(
            category="llm",
            key="qwen35_moe_35b",
            name="Qwen3.5-35B-A3B-Instruct-4bit",
            hf_id="mlx-community/Qwen3.5-35B-A3B-4bit",
            description="35B MoE with only 3B active params. Runs on 16 GB RAM with quality rivaling 32B dense.",
            size="~22 GB",
            output_dir="qwen3.5-35b-a3b-mlx",
            strategy="snapshot",
            context="32K",
            recommended_for="large MoE, 24 GB+ RAM",
        ),
        # ── Qwen3 MoE (30B-A3B) ───────────────────────────────────────────────
        ModelSpec(
            category="llm",
            key="qwen3_moe_30b",
            name="Qwen3-30B-A3B-Instruct-4bit",
            hf_id="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
            description="30B MoE (3B active). Latest Qwen3 generation. Excellent reasoning + coding.",
            size="~18 GB",
            output_dir="qwen3-30b-a3b-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="reasoning, coding, 24 GB+ RAM",
        ),
        ModelSpec(
            category="llm",
            key="qwen3_coder_30b",
            name="Qwen3-Coder-30B-A3B-Instruct-4bit",
            hf_id="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            description="Coding-specialised 30B MoE. Top open-source coding model for macOS.",
            size="~18 GB",
            output_dir="qwen3-coder-30b-a3b-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="coding assistant, 24 GB+ RAM",
        ),
        # ── GPT-OSS (OpenAI open-weight MoE) ──────────────────────────────────
        ModelSpec(
            category="llm",
            key="gpt_oss_20b",
            name="gpt-oss-20b-MXFP4-Q4",
            hf_id="mlx-community/gpt-oss-20b-MXFP4-Q4",
            description="OpenAI open-weight 20B MoE (128 experts, 4B active). Native mlx-lm support.",
            size="~12 GB",
            output_dir="gpt-oss-20b-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="OpenAI-architecture MoE, 16 GB+ RAM",
        ),
        ModelSpec(
            category="llm",
            key="gpt_oss_20b_q8",
            name="gpt-oss-20b-MXFP4-Q8",
            hf_id="mlx-community/gpt-oss-20b-MXFP4-Q8",
            description="OpenAI open-weight 20B MoE, 8-bit quantisation. Higher quality than Q4 variant.",
            size="~22 GB",
            output_dir="gpt-oss-20b-8bit-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="OpenAI-architecture MoE quality, 32 GB+ RAM",
        ),
        # ── DeepSeek ──────────────────────────────────────────────────────────
        ModelSpec(
            category="llm",
            key="deepseek_r1_8b",
            name="DeepSeek-R1-0528-Qwen3-8B-4bit",
            hf_id="mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
            description="DeepSeek-R1 reasoning distilled to 8B on Qwen3 backbone. Strong reasoning at 8B.",
            size="~5 GB",
            output_dir="deepseek-r1-8b-mlx",
            strategy="snapshot",
            context="32K",
            recommended_for="chain-of-thought reasoning",
        ),
        ModelSpec(
            category="llm",
            key="deepseek_r1_32b",
            name="DeepSeek-R1-Distill-Qwen-32B-4bit",
            hf_id="mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
            description="DeepSeek-R1 distilled to 32B Qwen backbone. Best open reasoning model at this size.",
            size="~20 GB",
            output_dir="deepseek-r1-32b-mlx",
            strategy="snapshot",
            context="32K",
            recommended_for="best open reasoning, 32 GB+ RAM",
        ),
        # ── Llama ─────────────────────────────────────────────────────────────
        ModelSpec(
            category="llm",
            key="llama_8b",
            name="Llama-3.1-8B-Instruct-4bit",
            hf_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            description="Meta Llama 3.1 8B in 4-bit. Reliable general-purpose model with strong tool use.",
            size="~5 GB",
            output_dir="llama3.1-8b-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="general chat, function calling",
        ),
        ModelSpec(
            category="llm",
            key="llama_70b",
            name="Llama-3.3-70B-Instruct-4bit",
            hf_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
            description="Llama 3.3 70B at 4-bit. Very high quality; requires 48 GB+ RAM.",
            size="~40 GB",
            output_dir="llama3.3-70b-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="highest quality, 64 GB+ RAM",
        ),
        # ── Mistral ───────────────────────────────────────────────────────────
        ModelSpec(
            category="llm",
            key="mistral_small_24b",
            name="Mistral-Small-3.2-24B-Instruct-4bit",
            hf_id="mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
            description="Mistral's latest small model. Very capable for 24B with vision support.",
            size="~14 GB",
            output_dir="mistral-small-24b-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="multimodal, multilingual, 24 GB+ RAM",
        ),
        # ── Phi (Microsoft) ───────────────────────────────────────────────────
        ModelSpec(
            category="llm",
            key="phi4_mini",
            name="Phi-4-mini-instruct-4bit",
            hf_id="mlx-community/Phi-4-mini-instruct-4bit",
            description="Microsoft Phi-4 mini (3.8B). Extremely efficient; top quality at tiny size.",
            size="~2.5 GB",
            output_dir="phi4-mini-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="fast inference, 8 GB RAM",
        ),
        # ── Gemma ─────────────────────────────────────────────────────────────
        ModelSpec(
            category="llm",
            key="gemma3_12b",
            name="gemma-3-text-12b-it-4bit",
            hf_id="mlx-community/gemma-3-text-12b-it-4bit",
            description="Google Gemma 3 12B text model. Strong multilingual and coding capability.",
            size="~7 GB",
            output_dir="gemma3-12b-mlx",
            strategy="snapshot",
            context="128K",
            recommended_for="multilingual, coding, 16 GB RAM",
        ),
    ],
    "audio": [
        # ── ASR ───────────────────────────────────────────────────────────────
        ModelSpec(
            category="audio",
            key="asr",
            name="Qwen3-ASR-0.6B",
            hf_id="Qwen/Qwen3-ASR-0.6B",
            description="Primary Aster ASR model (vllm-mlx backend). Multilingual speech recognition.",
            size="~0.66 GB",
            output_dir="qwen3-asr-0.6b",
            strategy="snapshot",
            recommended_for="ASR (primary)",
        ),
        ModelSpec(
            category="audio",
            key="asr_1.7b",
            name="Qwen3-ASR-1.7B-8bit",
            hf_id="mlx-community/Qwen3-ASR-1.7B-8bit",
            description="Larger Qwen3 ASR model (1.7B, 8-bit MLX). Higher accuracy for complex audio.",
            size="~1.7 GB",
            output_dir="qwen3-asr-1.7b-mlx",
            strategy="snapshot",
            recommended_for="high-accuracy ASR",
        ),
        ModelSpec(
            category="audio",
            key="whisper_turbo",
            name="whisper-large-v3-turbo",
            hf_id="mlx-community/whisper-large-v3-turbo-fp16",
            description="OpenAI Whisper large-v3-turbo in MLX fp16. Fast and accurate multilingual ASR.",
            size="~1.5 GB",
            output_dir="whisper-large-v3-turbo-mlx",
            strategy="snapshot",
            recommended_for="whisper-based ASR, fast",
        ),
        ModelSpec(
            category="audio",
            key="parakeet",
            name="parakeet-tdt-0.6b-v3",
            hf_id="mlx-community/parakeet-tdt-0.6b-v3",
            description="NVIDIA NeMo Parakeet 0.6B (MLX). Best English-only ASR accuracy at this size.",
            size="~0.6 GB",
            output_dir="parakeet-tdt-0.6b-mlx",
            strategy="snapshot",
            recommended_for="English ASR accuracy",
        ),
        # ── TTS ───────────────────────────────────────────────────────────────
        ModelSpec(
            category="audio",
            key="tts",
            name="Qwen3-TTS-0.6B-Base",
            hf_id="Qwen/Qwen3-TTS-0.6B",
            description="Primary Aster TTS model (vllm-mlx backend). Natural-sounding multilingual TTS.",
            size="~1.59 GB",
            output_dir="qwen3-tts-0.6b-base",
            strategy="snapshot",
            recommended_for="TTS (primary)",
        ),
        ModelSpec(
            category="audio",
            key="tts_custom",
            name="Qwen3-TTS-CustomVoice-0.6B",
            hf_id="Qwen/Qwen3-TTS-CustomVoice-0.6B",
            description="Voice cloning add-on for Qwen3 TTS. Clone any voice from a reference sample.",
            size="~1.2 GB",
            output_dir="qwen3-tts-0.6b-custom",
            strategy="snapshot",
            recommended_for="voice cloning",
        ),
        ModelSpec(
            category="audio",
            key="tts_1.7b",
            name="Qwen3-TTS-1.7B-VoiceDesign-8bit",
            hf_id="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            description="Larger 1.7B Qwen3 TTS in MLX 8-bit. More expressive and controllable voices.",
            size="~1.7 GB",
            output_dir="qwen3-tts-1.7b-mlx",
            strategy="snapshot",
            recommended_for="high-quality expressive TTS",
        ),
        ModelSpec(
            category="audio",
            key="kokoro",
            name="Kokoro-82M-8bit",
            hf_id="mlx-community/Kokoro-82M-8bit",
            description="Kokoro 82M TTS in MLX 8-bit. Extremely small, fast, English-focused.",
            size="~0.1 GB",
            output_dir="kokoro-82m-mlx",
            strategy="snapshot",
            recommended_for="ultra-fast English TTS",
        ),
        ModelSpec(
            category="audio",
            key="spark_tts",
            name="Spark-TTS-0.5B-8bit",
            hf_id="mlx-community/Spark-TTS-0.5B-8bit",
            description="Spark TTS 0.5B in MLX 8-bit. High-quality zero-shot voice cloning.",
            size="~0.5 GB",
            output_dir="spark-tts-mlx",
            strategy="snapshot",
            recommended_for="zero-shot voice cloning TTS",
        ),
        ModelSpec(
            category="audio",
            key="cosyvoice2",
            name="CosyVoice2-0.5B-8bit",
            hf_id="mlx-community/CosyVoice2-0.5B-8bit",
            description="CosyVoice2 0.5B (MLX 8-bit). Expressive streaming TTS with emotion control.",
            size="~0.5 GB",
            output_dir="cosyvoice2-mlx",
            strategy="snapshot",
            recommended_for="expressive streaming TTS",
        ),
    ],
    "embeddings": [
        ModelSpec(
            category="embeddings",
            key="qwen_0.6b",
            name="Qwen3-Embedding-0.6B-4bit-DWQ",
            hf_id="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            description="1024-dim Qwen3 embedding model. Primary Aster embedding backend (vllm-mlx).",
            size="~0.5 GB",
            output_dir="qwen3-embedding-0.6b-4bit-dwq",
            strategy="snapshot",
            recommended_for="RAG / memory (primary)",
            dimensions=1024,
        ),
        ModelSpec(
            category="embeddings",
            key="qwen_4b",
            name="Qwen3-Embedding-4B-4bit-DWQ",
            hf_id="mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
            description="4B Qwen3 embedding model in 4-bit DWQ. Higher quality for semantic search.",
            size="~2.5 GB",
            output_dir="qwen3-embedding-4b-4bit-dwq",
            strategy="snapshot",
            recommended_for="high-quality RAG",
            dimensions=2560,
        ),
        ModelSpec(
            category="embeddings",
            key="bge_m3",
            name="bge-m3-mlx-8bit",
            hf_id="mlx-community/bge-m3-mlx-8bit",
            description="BAAI BGE-M3 in MLX 8-bit. Best multilingual embedding model (100+ languages).",
            size="~0.6 GB",
            output_dir="bge-m3-mlx",
            strategy="snapshot",
            recommended_for="multilingual RAG",
            dimensions=1024,
        ),
        ModelSpec(
            category="embeddings",
            key="minilm",
            name="all-MiniLM-L6-v2-4bit",
            hf_id="mlx-community/all-MiniLM-L6-v2-4bit",
            description="Tiny 22M embedding model in 4-bit. Matches vllm-mlx guide examples. Ultra-fast.",
            size="~0.05 GB",
            output_dir="all-minilm-l6-v2-4bit",
            strategy="snapshot",
            recommended_for="fast lightweight embeddings",
            dimensions=384,
        ),
        ModelSpec(
            category="embeddings",
            key="e5_mistral",
            name="e5-mistral-7b-instruct-mlx",
            hf_id="mlx-community/e5-mistral-7b-instruct-mlx",
            description="Instruction-tuned 7B embedding model. Top quality for complex retrieval tasks.",
            size="~4 GB",
            output_dir="e5-mistral-7b-mlx",
            strategy="snapshot",
            recommended_for="high-quality instruction-based retrieval",
            dimensions=4096,
        ),
    ],
}


def print_header(text: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {text}")
    print(f"{'=' * 72}\n")


def all_specs() -> list[ModelSpec]:
    return [spec for group in CATALOG.values() for spec in group]


def lookup_spec(full_key: str) -> ModelSpec:
    for spec in all_specs():
        if spec.full_key == full_key:
            return spec
    raise KeyError(full_key)


# Flexible model key lookup: allows either full_key or short key, warns on ambiguity
def lookup_spec_flexible(model_key: str) -> ModelSpec:
    normalized = model_key.strip()

    for spec in all_specs():
        if normalized == spec.full_key:
            return spec

    matches = [spec for spec in all_specs() if normalized == spec.key]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        available = ", ".join(spec.full_key for spec in matches)
        raise KeyError(
            f"Ambiguous model key '{model_key}'. Use one of: {available}"
        )

    raise KeyError(model_key)


def list_models() -> None:
    print_header("Available Models")
    for category in ("llm", "audio", "embeddings"):
        print(f"[{category}]")
        for spec in CATALOG[category]:  # type: ignore[index]
            print(f"  - {spec.full_key}")
            print(f"    name: {spec.name}")
            print(f"    size: {spec.size}")
            print(f"    repo: {spec.hf_id}")
            print(f"    output: {spec.destination}")
            print(f"    note: {spec.description}")
            if spec.recommended_for:
                print(f"    best for: {spec.recommended_for}")
        print()


def run_hfd_download(spec: ModelSpec) -> bool:
    use_hfd = os.environ.get("USE_HFD", "0") == "1"
    hfd_path = os.environ.get("HFD_PATH", "").strip()
    if not use_hfd or not hfd_path:
        return False

    hfd_executable = Path(hfd_path)
    if not hfd_executable.exists():
        return False

    if shutil.which("aria2c") is None:
        return False

    cmd = [str(hfd_executable), spec.hf_id, "--dir", str(spec.destination)]
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        cmd.extend(["--hf_token", token])

    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    print("\nAttempting accelerated download via hfd + aria2...\n")
    if endpoint:
        print(f"hfd endpoint: {endpoint}")

    subprocess.run(cmd, check=True)
    return True


def run_snapshot_download(spec: ModelSpec) -> None:
    spec.destination.mkdir(parents=True, exist_ok=True)

    if run_hfd_download(spec):
        return

    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    download_kwargs = {
        "repo_id": spec.hf_id,
        "local_dir": str(spec.destination),
    }

    try:
        snapshot_download(**download_kwargs)
        return
    except (httpx.HTTPError, OSError):
        is_non_default_endpoint = bool(endpoint) and endpoint.rstrip("/") != DEFAULT_HF_ENDPOINT
        if not is_non_default_endpoint:
            raise

        print(
            "\nPrimary snapshot download failed while using a non-default HF_ENDPOINT. "
            "Retrying once with the official Hugging Face endpoint...\n"
        )
        print(f"original HF_ENDPOINT: {endpoint}")
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT
        try:
            snapshot_download(**download_kwargs)
            print("Retry with official Hugging Face endpoint succeeded.\n")
            return
        except Exception:
            os.environ["HF_ENDPOINT"] = endpoint
            raise


def run_mlx_convert(spec: ModelSpec) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        spec.hf_id,
        "--output-dir",
        str(spec.destination),
    ]
    if spec.quantize:
        cmd.extend(["--quantize", "4bit"])
    subprocess.run(cmd, check=True)


def download_spec(spec: ModelSpec) -> None:
    print_header(f"Downloading {spec.name}")
    print(f"source:      {spec.hf_id}")
    print(f"destination: {spec.destination}")
    print(f"size:        {spec.size}")
    print(f"strategy:    {spec.strategy}")
    if spec.destination.exists() and any(spec.destination.iterdir()):
        print("\nDestination already contains files. Reusing existing directory and syncing missing files.\n")

    if spec.strategy == "snapshot":
        run_snapshot_download(spec)
    elif spec.strategy == "mlx_convert":
        run_mlx_convert(spec)
    else:
        raise ValueError(f"Unsupported strategy: {spec.strategy}")

    print(f"\nDownload complete: {spec.destination}\n")


def print_config_hint(spec: ModelSpec) -> None:
    print_header("Config Hint")
    if spec.category == "llm":
        print("Update your Aster config like this:\n")
        print("model:")
        print(f"  name: {spec.name}")
        print(f"  path: {spec.destination}")
        print("  runtime: mlx")
        if spec.context:
            print(f"  context_length: {spec.context.replace('K', '000')}")
        return

    if spec.category == "audio":
        asr_keys = {"asr", "asr_1.7b", "whisper_turbo", "parakeet"}
        tts_custom_keys = {"tts_custom"}
        if spec.key in asr_keys:
            print("audio:")
            print("  asr:")
            print("    enabled: true")
            print("    backend: vllm_mlx")
            print(f"    model: {spec.name}")
            print(f"    path: {spec.destination}")
        elif spec.key in tts_custom_keys:
            print("audio:")
            print("  tts:")
            print(f"    custom_voice_model: {spec.hf_id}")
            print(f"    custom_voice_path: {spec.destination}")
        else:
            # All other audio keys are TTS variants
            print("audio:")
            print("  tts:")
            print("    enabled: true")
            print("    backend: vllm_mlx")
            print(f"    model: {spec.name}")
            print(f"    path: {spec.destination}")
        return

    if spec.category == "embeddings":
        dim = spec.dimensions or 1024
        print("For direct MLX embeddings inside Aster:")
        print("embeddings:")
        print("  enabled: true")
        print("  backend: mlx")
        print(f"  model: {spec.hf_id}")
        print(f"  model_path: {spec.destination}")
        print(f"  dimensions: {dim}")
        print("  max_length: 512\n")

        print("For vllm-mlx-backed embeddings inside Aster:")
        print("embeddings:")
        print("  enabled: true")
        print("  backend: vllm_mlx")
        print(f"  model: {spec.hf_id}")
        print(f"  model_path: {spec.destination}")
        print(f"  dimensions: {dim}")
        print("vllm_mlx:")
        print("  embedding_model: null   # allow lazy loading by requested model/path")
        return


def prompt_choice(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Please choose a value between {minimum} and {maximum}.")


def choose_category() -> Category | None:
    categories: list[Category] = ["llm", "audio", "embeddings"]
    print_header("Select Model Category")
    for index, category in enumerate(categories, start=1):
        print(f"  {index}. {category}")
    print("  0. exit")
    choice = prompt_choice("\nEnter choice: ", 0, len(categories))
    if choice == 0:
        return None
    return categories[choice - 1]


def choose_spec(category: Category) -> ModelSpec | None:
    options = CATALOG[category]
    print_header(f"Select {category} Model")
    for index, spec in enumerate(options, start=1):
        print(f"  {index}. {spec.name}")
        print(f"     {spec.description}")
        print(f"     output: {spec.destination}")
        print()
    print("  0. back")
    choice = prompt_choice("Enter choice: ", 0, len(options))
    if choice == 0:
        return None
    return options[choice - 1]


def interactive_main() -> int:
    print_header("Aster Model Downloader")
    while True:
        category = choose_category()
        if category is None:
            print("Bye.")
            return 0
        spec = choose_spec(category)
        if spec is None:
            continue
        download_spec(spec)
        print_config_hint(spec)
        again = input("\nDownload another model? [y/N]: ").strip().lower()
        if again != "y":
            print("Bye.")
            return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Aster models into the local models/ directory")
    parser.add_argument("--list", action="store_true", help="List available model keys and exit")
    parser.add_argument(
        "--model",
        help="Download a model non-interactively, e.g. llm:qwen35_moe_35b or qwen35_moe_35b",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the config snippet for the selected model after download",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_models()
        return 0

    if args.model:
        try:
            spec = lookup_spec_flexible(args.model)
        except KeyError as exc:
            message = str(exc)
            if message.startswith("\"") and message.endswith("\""):
                message = message[1:-1]
            print(message or f"Unknown model key: {args.model}", file=sys.stderr)
            print("Run with --list to see available model keys.", file=sys.stderr)
            return 1
        download_spec(spec)
        if args.show_config:
            print_config_hint(spec)
        return 0

    return interactive_main()


if __name__ == "__main__":
    raise SystemExit(main())
