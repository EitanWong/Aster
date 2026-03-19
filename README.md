<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">

  # Aster

  **Production-oriented Apple Silicon local LLM inference runtime**

  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster is a production-oriented Apple Silicon local LLM inference runtime for long-context, OpenClaw-style agent workloads.

## Why Aster

Aster is optimized for:

- huge prompts and repeated long prefixes
- tool-heavy agent prompts
- long conversations
- continuous local background serving
- benchmark-validated runtime policy selection
- Apple Silicon + MLX deployment

It exposes an OpenAI-compatible API and treats advanced optimizations as candidate strategies, not dogma. Speculative decoding, prefix caching, batching, scheduling, and streaming cadence are benchmarked and selected based on measured local performance and stability.

## Core ideas

- OpenAI-compatible API with streaming and non-streaming endpoints
- explicit prefill/decode split
- adaptive scheduler with queue-aware batching
- paged KV manager abstraction
- automatic prefix cache with deterministic hashing
- speculative decoding controller with auto-disable fallback
- benchmark/autotune subsystem that persists the fastest stable profile
- structured logs, metrics, supervision, and readiness/health reporting

## Quick start

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
python server.py --config configs/config.yaml
```

## Python version

Aster targets modern Python and should be run on Python 3.13.x when available (3.12+ minimum). The macOS system Python is considered unsupported for this project.

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Compatibility notes:
- See `docs/OPENAI_COMPAT.md` for Aster's default compatibility contract and opt-in debug extensions.

## Benchmarking philosophy

Startup autotune can run a short warmup benchmark to choose the fastest stable policy. The benchmark subsystem compares:

- speculative on/off
- draft token counts
- prefix cache on/off
- batch windows
- batch caps
- page sizes
- scheduling modes
- streaming flush cadence

Profiles are persisted and used on subsequent startups.

## Apple Silicon tuning notes

- favor preallocation and page pools over repeated dynamic allocations
- use MLX model residency carefully to avoid unified-memory thrash
- benchmark prefix caching and speculative decoding per machine
- keep Python hot paths small; move coordination into stable loops
- prioritize consistent first-token latency under long prompts

## Dynamic optimization philosophy

Aster enables only optimizations that prove beneficial on the local machine:

- speculative decoding can be disabled globally or by request class
- prefix cache can be reduced or disabled when hit rate is low or memory pressure rises
- batching windows shrink automatically when latency rises
- fallback profiles are selected when instability or regressions are detected

## Model Setup

One-click model download with hfd + aria2 acceleration:

```bash
# Download all required models (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# Or use Python directly for more control
python scripts/download_models.py --all
python scripts/download_models.py --group llm
python scripts/download_models.py --list
```

See `scripts/setup/README-model-download.md` for detailed instructions.

## Model paths

`model.path` and `model.draft_path` can be either:
- absolute local paths to MLX-converted model directories
- compatible Hugging Face repo ids loadable by `mlx-lm`

For production, prefer local MLX-converted directories. Update `configs/config.yaml`:

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

## OpenClaw integration

Point OpenClaw to Aster's OpenAI-compatible base URL and model id. Aster is built for repeated system/tool prefixes and long-lived agent sessions, so it should particularly benefit workloads with stable scaffolding and long-context reuse.

## Project guidance docs

- `scripts/setup/README-model-download.md` — model download guide
- `docs/MODEL_SETUP.md` — detailed setup and troubleshooting
- `docs/MODEL_DOWNLOAD_ARCHITECTURE.md` — system design
- `docs/ROADMAP.md` — long-term architectural evolution plan
- `docs/OPENAI_COMPAT.md` — compatibility boundary and debug extensions
- `docs/DEBUGGING.md` — operator debugging guide
- `docs/OPERATIONS.md` — day-to-day service operations
