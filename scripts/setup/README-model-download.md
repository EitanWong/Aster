# Aster Model Download Instructions

One-click model download for all Aster models (ASR, LLM, TTS).

## Quick Start

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
bash scripts/setup/download_models.sh
```

This downloads all **required models** (~7.2GB):
- Qwen3-ASR-0.6B-4bit (0.4GB)
- Qwen3.5-9B-4bit (5.2GB)
- Qwen3-TTS-0.6B-Base-4bit (0.5GB)

## What the script does automatically

- Checks macOS + project structure
- Installs Homebrew if missing
- Installs Python 3.13 if needed
- Creates/reuses `.venv`
- Installs dependencies (PyYAML, huggingface-hub)
- Installs aria2 for accelerated downloads
- Downloads hfd helper for fast HF downloads
- Sets HF mirror to `https://hf-mirror.com` (fast in mainland China)
- Downloads all models with hfd + aria2 acceleration
- Verifies downloads

## Common Commands

```bash
# Download all required models (default)
bash scripts/setup/download_models.sh

# Download all models (required + optional)
bash scripts/setup/download_models.sh --all

# Download specific group
bash scripts/setup/download_models.sh --group llm
bash scripts/setup/download_models.sh --group asr
bash scripts/setup/download_models.sh --group tts

# Download specific model
bash scripts/setup/download_models.sh --model qwen3_5_9b

# List available models
bash scripts/setup/download_models.sh --list

# Verify existing models
bash scripts/setup/download_models.sh --verify-only

# Force re-download
bash scripts/setup/download_models.sh --all --force
```

## Environment Variables

```bash
# Use custom HF mirror (default: https://hf-mirror.com)
HF_ENDPOINT=https://huggingface.co bash scripts/setup/download_models.sh

# Disable hfd acceleration (use standard huggingface-hub)
USE_HFD=0 bash scripts/setup/download_models.sh

# Skip aria2 installation
INSTALL_ARIA2=0 bash scripts/setup/download_models.sh

# Pass HF token for gated repos
HF_TOKEN="hf_your_token" bash scripts/setup/download_models.sh
```

## Download Performance

With hfd + aria2 (default):
- **Fast**: ~50-100 MB/s on good connections
- **Resumable**: Interrupted downloads resume automatically
- **Parallel**: aria2 uses multiple connections

## Models Downloaded

| Model | Size | Required | Purpose |
|-------|------|----------|---------|
| Qwen3-ASR-0.6B | 0.4GB | ✓ | Speech-to-text |
| Qwen3.5-9B | 5.2GB | ✓ | Main LLM |
| Qwen3.5-0.8B | 0.6GB | ✗ | Speculative decoding |
| Qwen3-TTS-Base | 0.5GB | ✓ | Text-to-speech (voice cloning) |
| Qwen3-TTS-CustomVoice | 0.5GB | ✗ | Text-to-speech (preset voices) |

## After Download

Models are automatically organized in `models/`:

```
models/
├── qwen3-asr-0.6b/
├── qwen3.5-9b-mlx/
├── qwen3.5-0.8b-mlx/
├── qwen3-tts-0.6b-base/
└── qwen3-tts-0.6b-custom/
```

Update `configs/config.yaml` with model paths:

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
  tts_custom_voice_path: models/qwen3-tts-0.6b-custom
```

Then start the server:

```bash
source .venv/bin/activate
python server.py --config configs/config.yaml
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Download interrupted | Rerun same command (resumes) |
| aria2 installation fails | Continue without it (slower) |
| hfd download fails | Set `USE_HFD=0` to use standard download |
| Slow downloads | Check internet, try off-peak hours |
| Disk space issues | Need ~8GB free |

## Architecture

```
Shell wrapper (download_models.sh)
    ↓
Setup (Homebrew, Python, venv, dependencies)
    ↓
Configure (HF mirror, hfd, aria2)
    ↓
Python downloader (download_models.py)
    ↓
hfd + aria2 (fast downloads)
    ↓
Verify & report
```
