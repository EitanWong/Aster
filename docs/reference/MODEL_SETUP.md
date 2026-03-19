# Model Setup Guide for Aster

This guide explains how to download and set up models for the Aster inference runtime.

## Quick Start

After cloning the repository, download all required models with a single command:

```bash
cd /path/to/Aster
python scripts/download_models.py --all
```

That's it. Models will be downloaded to `models/` and organized automatically.

## What Gets Downloaded

By default, `--all` downloads:

- **ASR (Speech-to-Text)**
  - Qwen3-ASR-0.6B-4bit (~0.4GB) — required

- **LLM (Language Model)**
  - Qwen3.5-9B-4bit (~5.2GB) — required
  - Qwen3.5-0.8B-4bit (~0.6GB) — optional (for speculative decoding)

- **TTS (Text-to-Speech)**
  - Qwen3-TTS-0.6B-Base-4bit (~0.5GB) — required
  - Qwen3-TTS-0.6B-CustomVoice-4bit (~0.5GB) — optional

**Total size: ~7.2GB (required) or ~8.2GB (all models)**

## Common Commands

### Download only required models
```bash
python scripts/download_models.py --required-only
```

### Download specific model group
```bash
python scripts/download_models.py --group llm
python scripts/download_models.py --group asr
python scripts/download_models.py --group tts
```

### Download a specific model
```bash
python scripts/download_models.py --model qwen3_5_9b
python scripts/download_models.py --model qwen3_tts_custom_voice
```

### List all available models
```bash
python scripts/download_models.py --list
```

### Verify existing models
```bash
python scripts/download_models.py --verify-only
```

### Force re-download (skip cache)
```bash
python scripts/download_models.py --all --force
```

## Directory Structure

After downloading, your `models/` directory will look like:

```
models/
├── manifest.yaml                          # Model registry (this file)
├── qwen3-asr-0.6b/                        # ASR model
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── qwen3.5-9b-mlx/                        # Main LLM
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── qwen3.5-0.8b-mlx/                      # Draft LLM (optional)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── qwen3-tts-0.6b-base/                   # TTS Base
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── qwen3-tts-0.6b-custom/                 # TTS CustomVoice (optional)
    ├── config.json
    ├── model.safetensors
    └── ...
```

## Configuration

Models are defined in `models/manifest.yaml`. Each model entry includes:

- **name**: Human-readable model name
- **description**: What the model does
- **purpose**: Role in the pipeline
- **required**: Whether it's mandatory
- **source**: Download source (currently: `huggingface`)
- **repo_id**: Hugging Face repository ID
- **target_path**: Local directory path
- **size_gb**: Approximate size for reference
- **notes**: Additional information

## Adding a New Model

To add a new model to the manifest:

1. Open `models/manifest.yaml`
2. Add an entry under the appropriate group (asr, llm, tts):

```yaml
models:
  llm:
    my_new_model:
      name: "My Model Name"
      description: "What it does"
      purpose: "Its role"
      required: false
      source: "huggingface"
      repo_id: "username/model-name"
      target_path: "models/my-model"
      size_gb: 5.0
      notes: "Any notes"
```

3. Run the downloader:
```bash
python scripts/download_models.py --model my_new_model
```

## Troubleshooting

### "huggingface-hub not installed"
Install it with:
```bash
pip install huggingface-hub
```

### Download interrupted
The downloader is resumable. Just run the same command again:
```bash
python scripts/download_models.py --all
```

It will skip already-downloaded models and resume any incomplete ones.

### Model verification warnings
If you see size mismatch warnings, the model may be incomplete. Force re-download:
```bash
python scripts/download_models.py --model <name> --force
```

### Slow downloads
Model downloads depend on your internet connection. Large models (5GB+) may take 10-30 minutes.

For faster downloads, consider:
- Using a wired connection
- Downloading during off-peak hours
- Checking your ISP's speed

### Disk space issues
Check available space:
```bash
df -h
```

Required space: ~7-8GB for all models.

## Configuration Integration

After downloading, update `configs/config.yaml` to point to your models:

```yaml
model:
  name: Qwen3.5-9B
  path: models/qwen3.5-9b-mlx
  draft_name: Qwen3.5-0.8B
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
  tts_custom_voice_path: models/qwen3-tts-0.6b-custom
```

## Advanced Usage

### Dry-run (list what would be downloaded)
```bash
python scripts/download_models.py --list
```

### Custom manifest location
```bash
python scripts/download_models.py --all --manifest /path/to/custom/manifest.yaml
```

### Custom base directory
```bash
python scripts/download_models.py --all --base-dir /mnt/models
```

## Architecture

The model download system consists of:

- **manifest.yaml**: Declarative model registry
- **scripts/download_models.py**: CLI entry point
- **scripts/lib/model_manager.py**: Core logic
  - `ModelManifest`: Loads and parses manifest
  - `ModelDownloader`: Handles downloads and verification
  - `DownloadSummary`: Formats results

The system is:
- **Idempotent**: Safe to run multiple times
- **Resumable**: Interrupted downloads can be resumed
- **Verifiable**: Checks model integrity
- **Extensible**: Easy to add new models or sources
- **Maintainable**: Clear separation of concerns

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your internet connection
3. Ensure you have enough disk space
4. Check `models/manifest.yaml` for model definitions
