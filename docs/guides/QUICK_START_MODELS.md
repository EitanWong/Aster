# Model Download System - Quick Reference

## One-Liner Setup

```bash
python scripts/download_models.py --all
```

That's it. All required models download to `models/` automatically.

## Common Commands

```bash
# Download all models
python scripts/download_models.py --all

# Download only required models
python scripts/download_models.py --required-only

# Download specific group
python scripts/download_models.py --group llm
python scripts/download_models.py --group asr
python scripts/download_models.py --group tts

# Download specific model
python scripts/download_models.py --model qwen3_5_9b

# List available models
python scripts/download_models.py --list

# Verify existing models
python scripts/download_models.py --verify-only

# Force re-download
python scripts/download_models.py --all --force
```

## What Gets Downloaded

| Model | Size | Required | Purpose |
|-------|------|----------|---------|
| Qwen3-ASR-0.6B | 0.4GB | ✓ | Speech-to-text |
| Qwen3.5-9B | 5.2GB | ✓ | Main LLM |
| Qwen3.5-0.8B | 0.6GB | ✗ | Speculative decoding |
| Qwen3-TTS-Base | 0.5GB | ✓ | Text-to-speech (voice cloning) |
| Qwen3-TTS-CustomVoice | 0.5GB | ✗ | Text-to-speech (preset voices) |

**Total: 7.2GB (required) or 8.2GB (all)**

## Directory Structure After Download

```
models/
├── manifest.yaml
├── qwen3-asr-0.6b/
├── qwen3.5-9b-mlx/
├── qwen3.5-0.8b-mlx/
├── qwen3-tts-0.6b-base/
└── qwen3-tts-0.6b-custom/
```

## Configuration

Update `configs/config.yaml`:

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
  tts_custom_voice_path: models/qwen3-tts-0.6b-custom
```

## Architecture

```
models/manifest.yaml (declarative registry)
    ↓
scripts/download_models.py (CLI)
    ↓
scripts/lib/model_manager.py (core logic)
    ├─ ModelManifest (load/parse)
    ├─ ModelDownloader (download/verify)
    └─ DownloadSummary (report)
```

## Key Features

✓ **Idempotent**: Safe to run multiple times  
✓ **Resumable**: Interrupted downloads resume  
✓ **Verifiable**: Checks model integrity  
✓ **Extensible**: Easy to add new models  
✓ **Observable**: Clear logging and summary  

## Troubleshooting

| Issue | Solution |
|-------|----------|
| huggingface-hub not installed | `pip install huggingface-hub` |
| Download interrupted | Rerun same command (resumes) |
| Size mismatch warning | Force re-download: `--force` |
| Slow downloads | Check internet, try off-peak |
| Disk space issues | Need ~8GB free |

## Adding a New Model

1. Edit `models/manifest.yaml`
2. Add entry under appropriate group
3. Run: `python scripts/download_models.py --model <name>`

See `docs/MODEL_SETUP.md` for details.
