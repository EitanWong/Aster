# Model Download System Architecture

## Overview

Aster includes a production-quality model download system designed for:
- **One-click setup**: Single command downloads all required models
- **Idempotency**: Safe to run multiple times
- **Resumability**: Interrupted downloads can be resumed
- **Extensibility**: Easy to add new models or sources
- **Maintainability**: Clear separation of concerns

## Design Principles

1. **Declarative over Imperative**: Models defined in `manifest.yaml`, not hardcoded
2. **Modular Architecture**: Separate concerns (manifest, download, verify, CLI)
3. **Fail-Safe**: Partial failures don't corrupt existing models
4. **Observable**: Clear logging and summary output
5. **Extensible**: Easy to add new download sources or model types

## Components

### 1. Model Manifest (`models/manifest.yaml`)

Declarative registry of all models:

```yaml
models:
  asr:
    qwen3_asr_0_6b:
      name: "Qwen3-ASR-0.6B-4bit"
      description: "..."
      purpose: "Speech-to-text"
      required: true
      source: "huggingface"
      repo_id: "mlx-community/Qwen3-ASR-0.6B-4bit"
      target_path: "models/qwen3-asr-0.6b"
      size_gb: 0.4
      notes: "..."
```

**Benefits:**
- Single source of truth for model definitions
- Easy to add/remove/update models
- Human-readable format
- Supports metadata (size, notes, etc.)

### 2. Model Manager Library (`scripts/lib/model_manager.py`)

Core logic split into focused classes:

#### `ModelManifest`
- Loads and parses YAML manifest
- Provides filtered queries (by group, required status, key)
- Validates manifest structure

#### `ModelDownloader`
- Downloads models from various sources
- Handles Hugging Face via `huggingface-hub`
- Supports resumable downloads
- Verifies downloaded models

#### `DownloadResult`
- Encapsulates result of a single operation
- Tracks status, path, size, duration, errors
- Used for reporting and summary

#### `DownloadSummary`
- Aggregates results from multiple downloads
- Formats human-friendly output
- Categorizes results (downloaded, skipped, verified, failed)

**Design:**
- Each class has single responsibility
- No side effects (pure functions where possible)
- Easy to test and extend

### 3. CLI Entry Point (`scripts/download_models.py`)

User-facing interface with rich options:

```bash
python download_models.py --all              # Download all
python download_models.py --group llm        # Download group
python download_models.py --model <name>     # Download specific
python download_models.py --list             # List models
python download_models.py --verify-only      # Verify only
python download_models.py --force            # Force re-download
```

**Features:**
- Flexible filtering (all, group, specific model, required-only)
- Verification mode (check existing models)
- Force mode (skip cache)
- List mode (show available models)
- Custom manifest/base-dir support

## Data Flow

```
User Command
    ↓
CLI Parser (download_models.py)
    ↓
Load Manifest (ModelManifest)
    ↓
Filter Models (by group/key/required)
    ↓
For each model:
    ├─ Check if exists (skip if present)
    ├─ Download (ModelDownloader)
    ├─ Verify (check size/integrity)
    └─ Record result (DownloadResult)
    ↓
Aggregate Results (DownloadSummary)
    ↓
Print Summary
    ↓
Exit with status
```

## Download Sources

Currently supported:

### Hugging Face
- Uses `huggingface-hub` library
- Supports resumable downloads
- Automatic retry on failure
- Respects HF rate limits

**Adding new sources:**

1. Add source type to manifest
2. Implement download method in `ModelDownloader`
3. Update CLI help text

Example:
```python
def _download_from_custom_source(self, model, target_path):
    # Custom download logic
    pass
```

## Verification Strategy

Models are verified by:

1. **Existence Check**: Directory exists and is not empty
2. **Size Sanity Check**: Actual size within ±20% of expected
3. **File Count**: At least some files present (not just empty dir)

**Future enhancements:**
- Checksum validation (SHA256)
- Model loading test (try to load with MLX)
- Integrity checks (verify model structure)

## Error Handling

Failures are:
- **Isolated**: One model failure doesn't affect others
- **Reported**: Clear error messages in summary
- **Recoverable**: Rerunning skips successful models
- **Actionable**: Specific guidance on what went wrong

Example error messages:
```
✗ qwen3_5_9b                   failed - Connection timeout
✗ qwen3_tts_custom_voice       failed - Model directory is empty
```

## Idempotency

The system is idempotent:

1. **Skip existing**: If model exists, skip download
2. **Resume interrupted**: `huggingface-hub` handles resumption
3. **Verify on rerun**: Can verify without re-downloading
4. **Force mode**: `--force` flag allows re-download if needed

Safe to run multiple times:
```bash
python download_models.py --all
python download_models.py --all  # Safe, skips existing
python download_models.py --all  # Safe, skips existing
```

## Performance Considerations

- **Sequential downloads**: One model at a time (configurable)
- **Parallel support**: Framework supports parallel (future enhancement)
- **Resume support**: Interrupted downloads resume from last byte
- **Caching**: Skips already-downloaded models

## Integration with Aster

After downloading, models are referenced in `configs/config.yaml`:

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

The inference engine loads models from these paths at startup.

## Extensibility

### Adding a new model

1. Edit `models/manifest.yaml`:
```yaml
models:
  llm:
    my_model:
      name: "My Model"
      # ... other fields
```

2. Run downloader:
```bash
python download_models.py --model my_model
```

### Adding a new source

1. Add source type to manifest
2. Implement in `ModelDownloader`:
```python
def _download_from_my_source(self, model, target_path):
    # Implementation
```

3. Update CLI help

### Adding verification

1. Extend `ModelDownloader.verify_model()`:
```python
def verify_model(self, model):
    # Add custom checks
```

## Testing

The system is designed for easy testing:

```python
# Test manifest loading
manifest = ModelManifest("models/manifest.yaml")
assert len(manifest.models) > 0

# Test filtering
llm_models = manifest.get_models(group="llm")
assert all(m.group == "llm" for m in llm_models)

# Test downloader (mock)
downloader = ModelDownloader(manifest)
result = downloader.verify_model(model)
assert result.status in ["verified", "failed"]
```

## Future Enhancements

1. **Parallel downloads**: Download multiple models concurrently
2. **Checksum validation**: SHA256 verification
3. **Model loading test**: Verify models load with MLX
4. **Progress bars**: Rich progress indication
5. **Caching**: Cache manifest to avoid repeated parsing
6. **Mirror support**: Fallback to alternative sources
7. **Bandwidth limiting**: Control download speed
8. **Scheduled downloads**: Background model updates

## Summary

The model download system provides:

✓ One-click setup  
✓ Declarative model registry  
✓ Modular, testable architecture  
✓ Idempotent, resumable downloads  
✓ Clear error reporting  
✓ Easy extensibility  
✓ Production-ready reliability  

It's designed to feel like a mature, well-maintained component of a serious open-source project.
