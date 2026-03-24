# Aster Models Guide

This guide helps you discover, download, and configure models for Aster. All models are optimized for Apple Silicon (MLX runtime).

## Quick Start

**Download and use a model in 2 steps:**

```bash
# 1. Download the model
python -m mlx_lm.convert \
  --hf-path <model-id> \
  --output-dir ./models/<model-name>

# 2. Update configs/config.yaml with the model path
```

Then restart Aster:
```bash
python scripts/ops/daemon.py restart
```

---

## LLM Models (Main Inference)

### Recommended Models by Use Case

#### 🚀 Best for Agent Workloads (Recommended)
- **Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit**
  - Size: ~6GB (4-bit quantized)
  - Context: 32K tokens
  - Strengths: Reasoning, tool use, long context, Claude-style responses
  - Best for: OpenClaw agents, complex reasoning, multi-step tasks
  - Download:
    ```bash
    python -m mlx_lm.convert \
      --hf-path Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit \
      --output-dir ./models/qwen3.5-9b-reasoning-distilled-4bit
    ```

#### ⚡ Fast & Balanced
- **Qwen/Qwen3.5-9B-Instruct** (MLX-converted)
  - Size: ~9GB (full precision) / ~5GB (4-bit)
  - Context: 32K tokens
  - Strengths: Fast inference, good quality, balanced performance
  - Best for: General chat, API serving, production workloads
  - Download:
    ```bash
    python -m mlx_lm.convert \
      --hf-path Qwen/Qwen3.5-9B-Instruct \
      --output-dir ./models/qwen3.5-9b-instruct-4bit \
      --quantize 4bit
    ```

#### 💪 High Quality (Larger)
- **Qwen/Qwen3.5-32B-Instruct** (MLX-converted)
  - Size: ~20GB (4-bit)
  - Context: 32K tokens
  - Strengths: Better reasoning, higher quality outputs
  - Best for: Complex tasks, when you have GPU memory available
  - Download:
    ```bash
    python -m mlx_lm.convert \
      --hf-path Qwen/Qwen3.5-32B-Instruct \
      --output-dir ./models/qwen3.5-32b-instruct-4bit \
      --quantize 4bit
    ```

#### 🎯 Lightweight (Mobile/Edge)
- **Qwen/Qwen3.5-1B-Instruct** (MLX-converted)
  - Size: ~1.5GB (4-bit)
  - Context: 32K tokens
  - Strengths: Very fast, low memory, good for edge devices
  - Best for: Quick responses, resource-constrained environments
  - Download:
    ```bash
    python -m mlx_lm.convert \
      --hf-path Qwen/Qwen3.5-1B-Instruct \
      --output-dir ./models/qwen3.5-1b-instruct-4bit \
      --quantize 4bit
    ```

#### 🧠 Specialized: Reasoning
- **Qwen/QwQ-32B-Preview**
  - Size: ~20GB (4-bit)
  - Context: 32K tokens
  - Strengths: Deep reasoning, math, code generation
  - Best for: Complex problem-solving, research tasks
  - Download:
    ```bash
    python -m mlx_lm.convert \
      --hf-path Qwen/QwQ-32B-Preview \
      --output-dir ./models/qwq-32b-preview-4bit \
      --quantize 4bit
    ```

#### 🔬 Specialized: Code
- **Qwen/Qwen3.5-Coder-32B-Instruct** (MLX-converted)
  - Size: ~20GB (4-bit)
  - Context: 32K tokens
  - Strengths: Code generation, debugging, technical tasks
  - Best for: Programming assistance, code review
  - Download:
    ```bash
    python -m mlx_lm.convert \
      --hf-path Qwen/Qwen3.5-Coder-32B-Instruct \
      --output-dir ./models/qwen3.5-coder-32b-4bit \
      --quantize 4bit
    ```

---

## Audio Models

### ASR (Speech-to-Text)

#### Qwen3-ASR-0.6B (Default)
- Size: 0.66GB
- Languages: Chinese, English, and 100+ others
- Accuracy: High
- Speed: Real-time on Apple Silicon
- Download:
  ```bash
  python -m mlx_lm.convert \
    --hf-path Qwen/Qwen3-ASR-0.6B \
    --output-dir ./models/qwen3-asr-0.6b
  ```

### TTS (Text-to-Speech)

#### Qwen3-TTS-0.6B (Base)
- Size: 1.59GB
- Languages: Chinese, English, and others
- Voices: Multiple natural voices
- Speed: Adjustable (0.5x - 2.0x)
- Download:
  ```bash
  python -m mlx_lm.convert \
    --hf-path Qwen/Qwen3-TTS-0.6B \
    --output-dir ./models/qwen3-tts-0.6b-base
  ```

#### Qwen3-TTS-CustomVoice-0.6B (Optional)
- Size: 1.2GB
- Features: Voice cloning from reference audio
- Use case: Personalized voice synthesis
- Download:
  ```bash
  python -m mlx_lm.convert \
    --hf-path Qwen/Qwen3-TTS-CustomVoice-0.6B \
    --output-dir ./models/qwen3-tts-customvoice-0.6b
  ```

---

## Configuration

### Update Your Model

Edit `configs/config.yaml`:

```yaml
model:
  name: Qwen3.5-9B-Reasoning-Distilled
  path: /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-9b-reasoning-distilled-4bit
  runtime: mlx
  context_length: 32768

audio:
  asr_model_path: /Users/eitan/Documents/Projects/Python/Aster/models/qwen3-asr-0.6b
  tts_model_path: /Users/eitan/Documents/Projects/Python/Aster/models/qwen3-tts-0.6b-base
```

### Restart Aster

```bash
python scripts/ops/daemon.py restart
```

### Verify

```bash
curl http://127.0.0.1:8080/v1/models
```

---

## Performance Comparison

| Model | Size (4-bit) | Speed | Quality | Best For |
|-------|-------------|-------|---------|----------|
| Qwen3.5-1B | 1.5GB | ⚡⚡⚡ | ⭐⭐ | Edge, quick responses |
| Qwen3.5-9B | 5GB | ⚡⚡ | ⭐⭐⭐⭐ | General purpose, agents |
| Qwen3.5-Reasoning-Distilled-9B | 6GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Reasoning, agents |
| Qwen3.5-Coder-32B | 20GB | ⚡ | ⭐⭐⭐⭐⭐ | Code generation |
| QwQ-32B | 20GB | ⚡ | ⭐⭐⭐⭐⭐ | Deep reasoning |
| Qwen3.5-32B | 20GB | ⚡ | ⭐⭐⭐⭐⭐ | High quality |

---

## Advanced: Speculative Decoding (Draft Models)

If you enable speculative decoding in `configs/config.yaml`, you can use a smaller draft model for faster inference:

```yaml
speculative:
  enabled: true
  max_draft_tokens: 2
```

Then add a draft model:

```yaml
model:
  name: Qwen3.5-9B-Reasoning-Distilled
  path: ./models/qwen3.5-9b-reasoning-distilled-4bit
  draft_name: Qwen3.5-1B
  draft_path: ./models/qwen3.5-1b-instruct-4bit
```

**Note:** Speculative decoding is disabled by default because it often adds overhead in agent workloads. Enable only if benchmarks show improvement on your machine.

---

## Troubleshooting

### Model Download Fails
```bash
# Check internet connection
ping huggingface.co

# Try with explicit HF token
huggingface-cli login
python -m mlx_lm.convert --hf-path <model-id> --output-dir ./models/<name>
```

### Out of Memory
- Use 4-bit quantization: `--quantize 4bit`
- Try a smaller model (1B or 9B instead of 32B)
- Check available GPU memory: `python -c "import mlx.core as mx; print(mx.metal.get_peak_memory())"`

### Model Not Loading
- Verify path in `configs/config.yaml` is correct
- Check model directory contains `config.json` and `weights.safetensors`
- View logs: `python scripts/ops/daemon.py logs`

---

## Resources

- [MLX LM Documentation](https://github.com/ml-explore/mlx-lm)
- [Qwen Models on HuggingFace](https://huggingface.co/Qwen)
- [Aster Model Setup Guide](docs/MODEL_SETUP.md)
- [Model Download Architecture](docs/MODEL_DOWNLOAD_ARCHITECTURE.md)

---

## Contributing

Found a great model for Aster? Have performance benchmarks? Open an issue or PR to add it to this guide!
