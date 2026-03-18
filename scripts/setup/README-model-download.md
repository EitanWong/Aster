# Aster model download instructions

Use `scripts/download_models.sh` to prepare the local environment and manually download the two MLX model directories Aster expects.

## Mirror support

The script now defaults to:

```bash
HF_ENDPOINT=https://hf-mirror.com
```

This matches the hf-mirror guidance and helps accelerate Hugging Face downloads in mainland China.

## What the script now does automatically

- checks that it is running on macOS
- checks project structure
- checks for Homebrew and installs it if missing
- checks for Python 3.13 / 3.12+ and installs Python 3.13 with Homebrew if needed
- creates or reuses the project `.venv`
- upgrades pip/setuptools/wheel in the venv
- installs Hugging Face CLI tooling if missing
- supports both `hf` and `huggingface-cli`
- can optionally install `aria2`
- can optionally use `hfd` for more robust accelerated downloads
- creates model directories
- updates `configs/config.yaml` with intended local model paths
- downloads the main and draft MLX model repositories

## Default target directories

- `/Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-9b-mlx`
- `/Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx`

## Default repo ids in the script

- `mlx-community/Qwen3.5-9B-4bit`
- `mlx-community/Qwen3.5-0.8B-4bit`

## Simplest command

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
bash scripts/download_models.sh
```

## Optional: use hfd + aria2 path

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
USE_HFD=1 bash scripts/download_models.sh
```

## Optional: override repo ids

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
MAIN_REPO="your-main-repo" DRAFT_REPO="your-draft-repo" bash scripts/download_models.sh
```

## Optional: pass a token for gated repos

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
HF_TOKEN="hf_your_token" bash scripts/download_models.sh
```

## Manual login if needed

Depending on the installed CLI, use one of:

```bash
source .venv/bin/activate
hf auth login
```

or:

```bash
source .venv/bin/activate
huggingface-cli login
```

## After download

Run:

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
source .venv/bin/activate
python scripts/model_smoke.py --config configs/config.yaml
python scripts/benchmark_live.py --config configs/config.yaml
```
