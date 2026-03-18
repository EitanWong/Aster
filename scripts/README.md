# Aster Scripts

This directory contains utility scripts for setup, operations, and development.

## Setup Scripts (`setup/`)

Scripts for initial setup and configuration:

- `download_models.sh` - Download MLX-converted models from Hugging Face
- `bootstrap.py` - Bootstrap script for initial setup
- `hfd.sh` - Hugging Face download utility
- `_common.sh` - Common shell utilities

Usage:
```bash
bash scripts/setup/download_models.sh
```

## Operations Scripts (`ops/`)

Scripts for running and managing the Aster service:

- `start.sh` - Start the Aster server
- `stop.sh` - Stop the Aster server
- `restart.sh` - Restart the Aster server
- `status.sh` - Check server status
- `health.sh` - Check server health
- `tail_logs.sh` - Tail server logs
- `smoke_test.sh` - Run smoke tests

Usage:
```bash
bash scripts/ops/start.sh
bash scripts/ops/status.sh
bash scripts/ops/stop.sh
```

## Development Scripts (`dev/`)

Scripts for development, benchmarking, and testing:

- `benchmark_live.py` - Run live benchmarks against running server
- `chat_cli.py` - Interactive chat CLI for testing
- `model_smoke.py` - Smoke test for model loading
- `benchmark.sh` - Benchmark script wrapper
- `chat.sh` - Chat script wrapper
- `metrics_summary.sh` - Print metrics summary
- `analyze_speculative.py` - Analyze speculative decoding performance

Usage:
```bash
python scripts/dev/chat_cli.py --config configs/config.yaml
python scripts/dev/benchmark_live.py --config configs/config.yaml
python scripts/dev/model_smoke.py --config configs/config.yaml
```

## Using Make

For convenience, use the Makefile:

```bash
make run              # Start the server
make benchmark        # Run benchmarks
make test             # Run tests
make lint             # Run linting
make type-check       # Run type checking
```

See `Makefile` for all available commands.
