# Development Guide

This guide covers development setup, testing, and contribution workflows for Aster.

## Quick Start

### Prerequisites

- Python 3.12+ (3.13 recommended)
- macOS with Apple Silicon
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aster.git
cd aster
```

2. Create and activate virtual environment:
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

3. Install development dependencies:
```bash
make install-dev
```

4. Download models (optional):
```bash
bash scripts/setup/download_models.sh
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_health_endpoint -v
```

### Code Quality

```bash
# Check linting
make lint

# Check types
make type-check

# Format code
make format
```

### Running the Server

```bash
# Start server with default config
make run

# Start with custom config
python -m aster --config configs/config.yaml
```

### Benchmarking

```bash
# Run benchmarks
make benchmark

# Or directly
python scripts/dev/benchmark_live.py --config configs/config.yaml
```

### Interactive Testing

```bash
# Chat CLI
python scripts/dev/chat_cli.py --config configs/config.yaml

# Model smoke test
python scripts/dev/model_smoke.py --config configs/config.yaml
```

## Project Structure

```
aster/
├── api/              # OpenAI-compatible API
├── scheduler/        # Request scheduling
├── inference/        # Inference engines
├── cache/            # KV cache management
├── autotune/         # Policy selection
├── core/             # Configuration and lifecycle
├── telemetry/        # Logging and metrics
└── workers/          # Worker supervision

tests/
├── test_api.py       # API tests
├── test_cache.py     # Cache tests
├── test_config.py    # Configuration tests
├── test_scheduler.py # Scheduler tests
└── test_speculative_tokens.py  # Speculative decoding tests

scripts/
├── setup/            # Setup scripts
├── ops/              # Operations scripts
└── dev/              # Development scripts
```

## Key Modules

### api/
Handles OpenAI-compatible API endpoints:
- `routes.py` - FastAPI route definitions
- `schemas.py` - Pydantic request/response schemas
- `streaming.py` - Server-sent events streaming

### scheduler/
Manages request scheduling and batching:
- `scheduler.py` - Main scheduler loop
- `adaptive_batcher.py` - Dynamic batch formation
- `policy_engine.py` - Policy selection logic

### inference/
Handles model inference:
- `engine.py` - Main inference orchestrator
- `prefill_engine.py` - Prompt processing
- `decode_engine.py` - Token generation
- `speculative.py` - Speculative decoding control
- `mlx_runtime.py` - MLX backend integration

### cache/
Manages KV cache:
- `prefix_cache.py` - Prefix reuse cache
- `paged_kv_cache.py` - Paged KV allocation
- `cache_keys.py` - Cache key generation
- `eviction.py` - Eviction policies

### core/
Core infrastructure:
- `config.py` - Configuration management
- `lifecycle.py` - Application lifecycle
- `errors.py` - Custom exceptions

### telemetry/
Observability:
- `logging.py` - Structured logging
- `metrics.py` - Prometheus metrics

## Testing Guidelines

### Writing Tests

1. Place tests in `tests/` directory
2. Name test files `test_*.py`
3. Name test functions `test_*`
4. Use descriptive names
5. Add docstrings

Example:
```python
def test_prefix_cache_hit():
    """Test that prefix cache returns stored values."""
    cache = PrefixCache(CacheSettings(), MetricsRegistry("test"))
    cache.store("model_id", [1, 2, 3], [4, 5], 128)
    hit = cache.lookup("model_id", [1, 2, 3])
    assert hit is not None
```

### Test Coverage

Aim for >80% coverage. Check coverage with:
```bash
make test-cov
```

## Code Style

### Imports

- Use absolute imports: `from aster.core.config import ...`
- Group imports: stdlib, third-party, local
- Sort alphabetically within groups

### Type Hints

- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- Use `|` for unions (Python 3.10+)

Example:
```python
from __future__ import annotations

def process_tokens(tokens: list[int], max_length: int = 100) -> dict[str, int]:
    """Process tokens and return statistics."""
    return {"count": len(tokens)}
```

### Docstrings

- Use Google-style docstrings
- Include description, args, returns, raises

Example:
```python
def calculate_hash(tokens: list[int]) -> str:
    """Calculate deterministic hash of token sequence.
    
    Args:
        tokens: List of token IDs
        
    Returns:
        Hex string hash of the tokens
        
    Raises:
        ValueError: If tokens list is empty
    """
```

## Performance Considerations

### Profiling

Use Python's built-in profiler:
```bash
python -m cProfile -s cumtime scripts/dev/benchmark_live.py --config configs/config.yaml
```

### Benchmarking

Run benchmarks to measure performance:
```bash
make benchmark
```

### Memory Usage

Monitor memory with:
```bash
python -c "import psutil; print(psutil.virtual_memory())"
```

## Debugging

### Logging

Enable debug logging in config:
```yaml
logging:
  level: DEBUG
```

### Print Debugging

Use structured logging instead of print:
```python
from aster.telemetry.logging import get_logger

logger = get_logger(__name__)
logger.debug("Debug message", extra={"key": "value"})
```

### Breakpoints

Use Python debugger:
```python
import pdb; pdb.set_trace()
```

Or use IDE debugger (VS Code, PyCharm, etc.)

## Common Issues

### Import Errors

If you get import errors, ensure:
1. Virtual environment is activated
2. Package is installed: `pip install -e .`
3. Python path includes project root

### Test Failures

If tests fail:
1. Check Python version: `python --version` (should be 3.12+)
2. Reinstall dependencies: `pip install -e ".[dev]"`
3. Clear cache: `make clean`

### Type Checking Errors

If pyright reports errors:
1. Ensure all imports are correct
2. Check type hints are valid
3. Run: `pyright --outputjson` for detailed errors

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pytest Documentation](https://docs.pytest.org/)
