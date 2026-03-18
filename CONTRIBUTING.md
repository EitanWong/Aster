# Contributing to Aster

Thank you for your interest in contributing to Aster! We welcome contributions from the community.

## Getting Started

### Prerequisites

- Python 3.12 or later (3.13 recommended)
- macOS with Apple Silicon (M1/M2/M3 or later)
- MLX framework installed

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aster.git
cd aster
```

2. Create a virtual environment:
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Download models (optional for development):
```bash
bash scripts/setup/download_models.sh
```

## Development Workflow

### Code Style

We use:
- **Ruff** for linting and import sorting
- **Pyright** for type checking (strict mode)
- **Black** for code formatting (via Ruff)

Run checks before committing:
```bash
make lint
make type-check
```

### Testing

Run tests with:
```bash
make test
```

Run tests with coverage:
```bash
make test-cov
```

### Project Structure

```
aster/
├── api/              # OpenAI-compatible API routes
├── scheduler/        # Request scheduling and batching
├── inference/        # Inference engines (prefill, decode)
├── cache/            # KV cache management
├── autotune/         # Policy selection and benchmarking
├── core/             # Configuration and lifecycle
├── telemetry/        # Logging and metrics
└── workers/          # Worker supervision
```

## Making Changes

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure tests pass:
```bash
make test
make lint
make type-check
```

3. Commit with clear messages:
```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve issue with X"
git commit -m "docs: update README"
```

4. Push to your fork and create a pull request

## Commit Message Convention

We follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Build, dependencies, etc.

Example:
```
feat: add continuous batching support

- Implement token-level batch formation
- Add batch resizing logic
- Update scheduler to manage active batches

Closes #123
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Follow the code style guidelines
5. Write a clear PR description explaining the changes

## Reporting Issues

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages
- Your hardware (M1/M2/M3, RAM, etc.)

## Performance Considerations

When contributing optimizations:
1. Measure before and after with benchmarks
2. Test on multiple hardware configurations if possible
3. Ensure stability is not compromised
4. Document the performance impact

## Questions?

Feel free to open an issue or discussion for questions about contributing.

Thank you for helping make Aster better!
