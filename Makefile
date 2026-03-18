.PHONY: help install install-dev test test-cov lint type-check format clean run benchmark

help:
	@echo "Aster Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install aster package"
	@echo "  make install-dev      Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make lint             Run linting checks (ruff)"
	@echo "  make type-check       Run type checking (pyright)"
	@echo "  make format           Format code with ruff"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Operations:"
	@echo "  make run              Run the server"
	@echo "  make benchmark        Run benchmark suite"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove build artifacts and cache"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=aster --cov-report=html --cov-report=term-missing

lint:
	ruff check aster tests scripts

type-check:
	pyright aster tests

format:
	ruff check --fix aster tests scripts
	ruff format aster tests scripts

run:
	python -m aster --config configs/config.yaml

benchmark:
	python scripts/dev/benchmark_live.py --config configs/config.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
