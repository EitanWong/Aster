# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release preparation
- Project structure reorganization for open source
- Comprehensive documentation
- CI/CD workflows
- Contributing guidelines

### Changed
- Restructured codebase into `aster/` package
- Updated imports to use absolute paths with `aster` prefix
- Enhanced pyproject.toml with full project metadata

### Fixed
- Import path consistency across modules

## [0.1.0] - 2026-03-18

### Added
- Core inference engine with MLX backend
- OpenAI-compatible API endpoints
- Adaptive scheduler with queue-aware batching
- Paged KV cache management
- Automatic prefix cache with deterministic hashing
- Speculative decoding controller with auto-disable fallback
- Benchmark/autotune subsystem for policy selection
- Structured logging and Prometheus metrics
- Worker supervision with heartbeat monitoring
- Health and readiness checks
- Request tracing and telemetry
- Support for long-context and agent-style workloads
- Configuration management with YAML
- Model loading from local paths or Hugging Face

### Features
- **Prefill/Decode Split**: Explicit separation of prompt ingestion and token generation
- **Adaptive Batching**: Dynamic batch window sizing based on queue depth
- **Prefix Caching**: Deterministic hashing for repeated prompt prefixes
- **Speculative Decoding**: Optional draft-verify pipeline with fallback
- **Policy Selection**: Benchmark-driven optimization profile selection
- **Streaming Support**: Token-level streaming with configurable flush cadence
- **Apple Silicon Optimized**: Tuned for MLX on M1/M2/M3 and later

### Documentation
- Architecture overview
- OpenAI compatibility guide
- Debugging guide
- Operations guide
- Long-term roadmap

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes to API or core behavior
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Future Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the long-term architectural evolution plan.
