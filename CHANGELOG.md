# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-10

### Added

**Core Features:**
- Multi-agent translation quality assessment system
- MQM (Multidimensional Quality Metrics) scoring engine
- Three specialized QA agents:
  - Accuracy Agent - semantic accuracy evaluation
  - Fluency Agent - grammatical fluency evaluation
  - Terminology Agent - terminology consistency evaluation
- Agent orchestrator with parallel execution support

**CLI Commands:**
- `kttc check` - Single file quality assessment
- `kttc batch` - Batch processing with configurable parallelism
- `kttc report` - Detailed quality report generation
- Multiple output formats: text, JSON, Markdown, HTML

**LLM Provider Support:**
- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic (Claude 3)
- Yandex (YandexGPT)
- GigaChat

**CI/CD Integration:**
- GitHub Actions workflow for translation QA
- Composite action for reusable workflows
- Example workflows for common scenarios
- Automated PyPI publishing with Trusted Publishers

**Documentation:**
- Comprehensive User Guide
- Developer Guide with contribution guidelines
- API Documentation
- Usage examples (basic, batch processing)
- Performance benchmarking suite

**Quality & Testing:**
- 294 unit tests
- 99.9% code coverage
- Strict type checking with mypy
- Code formatting with Black
- Linting with Ruff
- Pre-commit hooks

### Performance
- Processing speed: < 0.2s per translation (with mocked LLM)
- Parallel speedup: 3.0x (vs sequential)
- Throughput: 49+ docs/sec in batch mode
- MQM scoring: 21,000+ ops/sec

### Technical Details
- Python 3.11+ required
- Async/await architecture for parallel processing
- Pydantic v2 for data validation
- Rich terminal UI with progress bars
- Type-safe with comprehensive type hints

---

**Note:** This is an alpha release. API may change in future versions.
