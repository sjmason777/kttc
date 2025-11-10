# KTTC - Knowledge Translation Transmutation Core

**Transforming translations into gold-standard quality**

> Autonomous multi-agent platform with 90% cost reduction and 1000x speed improvement

[![CI](https://github.com/kttc-ai/kttc/workflows/CI/badge.svg)](https://github.com/kttc-ai/kttc/actions)
[![codecov](https://codecov.io/gh/kttc-ai/kttc/branch/main/graph/badge.svg)](https://codecov.io/gh/kttc-ai/kttc)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-ruff-red)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/kttc-ai/kttc)

---

## 🎯 Overview

KTTC (Knowledge Translation Transmutation Core) is an autonomous translation quality assurance platform. It uses multi-agent AI systems to transmute raw translations into gold-standard quality through automated detection and validation of translation issues.

**Key Features:**
- 🤖 **Multi-agent QA** - 7 specialized agents for different quality aspects
- 📊 **MQM Scoring** - Industry-standard quality metrics
- ⚡ **90% cost reduction** vs manual review
- 🚀 **100-1000x faster** than human evaluation
- 🔄 **CI/CD native** - GitHub Actions integration
- 🎯 **95+ MQM target** - Production-grade quality

---

## 🚀 Quick Start

### Installation

```bash
pip install kttc
```

### Basic Usage

```bash
# Check translation quality
kttc check \
  --source source.txt \
  --translation translation.txt \
  --source-lang en \
  --target-lang es \
  --threshold 95

# Output:
# ✅ MQM Score: 96.5 (PASS)
# ⚠️  2 minor issues found
```

### GitHub Actions Integration

Add automated translation quality checks to your CI/CD pipeline:

```yaml
name: Translation QA

on:
  pull_request:
    paths: ['translations/**']

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: your-org/kttc@v1
        with:
          source-dir: 'translations/en'
          translation-dir: 'translations/es'
          source-lang: 'en'
          target-lang: 'es'
          threshold: '95.0'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

See [GitHub Actions Documentation](docs/github-actions.md) for more examples.

---

## 🏗️ Architecture

```
CLI (Typer) → Orchestrator → [Accuracy, Fluency, Terminology] → Synthesis
                 ↓
            MQM Scorer → Report (JSON/Markdown)
```

**Tech Stack:**
- Python 3.11+
- Typer (CLI framework)
- pytest + pytest-asyncio (testing)
- Ruff (linting)
- OpenAI/Anthropic (LLM providers)

---

## 🛠️ Development

### Setup

```bash
# Clone repository
git clone git@github.com:kttc-ai/kttc.git
cd kttc

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Project Structure

```
kttc/
├── src/
│   └── kttc/              # Main package
│       ├── cli/           # CLI interface
│       ├── agents/        # QA agents
│       ├── core/          # Core logic
│       ├── llm/           # LLM providers
│       └── utils/         # Utilities
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
├── examples/
└── pyproject.toml
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Quick Start for Contributors:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run `pre-commit run --all-files` and `pytest`
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

### Reporting Issues

Found a bug or have a feature request? Please check our [issue tracker](https://github.com/kttc-ai/kttc/issues)

### Security

For security vulnerabilities, please see our [Security Policy](SECURITY.md)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Last Updated:** November 10, 2025
