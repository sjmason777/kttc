# KTTC - Translation Quality Assurance Platform

**Autonomous multi-agent platform for translation quality assurance**

> "Strix для переводов" - Automated quality checking with 90% cost reduction

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/kttc-ai/kttc)

---

## 🎯 Overview

KTTC is an autonomous translation quality assurance platform inspired by [Strix](https://github.com/usestrix/strix). It uses multi-agent AI systems to automatically detect and validate translation quality issues.

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

### In GitHub Actions

```yaml
- name: Translation QA
  run: |
    kttc check-pr --base main --threshold 95
```

---

## 📚 Documentation

**Full documentation:** [kttc-ai/docs](https://github.com/kttc-ai/docs) (private)

**Quick links:**
- [Development Plan](https://github.com/kttc-ai/docs) - 12-week MVP roadmap
- [Best Practices](https://github.com/kttc-ai/docs) - Modern Python/CLI practices
- [Architecture](https://github.com/kttc-ai/docs) - Multi-agent design

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

## 📊 Status

**Phase:** MVP Development (Week 1 of 12)

- [x] Research (18+ arXiv papers)
- [x] Design (architecture defined)
- [ ] **MVP Development** ← Current (12 weeks)
- [ ] Testing (WMT benchmarks)
- [ ] Production release

---

## 🤝 Contributing

This is currently a private project in active development.

For questions or suggestions, contact: dev@kttc.ai

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Inspired by:**
- [Strix](https://github.com/usestrix/strix) - Autonomous security testing
- [MAATS](https://arxiv.org/abs/2505.14848) - Multi-agent translation
- [Andrew Ng's Translation Agent](https://github.com/andrewyng/translation-agent)

**Built with:**
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal output
- [COMET](https://github.com/Unbabel/COMET) - Translation metrics

---

**Last Updated:** November 10, 2025
