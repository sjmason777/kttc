<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/content/assets/img/kttc.logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/content/assets/img/kttc.logo.png">
      <img alt="KTTC" title="KTTC" src="docs/content/assets/img/kttc.logo.png">
    </picture>
</p>

**English** · [Русский](README.ru.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [فارسی](README.fa.md)

# KTTC - Knowledge Translation Transmutation Core

[![CI](https://github.com/kttc-ai/kttc/workflows/CI/badge.svg)](https://github.com/kttc-ai/kttc/actions)
[![CodeQL](https://github.com/kttc-ai/kttc/workflows/CodeQL/badge.svg)](https://github.com/kttc-ai/kttc/security/code-scanning)
[![PyPI](https://img.shields.io/pypi/v/kttc)](https://pypi.org/project/kttc/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**Autonomous AI-powered translation quality assurance**

KTTC uses specialized multi-agent systems to automatically detect, analyze, and fix translation quality issues following the industry-standard MQM (Multidimensional Quality Metrics) framework. Get production-ready translation quality in seconds, not hours.

---

## Key Features

### Core Analysis
- **Multi-Agent QA System** - Specialized agents analyze accuracy, fluency, terminology, style, hallucination, and context
- **MQM Scoring** - Industry-standard quality metrics used in WMT benchmarks
- **Language-Specific Agents** - 5 native-level fluency agents for English, Chinese, Russian, Hindi, and Persian
- **60+ Domain Glossaries** - Automotive, customs, financial, legal, logistics, medical + language traps

### Smart Agent Orchestration
- **Weighted Consensus** - Agents with different trust weights, confidence scoring, agreement metrics
- **Multi-Agent Debate** - Cross-verification between agents reduces false positives by 30-50%
- **Dynamic Agent Selection** - Auto-selects 2-5 agents based on text complexity (30-50% cost savings)
- **Agent Presets** - `minimal` (2 agents), `default` (3), `full` (5) for quick configuration
- **Self-Assessment Retry** - Agents evaluate their confidence and retry on low certainty

### MQM Profiles System
- **Built-in Profiles** - default, strict, minimal, legal, medical, marketing, literary, technical
- **YAML Custom Profiles** - Define agent selection, weights, severity multipliers, thresholds
- **Domain Adaptation** - Medical: 98% threshold + hallucination agent; Legal: terminology focus
- **Quality Gates** - Configurable pass/fail thresholds per domain (88-98%)

### Literary Style Analysis
- **StyleFingerprint** - Automatic detection of literary style patterns (Burrows Delta method)
- **Style Preservation Agent** - Evaluates authorial voice preservation in literary translations
- **Country-Specific Patterns** - Leskov skaz, Platanov pleonasms, Joyce stream, Hafez ghazals, Chhayavad poetry
- **Fluency Tolerance** - Adjustable fluency weights for intentional stylistic deviations

### Language Intelligence
- **Language Traps Detection** - 60+ glossaries detect homophones, false friends, idioms, phrasal verbs, paronyms
- **Self-Check / Proofreading** - Grammar, spelling, punctuation checking using school curriculum rules
- **School Curricula** - FGOS (Russia), UK GPS (Britain), NCERT (India), PEP (China), Iranian grammar
- **Fast Lint Mode** - Rule-based checking without LLM, ideal for CI/CD and pre-commit hooks
- **Auto-Correction** - LLM-powered error fixing with iterative refinement (TEaR loop)

### Enterprise Infrastructure
- **Smart Routing** - Automatically selects optimal models based on text complexity (60% cost savings)
- **XLSX Export** - Excel reports with Summary, Errors, Breakdown sheets for enterprise reporting
- **Translation Memory** - Semantic search with quality tracking and reuse
- **Glossary Management** - Custom terminology + 60+ built-in multilingual glossaries
- **Batch Processing** - Process thousands of translations in parallel with CSV/JSON filtering
- **CI/CD Ready** - GitHub Actions integration, exit codes, JSON/Markdown/HTML/XLSX output
- **Multi-LLM Support** - OpenAI, Anthropic, GigaChat, YandexGPT with LanguageTool integration
- **Usage Analytics** - Token counts, API costs, and call statistics in reports

**Performance:** 90% cost reduction vs manual review • 100-1000x faster • 95+ MQM quality target

---

## Try KTTC Online

Experience KTTC without installation:

[![Open in Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kttc-ai/kttc/blob/main/examples/kttc_demo.ipynb)
[![Streamlit Demo](https://img.shields.io/badge/Streamlit_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://kttc-demo.streamlit.app)
[![Open in Codespaces](https://img.shields.io/badge/Open_in_Codespaces-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/codespaces/new?repo=kttc-ai/kttc)

- **Google Colab** - Interactive tutorial with examples (5 min, no setup)
- **Streamlit Demo** - Web UI to test your own translations (no code required)
- **GitHub Codespaces** - Full dev environment in browser (for contributors)

---

## Quick Start

### 1. Install

```bash
pip install kttc
```

Optional language enhancements:

```bash
pip install kttc[english]        # English: LanguageTool (5,000+ grammar rules)
pip install kttc[chinese]        # Chinese: HanLP (measure words, particles)
pip install kttc[hindi]          # Hindi: Indic NLP + Stanza + Spello
pip install kttc[persian]        # Persian: DadmaTools (spaCy-based)
pip install kttc[all-languages]  # All language helpers
```

### 2. Set API Key

```bash
export KTTC_OPENAI_API_KEY="sk-..."
# or
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Check Translation Quality

```bash
kttc check source.txt translation.txt --source-lang en --target-lang es
```

**Output:**

```
✅ MQM Score: 96.5 (PASS - Excellent Quality)
📊 5 agents analyzed translation
⚠️  Found 2 minor issues, 0 major, 0 critical
✓ Quality threshold met (≥95.0)
```

That's it! KTTC works out of the box with smart defaults:
- ✅ Smart routing (auto-selects cheaper models for simple texts)
- ✅ Auto-glossary (uses 'base' glossary if exists)
- ✅ Auto-format (detects output format from file extension)

---

## Commands

KTTC provides a unified CLI with smart auto-detection:

```bash
kttc check source.txt translation.txt          # Single quality check
kttc check source.txt t1.txt t2.txt t3.txt     # Auto-compares multiple translations
kttc check translations.csv                     # Auto-detects batch mode (CSV/JSON)
kttc check source_dir/ trans_dir/              # Auto-detects directory batch mode

kttc batch --file translations.csv              # Explicit batch processing
kttc compare --source src.txt -t t1 -t t2      # Compare translations side-by-side
kttc translate --text "Hello" --source-lang en --target-lang es  # Translate with QA
kttc benchmark --source text.txt --providers openai,anthropic    # Benchmark LLMs

# Glossary management (project + user global storage)
kttc glossary list                              # List all glossaries
kttc glossary create tech --from-csv terms.csv  # Create project glossary
kttc glossary create personal --from-csv my.csv --user  # Create user glossary

# 🥚 Self-check / Proofreading (new!)
kttc check article.md --self --lang ru          # Proofread without translation
kttc proofread article.md --lang ru             # Same as above (alias)
kttc lint article.md --lang ru --fix            # Quick rule-based check (no LLM)
```

**See full command reference:** [docs/en/reference/cli-commands.md](docs/en/reference/cli-commands.md)

---

## Python API

```python
import asyncio
from kttc.agents import AgentOrchestrator
from kttc.llm import OpenAIProvider
from kttc.core import TranslationTask

async def check_quality():
    llm = OpenAIProvider(api_key="your-key")
    orchestrator = AgentOrchestrator(llm)

    task = TranslationTask(
        source_text="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es",
    )

    report = await orchestrator.evaluate(task)
    print(f"MQM Score: {report.mqm_score}")
    print(f"Status: {report.status}")

asyncio.run(check_quality())
```

**See full API reference:** [docs/en/reference/api-reference.md](docs/en/reference/api-reference.md)

---

## Documentation

📚 **Complete documentation is available in [docs/](docs/)**

### Quick Links

- **[Quickstart Guide](docs/en/tutorials/quickstart.md)** - Get started in 5 minutes
- **[Installation Guide](docs/en/guides/installation.md)** - Detailed setup instructions
- **[CLI Reference](docs/en/reference/cli-commands.md)** - All commands and options
- **[Architecture](docs/en/explanation/architecture.md)** - How KTTC works
- **[Language Features](docs/en/explanation/language-features-explained.md)** - English/Chinese/Russian specialization

### Documentation Structure

Following the [Diátaxis](https://diataxis.fr/) framework:

- 📚 **[Tutorials](docs/en/tutorials/README.md)** - Learn by doing (step-by-step guides)
- 📖 **[Guides](docs/en/guides/README.md)** - Solve specific problems (how-to guides)
- 📋 **[Reference](docs/en/reference/README.md)** - Look up technical details (API, CLI)
- 💡 **[Explanation](docs/en/explanation/README.md)** - Understand concepts (architecture, design)

### Translations

**Languages:** 🇺🇸 [English](docs/en/) (primary) · 🇷🇺 [Русский](README.ru.md) · 🇨🇳 [中文](README.zh.md) · 🇮🇳 [हिन्दी](README.hi.md) · 🇮🇷 [فارسی](README.fa.md)

Full documentation available in:
- 🇺🇸 **[English](docs/en/README.md)** - Complete (primary source)
- 🇷🇺 **[Русский](docs/ru/README.md)** - Complete
- 🇨🇳 **[中文](docs/zh/README.md)** - Complete
- 🇮🇳 **[हिन्दी](docs/hi/README.md)** - Complete
- 🇮🇷 **[فارسی](docs/fa/README.md)** - Complete

---

## Development

### Setup

```bash
git clone https://github.com/kttc-ai/kttc.git
cd kttc
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Quality Standards

- **Type Checking:** mypy --strict
- **Formatting:** black (line length: 100)
- **Linting:** ruff
- **Testing:** pytest with asyncio support

```bash
# Run all checks
pre-commit run --all-files
pytest --cov=kttc
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick start:**
1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run quality checks: `pre-commit run --all-files && pytest`
5. Submit a pull request

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md). Do not open public issues for security concerns.

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2025 KTTC AI (https://github.com/kttc-ai)

---

## Citation

If you use KTTC in your research:

```bibtex
@software{kttc2025,
  title = {KTTC: Knowledge Translation Transmutation Core},
  author = {KTTC AI},
  year = {2025},
  url = {https://github.com/kttc-ai/kttc},
  version = {0.1.0}
}
```

---

## Links

- 📦 [PyPI Package](https://pypi.org/project/kttc/)
- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/kttc-ai/kttc/issues)
- 💬 [Discussions](https://github.com/kttc-ai/kttc/discussions)
- 🔒 [Security Policy](SECURITY.md)
