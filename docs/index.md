# KTTC - Knowledge Translation Transmutation Core

[![CI](https://github.com/kttc-ai/kttc/workflows/CI/badge.svg)](https://github.com/kttc-ai/kttc/actions)
[![PyPI](https://img.shields.io/pypi/v/kttc)](https://pypi.org/project/kttc/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**Autonomous AI-powered translation quality assurance**

KTTC uses specialized multi-agent systems to automatically detect, analyze, and fix translation quality issues following the industry-standard MQM (Multidimensional Quality Metrics) framework. Get production-ready translation quality in seconds, not hours.

---

## Key Features

- **Multi-Agent QA System** - 5 specialized agents analyze accuracy, fluency, terminology, hallucinations, and context
- **MQM Scoring** - Industry-standard quality metrics used in WMT benchmarks
- **Smart Routing** - Automatically selects optimal models based on text complexity (60% cost savings)
- **Auto-Correction** - LLM-powered error fixing with iterative refinement (TEaR loop)
- **Language-Specific Agents** - Native-level checks for English, Chinese, and Russian
- **Translation Memory** - Semantic search with quality tracking and reuse
- **Glossary Management** - Custom terminology validation and consistency
- **Batch Processing** - Process thousands of translations in parallel
- **CI/CD Ready** - GitHub Actions integration, exit codes, multiple output formats
- **Multi-LLM Support** - OpenAI, Anthropic, GigaChat, YandexGPT

**Performance:** 90% cost reduction vs manual review • 100-1000x faster • 95+ MQM quality target

---

## Quick Example

```bash
# Install
pip install kttc

# Set API key
export KTTC_OPENAI_API_KEY="sk-..."

# Check translation quality
kttc check source.txt translation.txt --source-lang en --target-lang es
```

**Output:**

```
✅ MQM Score: 96.5 (PASS - Excellent Quality)
📊 5 agents analyzed translation
⚠️  Found 2 minor issues, 0 major, 0 critical
✓ Quality threshold met (≥95.0)
```

---

## Getting Started

Ready to get started? Check out these pages:

- [Installation](installation.md) - Install KTTC and set up your environment
- [Quick Start](quickstart.md) - Get up and running in 5 minutes
- [CLI Usage](cli-usage.md) - Learn all the CLI commands
- [Configuration](configuration.md) - Configure KTTC for your needs

---

## Python API Example

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

---

## Links

- [GitHub Repository](https://github.com/kttc-ai/kttc)
- [PyPI Package](https://pypi.org/project/kttc/)
- [Issue Tracker](https://github.com/kttc-ai/kttc/issues)
- [Discussions](https://github.com/kttc-ai/kttc/discussions)

---

## License

Licensed under the Apache License 2.0. See [LICENSE](https://github.com/kttc-ai/kttc/blob/main/LICENSE) for details.

Copyright 2025 KTTC AI
