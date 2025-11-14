# KTTC Documentation

Welcome to the KTTC documentation! This documentation follows the [Di\u00e1taxis](https://diataxis.fr/) framework for clear, organized technical documentation.

## Documentation Structure

```
docs/
├── en/           # English (primary)
├── ru/           # Russian (translations)
└── zh/           # Chinese (translations)
```

## English Documentation

### 📚 [Tutorials](en/tutorials/README.md) - Learning-oriented

*New to KTTC? Start here!*

Step-by-step guides for beginners:

- [Quickstart Guide](en/tutorials/quickstart.md) - Your first quality check (5 min)
- [First Batch Processing](en/tutorials/first-batch.md) - Process multiple translations
- [Translation with Auto-Correction](en/tutorials/translation-with-qa.md) - AI translation + QA

### 📖 [Guides](en/guides/README.md) - Task-oriented

*How to accomplish specific goals.*

Practical how-to guides:

- [Installation](en/guides/installation.md) - Install KTTC and dependencies
- [CLI Usage](en/guides/cli-usage.md) - Command-line interface guide
- [Batch Processing](en/guides/batch-processing.md) - Process many translations
- [Auto-Correction](en/guides/auto-correction.md) - Fix errors automatically
- [Glossary Management](en/guides/glossary-management.md) - Custom terminology
- [Smart Routing](en/guides/smart-routing.md) - Optimize costs
- [Troubleshooting](en/guides/troubleshooting.md) - Common issues

### 📋 [Reference](en/reference/README.md) - Information-oriented

*Technical details and specifications.*

Complete reference documentation:

- [CLI Commands](en/reference/cli-commands.md) - All commands and options
- [API Reference](en/reference/api-reference.md) - Python API
- [Configuration](en/reference/configuration.md) - Settings reference
- [MQM Scoring](en/reference/mqm-scoring.md) - Quality metrics spec
- [File Formats](en/reference/batch-formats.md) - CSV, JSON, JSONL

### 💡 [Explanation](en/explanation/README.md) - Understanding-oriented

*Concepts, architecture, and design.*

Conceptual documentation:

- [Architecture Overview](en/explanation/architecture.md) - System design
- [Agent System](en/explanation/agent-system.md) - Multi-agent QA
- [MQM Framework](en/explanation/mqm-scoring.md) - Quality metrics
- [Smart Routing](en/explanation/smart-routing-explained.md) - Model selection

## Translations

### 🇷🇺 Russian Documentation

- [Tutorials (Руководства)](ru/tutorials/README.md)
- [Guides (Инструкции)](ru/guides/README.md)
- [Reference (Справочник)](ru/reference/README.md)
- [Explanation (Объяснения)](ru/explanation/README.md)

### 🇨🇳 Chinese Documentation

- [Tutorials (教程)](zh/tutorials/README.md)
- [Guides (指南)](zh/guides/README.md)
- [Reference (参考)](zh/reference/README.md)
- [Explanation (说明)](zh/explanation/README.md)

**Note:** Translations are works in progress. English documentation is the primary source.

## Quick Links

- 🏠 [Main README](../README.md) - Project overview
- 📦 [Installation](en/guides/installation.md) - Get started
- 🚀 [Quickstart](en/tutorials/quickstart.md) - First steps
- 🐛 [Troubleshooting](en/guides/troubleshooting.md) - Common issues
- 💬 [GitHub Issues](https://github.com/kttc-ai/kttc/issues) - Report bugs

## Contributing to Documentation

We welcome documentation contributions!

- **Found an error?** Open an issue or submit a PR
- **Want to translate?** See [Translation Guide](development/translation-guide.md)
- **Adding features?** Update relevant docs in your PR

**Documentation Standards:**

- All code samples must be tested and verified against actual code
- Use clear, concise language
- Include examples for every feature
- Keep English docs synchronized with codebase
- Translate to ru/zh after English docs are stable

## Diátaxis Framework

This documentation uses the [Diátaxis framework](https://diataxis.fr/):

| Type | Oriented to | Purpose | Example |
|------|-------------|---------|---------|
| **Tutorials** | Learning | Acquire skills | "Your first quality check" |
| **Guides** | Tasks | Solve problems | "How to batch process files" |
| **Reference** | Information | Look up facts | "kttc check command options" |
| **Explanation** | Understanding | Gain knowledge | "Why multi-agent architecture?" |

**When to use each:**

- 📚 **New to KTTC?** → Start with Tutorials
- 🎯 **Have a specific goal?** → Check Guides
- 🔍 **Need exact details?** → Search Reference
- 💭 **Want to understand?** → Read Explanation

## License

Documentation is licensed under [Apache License 2.0](../LICENSE).

Copyright 2025 KTTC AI (https://github.com/kttc-ai)
