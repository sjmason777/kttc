# KTTC v0.1.0 - Alpha Release

**Released:** November 10, 2025

We're excited to announce the first alpha release of KTTC (Knowledge Translation Transmutation Core) - an autonomous translation quality assurance platform powered by multi-agent AI.

## 🎯 What is KTTC?

KTTC transforms translation quality assessment by using specialized AI agents to automatically detect and evaluate translation issues. It provides:

- **90% cost reduction** vs manual review
- **100-1000x faster** than human evaluation
- **95+ MQM scores** for production-grade quality
- **Multi-agent architecture** with parallel processing

## 🚀 Key Features

### Multi-Agent QA System

Three specialized agents work in parallel:
- **Accuracy Agent** - Evaluates semantic accuracy and meaning preservation
- **Fluency Agent** - Checks grammar, naturalness, and readability
- **Terminology Agent** - Validates consistent terminology usage

### CLI Interface

```bash
# Check single translation
kttc check --source source.txt --translation translation.txt \
  --source-lang en --target-lang es --threshold 95

# Batch process multiple files
kttc batch --source-dir ./en/ --translation-dir ./es/ \
  --source-lang en --target-lang es --parallel 5

# Generate detailed report
kttc report --source source.txt --translation translation.txt \
  --output report.html --format html
```

### LLM Provider Support

- ✅ OpenAI (GPT-4, GPT-3.5-turbo)
- ✅ Anthropic (Claude 3)
- ✅ Yandex (YandexGPT)
- ✅ GigaChat

### CI/CD Integration

Automate translation quality checks in your pipeline:

```yaml
- uses: kttc-ai/kttc@v0.1.0
  with:
    source-dir: 'translations/en'
    translation-dir: 'translations/es'
    source-lang: 'en'
    target-lang: 'es'
    threshold: '95.0'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## 📊 Performance

- **Processing speed:** < 0.2s per translation (with mocked LLM)
- **Parallel speedup:** 3.0x vs sequential execution
- **Throughput:** 49+ documents/second in batch mode
- **MQM scoring:** 21,000+ operations/second

## 📦 Installation

```bash
pip install kttc
```

**Requirements:**
- Python 3.11+
- OpenAI or Anthropic API key

## 📚 Documentation

- [User Guide](docs/guides/user-guide.md) - Getting started and usage
- [Developer Guide](docs/development/developer-guide.md) - Contributing
- [API Documentation](docs/api/README.md) - Programmatic usage
- [Examples](examples/) - Code examples

## 🧪 Quality Metrics

- ✅ **294 unit tests** passing
- ✅ **99.9% code coverage**
- ✅ **Strict type checking** with mypy
- ✅ **Formatted** with Black
- ✅ **Linted** with Ruff

## 🎁 What's Included

**Core Components:**
- Multi-agent orchestrator with async/await
- MQM (Multidimensional Quality Metrics) scoring engine
- LLM provider abstraction layer
- Pydantic v2 models for type safety

**CLI Commands:**
- `kttc check` - Single file assessment
- `kttc batch` - Batch processing
- `kttc report` - Report generation
- Multiple output formats: text, JSON, Markdown, HTML

**GitHub Actions:**
- Translation QA workflow
- Composite action for reusability
- Example workflows
- Automated PyPI publishing

**Documentation:**
- User Guide (installation, usage, troubleshooting)
- Developer Guide (architecture, contributing)
- API Reference
- Usage examples

## ⚠️ Alpha Release Notice

This is an **alpha release**. While the core functionality is stable and well-tested, the API may change in future versions. We recommend:

- ✅ Use for experimentation and testing
- ✅ Provide feedback and report issues
- ⚠️ Be prepared for breaking changes
- ⚠️ Pin to specific version in production

## 🐛 Known Limitations

- `kttc translate` command not yet implemented (placeholder)
- Limited to text file formats (`.txt`, `.md`)
- COMET metrics integration planned for future release
- Style and Locale agents planned for future release

## 🔜 What's Next (v0.2.0)

- Style Agent for stylistic evaluation
- Locale Agent for cultural adaptation
- COMET metrics integration
- Support for `.xliff`, `.po` file formats
- Translation refinement loop (TEaR)
- Glossary integration

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards
- [GitHub Issues](https://github.com/kttc-ai/kttc/issues) - Report bugs or request features

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal UI
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [pytest](https://docs.pytest.org/) - Testing framework

## 📞 Support

- **Issues:** https://github.com/kttc-ai/kttc/issues
- **Discussions:** https://github.com/kttc-ai/kttc/discussions
- **Email:** dev@kttc.ai

---

**Thank you for trying KTTC!** We're excited to see what you build with it. 🚀

If you find KTTC useful, please star the repository and share it with others!
