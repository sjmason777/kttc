# Installation Guide

This guide covers all installation methods and optional dependencies.

## Requirements

- **Python:** 3.11 or higher
- **Operating Systems:** Linux, macOS, Windows
- **API Key:** OpenAI, Anthropic, GigaChat, or Yandex

## Basic Installation

### Using pip (Recommended)

```bash
pip install kttc
```

This installs core dependencies (~50MB):
- CLI interface
- Multi-agent QA system
- Russian NLP (MAWO libraries)
- Basic multi-language support (spaCy, jieba)

### Using pipx (Isolated Environment)

```bash
pipx install kttc
```

Benefits:
- Isolated from system Python
- Automatic PATH configuration
- Easy upgrades

### From Source (Development)

```bash
git clone https://github.com/kttc-ai/kttc.git
cd kttc
pip install -e ".[dev]"
```

## Optional Dependencies

### Metrics (Semantic Similarity)

Adds sentence-level similarity metrics:

```bash
pip install kttc[metrics]
```

Includes:
- sentence-transformers
- Semantic similarity scoring

### English Language Support

Adds advanced English grammar checking with LanguageTool:

```bash
pip install kttc[english]
```

**Requirements:**
- Java 17.0 or higher
- ~200MB disk space

Features:
- 5,000+ grammar rules
- Subject-verb agreement
- Article checking (a/an/the)
- Preposition validation

**Install Java (if needed):**

```bash
# macOS
brew install openjdk@17

# Ubuntu/Debian
sudo apt install openjdk-17-jre

# Windows
# Download from https://adoptium.net/
```

**Download spaCy model:**

```bash
python3 -m spacy download en_core_web_md
```

### Chinese Language Support

Adds advanced Chinese NLP with HanLP:

```bash
pip install kttc[chinese]
```

Features:
- Measure word validation (量词)
- Aspect particle checking (了/过)
- High-accuracy POS tagging (~95%)
- ~300MB model download on first use

### All Language Helpers

Install all language-specific enhancements:

```bash
pip install kttc[all-languages]
```

Equivalent to:

```bash
pip install kttc[english,chinese]
```

### Full Installation (Development + All Features)

```bash
pip install kttc[full,dev]
```

## Verify Installation

Check KTTC version:

```bash
kttc --version
```

Test with demo mode (no API key needed):

```bash
echo "Hello" > source.txt
echo "Hola" > trans.txt
kttc check source.txt trans.txt --source-lang en --target-lang es --demo
```

## API Key Setup

### OpenAI

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

### Anthropic

```bash
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### GigaChat

```bash
export KTTC_GIGACHAT_CLIENT_ID="your-client-id"
export KTTC_GIGACHAT_CLIENT_SECRET="your-client-secret"
```

### Yandex GPT

```bash
export KTTC_YANDEX_API_KEY="your-api-key"
export KTTC_YANDEX_FOLDER_ID="your-folder-id"
```

### Persistent Configuration

Add to your shell profile (`~/.bashrc`, `~/.zshrc`):

```bash
# KTTC API Keys
export KTTC_OPENAI_API_KEY="sk-..."
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

Then reload:

```bash
source ~/.bashrc  # or ~/.zshrc
```

## Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade kttc
```

### Upgrade with Optional Dependencies

```bash
pip install --upgrade kttc[all-languages,metrics]
```

## Uninstalling

```bash
pip uninstall kttc
```

Remove downloaded models:

```bash
# spaCy models
python3 -m spacy uninstall en_core_web_md

# HanLP models (if installed)
rm -rf ~/.hanlp
```

## Troubleshooting

### Python Version Issues

**Error:** `TypeError: unsupported operand type(s) for |`

**Solution:** Use Python 3.11+:

```bash
python3.11 -m pip install kttc
```

### Java Not Found (LanguageTool)

**Error:** `java.lang.RuntimeException: Could not find java`

**Solution:** Install Java 17+:

```bash
# Check Java version
java -version

# Should show: openjdk version "17.0.x" or higher
```

### Permission Denied

**Error:** `ERROR: Could not install packages`

**Solution:** Use user install:

```bash
pip install --user kttc
```

Or use virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install kttc
```

## Next Steps

- [Configuration](configuration.md) - Configure settings
- [CLI Usage](cli-usage.md) - Learn the commands
- [Quickstart](../tutorials/quickstart.md) - Run your first check
