# Installation

## Requirements

- Python 3.11 or higher
- pip (Python package installer)

## Basic Installation

Install KTTC using pip:

```bash
pip install kttc
```

## Optional Language Enhancements

KTTC provides optional language-specific helpers for enhanced quality checks:

### English

Install LanguageTool integration for 5,000+ grammar rules:

```bash
pip install kttc[english]
```

### Chinese

Install HanLP for Chinese-specific checks (measure words, particles):

```bash
pip install kttc[chinese]
```

### All Languages

Install all language helpers:

```bash
pip install kttc[all-languages]
```

## API Keys Setup

KTTC supports multiple LLM providers. You need at least one API key:

### OpenAI

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

### Anthropic (Claude)

```bash
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### GigaChat (Russian provider)

```bash
export KTTC_GIGACHAT_CLIENT_ID="your-client-id"
export KTTC_GIGACHAT_CLIENT_SECRET="your-client-secret"
```

### YandexGPT

```bash
export KTTC_YANDEXGPT_API_KEY="your-api-key"
export KTTC_YANDEXGPT_FOLDER_ID="your-folder-id"
```

## Using .env File

You can also create a `.env` file in your project directory:

```bash
# .env
KTTC_OPENAI_API_KEY=sk-...
KTTC_ANTHROPIC_API_KEY=sk-ant-...
```

KTTC will automatically load these variables.

## Verify Installation

Check that KTTC is installed correctly:

```bash
kttc --version
```

You should see the version number:

```
kttc 0.1.0
```

## Development Installation

If you want to contribute to KTTC or run the latest development version:

```bash
# Clone repository
git clone https://github.com/kttc-ai/kttc.git
cd kttc

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Next Steps

Now that you have KTTC installed, check out the [Quick Start](quickstart.md) guide to learn how to use it.
