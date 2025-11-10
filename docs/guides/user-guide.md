# KTTC User Guide

Welcome to the KTTC (Knowledge Translation Transmutation Core) User Guide. This guide will help you get started with using KTTC for translation quality assurance.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Commands](#commands)
4. [Configuration](#configuration)
5. [Output Formats](#output-formats)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python 3.11 or higher
- pip package manager
- OpenAI or Anthropic API key

### Install from PyPI

```bash
pip install kttc
```

### Verify Installation

```bash
kttc --version
kttc --help
```

### Setup API Keys

Set your LLM provider API key as an environment variable:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-api-key-here"

# Or Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# Or Yandex
export YANDEX_API_KEY="your-api-key-here"
export YANDEX_FOLDER_ID="your-folder-id"

# Or GigaChat
export GIGACHAT_CREDENTIALS="your-credentials"
```

You can also add these to your `.env` file or shell profile for persistence.

---

## Quick Start

### Basic Translation Check

Check the quality of a single translation:

```bash
kttc check \
  --source source.txt \
  --translation translation.txt \
  --source-lang en \
  --target-lang es \
  --threshold 95
```

**Output:**
```
✅ PASS - MQM Score: 96.5
⚠️  2 minor issues found

Issues:
  • [MINOR] Fluency: Slightly awkward phrasing at position 45-52
  • [MINOR] Terminology: Inconsistent term usage at position 120-135
```

### Batch Processing

Process multiple translation files:

```bash
kttc batch \
  --source-dir translations/en/ \
  --translation-dir translations/es/ \
  --source-lang en \
  --target-lang es \
  --threshold 95 \
  --parallel 5
```

### Generate Report

Create a detailed quality report:

```bash
kttc report \
  --source source.txt \
  --translation translation.txt \
  --source-lang en \
  --target-lang es \
  --output report.html \
  --format html
```

---

## Commands

### `kttc check`

Check translation quality for a single file pair.

**Options:**
- `--source`, `-s` (required): Path to source text file
- `--translation`, `-t` (required): Path to translation file
- `--source-lang` (required): Source language code (e.g., `en`, `es`, `fr`)
- `--target-lang` (required): Target language code
- `--threshold` (default: `95.0`): Quality threshold (MQM score)
- `--provider`: LLM provider (`openai`, `anthropic`, `yandex`, `gigachat`)
- `--output`, `-o`: Output file path (optional)
- `--format`: Output format (`text`, `json`, `markdown`, `html`)
- `--verbose`, `-v`: Enable verbose output

**Example:**
```bash
kttc check \
  -s source.txt \
  -t translation.txt \
  --source-lang en \
  --target-lang es \
  --threshold 95 \
  --format json \
  -o report.json
```

**Exit Codes:**
- `0`: Translation passed quality threshold
- `1`: Translation failed quality threshold
- `2`: Error during execution

### `kttc batch`

Process multiple translation files in parallel.

**Options:**
- `--source-dir` (required): Directory containing source files
- `--translation-dir` (required): Directory containing translation files
- `--source-lang` (required): Source language code
- `--target-lang` (required): Target language code
- `--threshold` (default: `95.0`): Quality threshold
- `--parallel` (default: `3`): Number of parallel workers
- `--provider`: LLM provider
- `--output`, `-o`: Output directory for reports
- `--format`: Output format (`json`, `markdown`, `html`)
- `--verbose`, `-v`: Enable verbose output

**Example:**
```bash
kttc batch \
  --source-dir ./en/ \
  --translation-dir ./es/ \
  --source-lang en \
  --target-lang es \
  --parallel 5 \
  --output ./reports/ \
  --format html
```

**File Matching:**
- Files are matched by name: `source.txt` → `source.txt`
- Supported formats: `.txt`, `.md`, `.html` (text content)

### `kttc report`

Generate detailed quality assessment reports.

**Options:**
- `--source`, `-s` (required): Path to source file
- `--translation`, `-t` (required): Path to translation file
- `--source-lang` (required): Source language code
- `--target-lang` (required): Target language code
- `--output`, `-o` (required): Output file path
- `--format` (default: `html`): Report format (`json`, `markdown`, `html`)
- `--provider`: LLM provider

**Example:**
```bash
kttc report \
  -s source.txt \
  -t translation.txt \
  --source-lang en \
  --target-lang es \
  -o report.html \
  --format html
```

### `kttc translate`

*(Coming soon)* Translate and check quality in one command.

---

## Configuration

### Environment Variables

KTTC uses environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required if using OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required if using Claude |
| `YANDEX_API_KEY` | Yandex Cloud API key | Required if using YandexGPT |
| `YANDEX_FOLDER_ID` | Yandex Cloud folder ID | Required if using YandexGPT |
| `GIGACHAT_CREDENTIALS` | GigaChat credentials | Required if using GigaChat |
| `KTTC_DEFAULT_PROVIDER` | Default LLM provider | `openai` |
| `KTTC_DEFAULT_THRESHOLD` | Default quality threshold | `95.0` |

### Configuration File

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
KTTC_DEFAULT_PROVIDER=openai
KTTC_DEFAULT_THRESHOLD=95.0
```

KTTC will automatically load this file.

---

## Output Formats

### Text Format (Default)

Human-readable console output:

```
✅ PASS - MQM Score: 96.5
⚠️  2 minor issues found
```

### JSON Format

Machine-readable structured data:

```json
{
  "task": {
    "source_text": "Hello, world!",
    "translation": "¡Hola, mundo!",
    "source_lang": "en",
    "target_lang": "es"
  },
  "mqm_score": 96.5,
  "status": "pass",
  "errors": [
    {
      "category": "fluency",
      "subcategory": "grammar",
      "severity": "minor",
      "location": [45, 52],
      "description": "Slightly awkward phrasing"
    }
  ]
}
```

### Markdown Format

Formatted report for documentation:

```markdown
# Translation Quality Report

## Summary
- **MQM Score**: 96.5
- **Status**: ✅ PASS
- **Errors**: 2 minor

## Issues
### 1. Fluency Issue (Minor)
- **Location**: 45-52
- **Description**: Slightly awkward phrasing
```

### HTML Format

Interactive web report with syntax highlighting:

```html
<!DOCTYPE html>
<html>
<head>
  <title>Translation Quality Report</title>
  <style>/* ... */</style>
</head>
<body>
  <h1>Translation Quality Report</h1>
  <div class="summary">
    <p class="score">MQM Score: 96.5</p>
    <p class="status pass">✅ PASS</p>
  </div>
  <!-- ... -->
</body>
</html>
```

---

## Best Practices

### 1. Set Appropriate Thresholds

Different use cases require different quality thresholds:

- **Professional/Published Content**: 95-100 (very high quality)
- **Internal Documentation**: 90-95 (high quality)
- **Draft/Informal Content**: 85-90 (good quality)

```bash
# Professional translation
kttc check -s source.txt -t translation.txt --threshold 95

# Internal docs
kttc check -s source.txt -t translation.txt --threshold 90
```

### 2. Use Batch Processing for Multiple Files

For projects with many translations, use batch mode:

```bash
kttc batch \
  --source-dir ./en/ \
  --translation-dir ./es/ \
  --source-lang en \
  --target-lang es \
  --parallel 5
```

**Performance tips:**
- Use `--parallel` to process files concurrently
- Recommended: 3-5 parallel workers
- Higher parallelism = more API usage

### 3. Choose the Right LLM Provider

Different providers have different strengths:

- **OpenAI (`openai`)**: Best overall quality, fastest
- **Anthropic (`anthropic`)**: Excellent for nuanced analysis
- **Yandex (`yandex`)**: Good for Russian language pairs
- **GigaChat (`gigachat`)**: Good for Russian language pairs

```bash
kttc check \
  -s source.txt \
  -t translation.txt \
  --provider anthropic
```

### 4. Save Reports for Documentation

Generate HTML reports for review:

```bash
kttc report \
  -s source.txt \
  -t translation.txt \
  --source-lang en \
  --target-lang es \
  -o report.html \
  --format html
```

### 5. Integrate with CI/CD

Add translation checks to your CI pipeline:

```yaml
# .github/workflows/translation-qa.yml
- name: Check Translation Quality
  run: |
    kttc batch \
      --source-dir ./translations/en/ \
      --translation-dir ./translations/es/ \
      --source-lang en \
      --target-lang es \
      --threshold 95
```

See [GitHub Actions Guide](../development/github-actions.md) for more details.

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Error:**
```
Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable.
```

**Solution:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### 2. File Not Found

**Error:**
```
Error: Source file not found: source.txt
```

**Solution:**
- Check file path is correct
- Use absolute paths if relative paths fail
- Ensure file exists: `ls -l source.txt`

#### 3. Low MQM Score

**Error:**
```
❌ FAIL - MQM Score: 82.3
```

**Solution:**
- Review the error list to identify issues
- Common issues: mistranslations, grammar, terminology
- Lower threshold if acceptable: `--threshold 80`

#### 4. API Rate Limiting

**Error:**
```
Error: Rate limit exceeded. Please try again later.
```

**Solution:**
- Reduce parallelism: `--parallel 1`
- Wait before retrying
- Use a different provider: `--provider anthropic`

#### 5. Timeout Errors

**Error:**
```
Error: Request timed out after 60s
```

**Solution:**
- Break large files into smaller chunks
- Increase timeout (future feature)
- Check API service status

### Getting Help

- **Documentation**: https://github.com/kttc-ai/kttc/tree/main/docs
- **Issues**: https://github.com/kttc-ai/kttc/issues
- **Discussions**: https://github.com/kttc-ai/kttc/discussions

---

## Advanced Usage

### Custom Quality Thresholds by Category

*(Future feature)* Set different thresholds for different error categories:

```bash
kttc check \
  -s source.txt \
  -t translation.txt \
  --accuracy-threshold 95 \
  --fluency-threshold 90 \
  --terminology-threshold 98
```

### Glossary Integration

*(Future feature)* Use custom terminology glossaries:

```bash
kttc check \
  -s source.txt \
  -t translation.txt \
  --glossary terms.csv
```

### Translation Memory

*(Future feature)* Leverage translation memory:

```bash
kttc check \
  -s source.txt \
  -t translation.txt \
  --tm translation-memory.tmx
```

---

## Next Steps

- Read the [Developer Guide](../development/developer-guide.md) to contribute
- Check the [API Documentation](../api/README.md) for programmatic usage
- See [GitHub Actions Integration](../development/github-actions.md) for CI/CD

---

**Last Updated:** November 10, 2025
**Version:** 0.1.0
