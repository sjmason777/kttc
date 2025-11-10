# KTTC Examples

This directory contains example scripts showing how to use KTTC programmatically.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Simple example showing how to evaluate a single translation:

```bash
python examples/basic_usage.py
```

**Demonstrates:**
- Setting up an LLM provider
- Creating a translation task
- Running evaluation
- Printing results

### 2. Batch Processing (`batch_processing.py`)

Process multiple translations in parallel:

```bash
python examples/batch_processing.py
```

**Demonstrates:**
- Batch evaluation
- Parallel processing with semaphore
- Aggregate statistics
- JSON export

## Prerequisites

All examples require:
- KTTC installed: `pip install kttc`
- API key set: `export OPENAI_API_KEY="your-key"`

## Running Examples

```bash
# Install KTTC
pip install kttc

# Set API key
export OPENAI_API_KEY="your-openai-api-key"

# Run example
python examples/basic_usage.py
```

## Output

### Basic Usage Output

```
Evaluating translation quality...

============================================================
TRANSLATION QUALITY REPORT
============================================================

MQM Score: 98.50
Status: ✅ PASS
Errors Found: 1

Issues:

1. [MINOR] fluency
   Location: 15-22
   Description: Slightly awkward word order
   Suggestion: Consider rephrasing for better flow

============================================================
```

### Batch Processing Output

```
Evaluating 5 translations (max 3 parallel)...

============================================================
BATCH PROCESSING SUMMARY
============================================================
Total Translations: 5
Passed: 5 (100.0%)
Failed: 0 (0.0%)
Average MQM Score: 98.20
Total Errors Found: 3
============================================================

1. ✅ Score: 100.0 | Errors: 0
   Source: Hello, world!
   Translation: ¡Hola, mundo!

2. ✅ Score: 98.0 | Errors: 1
   Source: Good morning!
   Translation: ¡Buenos días!
   - [MINOR] fluency: Minor punctuation issue

...

Results exported to: /path/to/batch_results.json
```

## Advanced Usage

For more advanced usage, see:
- [User Guide](../docs/guides/user-guide.md)
- [Developer Guide](../docs/development/developer-guide.md)
- [API Documentation](../docs/api/README.md)

## Troubleshooting

### API Key Not Found

```
Error: OPENAI_API_KEY environment variable not set
```

**Solution:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Import Error

```
ModuleNotFoundError: No module named 'kttc'
```

**Solution:**
```bash
pip install kttc
# Or for development:
pip install -e .
```

## Contributing

Want to add more examples? See [CONTRIBUTING.md](../CONTRIBUTING.md).

---

**Last Updated:** November 10, 2025
