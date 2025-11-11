# KTTC CLI Examples

This directory contains example files for testing the KTTC CLI commands.

## Files

- `source_en.txt` - English source text (pangram example)
- `translation_ru_good.txt` - Good quality Russian translation
- `translation_ru_bad.txt` - Poor quality Russian translation (contains mixed languages)
- `translation_ru_deepl.txt` - DeepL translation (for comparison)

## Usage Examples

### 1. Check Translation Quality

Check a single translation:

```bash
kttc check \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru \
  --verbose
```

### 2. Compare Multiple Translations

Compare different translations side by side:

```bash
kttc compare \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_good.txt \
  --translation examples/cli/translation_ru_bad.txt \
  --translation examples/cli/translation_ru_deepl.txt \
  --source-lang en \
  --target-lang ru \
  --verbose
```

### 3. Benchmark Providers

Benchmark different LLM providers:

```bash
kttc benchmark \
  --source examples/cli/source_en.txt \
  --source-lang en \
  --target-lang ru \
  --providers gigachat,openai,anthropic \
  --reference examples/cli/translation_ru_good.txt \
  --output benchmark_results.json
```

### 4. Batch Processing

Process multiple files at once:

```bash
kttc batch \
  --source-dir examples/cli/sources/ \
  --translation-dir examples/cli/translations/ \
  --source-lang en \
  --target-lang ru \
  --output batch_report.json \
  --parallel 4
```

### 5. Translation with Quality Check

Translate and iteratively improve:

```bash
kttc translate \
  --text "Hello world, this is a test!" \
  --source-lang en \
  --target-lang ru \
  --threshold 95 \
  --max-iterations 3 \
  --output translation.txt \
  --verbose
```

### 6. Auto-Correction

Check and automatically correct errors:

```bash
kttc check \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_bad.txt \
  --source-lang en \
  --target-lang ru \
  --auto-correct \
  --correction-level full \
  --verbose
```

## Tips

### Getting Help

View all available commands:
```bash
kttc --help
```

View help for a specific command:
```bash
kttc check --help
kttc benchmark --help
kttc compare --help
```

### Configuration

Set environment variables in `.env` file:

```bash
# OpenAI
KTTC_OPENAI_API_KEY=sk-...

# Anthropic
KTTC_ANTHROPIC_API_KEY=sk-ant-...

# GigaChat (Sber)
KTTC_GIGACHAT_CLIENT_ID=...
KTTC_GIGACHAT_CLIENT_SECRET=...
```

### Output Formats

Save results in different formats:

```bash
# JSON output
kttc check ... --output report.json --format json

# Markdown report
kttc check ... --output report.md --format markdown

# HTML report
kttc report report.json --format html --output report.html
```

## Visual Features

The CLI includes beautiful visual features:

- ✓ **Color-coded status** - Green for pass, red for fail
- 📊 **Rich tables** - Beautiful comparison tables
- 📈 **Progress bars** - Real-time progress tracking
- 🎨 **Syntax highlighting** - Color-coded error severities
- 📋 **Structured panels** - Organized information display
- ⚡ **Spinners** - Visual feedback for long operations

## Advanced Usage

### Pipeline Integration

Use in CI/CD pipelines:

```bash
# Exit code 0 if pass, 1 if fail
kttc check --source src.txt --translation tgt.txt \
  --source-lang en --target-lang ru --threshold 95

# Check exit code
if [ $? -eq 0 ]; then
  echo "Quality check passed!"
else
  echo "Quality check failed!"
  exit 1
fi
```

### Scripting

Process results programmatically:

```bash
# Generate JSON output
kttc check ... --output results.json --format json

# Parse with jq
cat results.json | jq '.mqm_score'
cat results.json | jq '.errors[] | select(.severity=="critical")'
```

### Performance Optimization

For large-scale processing:

```bash
# Use parallel processing
kttc batch ... --parallel 8

# Optimize for speed
kttc benchmark ... --providers gigachat  # Fastest
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError:**
```bash
pip install -e ".[dev]"
```

**API Key not found:**
- Check `.env` file exists
- Verify variable names match `KTTC_*`
- Use absolute paths in `.env`

**Import errors with COMET:**
```bash
pip install unbabel-comet sentence-transformers
```

## Further Reading

- [KTTC Documentation](https://github.com/kttc-ai/kttc)
- [API Reference](https://github.com/kttc-ai/docs)
- [Contributing Guide](https://github.com/kttc-ai/kttc/blob/main/CONTRIBUTING.md)
