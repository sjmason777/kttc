# KTTC CLI Examples - Hybrid Format

This directory contains example files for testing KTTC's **smart CLI**.

## 📁 Files

- `source_en.txt` - English source text (pangram example)
- `translation_ru_good.txt` - Good quality Russian translation
- `translation_ru_bad.txt` - Poor quality Russian translation (contains mixed languages)
- `translation_ru_deepl.txt` - DeepL translation (for comparison)

## 🎯 Smart Check Examples (Recommended)

### 1. Single File Check

```bash
# Simple - auto-detects everything!
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru

# With smart defaults:
# ✅ Smart routing enabled (saves money)
# ✅ Auto-glossary detected
# ✅ Beautiful terminal UI
```

### 2. Compare Multiple Translations (Auto-Detected!)

Just add more translation files - compare mode activates automatically:

```bash
# 🎯 Auto-detected as COMPARE mode!
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_good.txt \
  examples/cli/translation_ru_bad.txt \
  examples/cli/translation_ru_deepl.txt \
  --source-lang en \
  --target-lang ru

# Shows beautiful comparison table automatically
```

### 3. Auto-Correction

Automatically fix detected errors:

```bash
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_bad.txt \
  --source-lang en \
  --target-lang ru \
  --auto-correct \
  --correction-level full
```

### 4. HTML Report (Auto-Format Detection!)

```bash
# Just use .html extension - format auto-detected!
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru \
  --output report.html

# 📄 Auto-detects HTML format from extension!
```

### 5. Disable Smart Features

```bash
# Turn off smart defaults if needed
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru \
  --no-smart-routing \
  --glossary none
```

---

## 🔄 Legacy Commands (Still Work)

### Compare (Old Way)

```bash
# Still works, but new way is simpler
kttc compare \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_good.txt \
  --translation examples/cli/translation_ru_bad.txt \
  --source-lang en \
  --target-lang ru
```

💡 **Recommended:** Use `kttc check` with multiple files instead!

### Benchmark Providers

```bash
kttc benchmark \
  --source examples/cli/source_en.txt \
  --source-lang en \
  --target-lang ru \
  --providers gigachat,openai,anthropic
```

### Translate with Quality Check

```bash
kttc translate \
  --text "Hello world, this is a test!" \
  --source-lang en \
  --target-lang ru \
  --threshold 95 \
  --max-iterations 3
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
# JSON output from check
kttc check SOURCE TRANSLATION \
  --source-lang en --target-lang ru \
  --output report.json --format json

# Markdown report from check
kttc check SOURCE TRANSLATION \
  --source-lang en --target-lang ru \
  --output report.md --format markdown

# HTML report from JSON
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
kttc check src.txt tgt.txt \
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
kttc check SOURCE TRANSLATION \
  --source-lang en --target-lang ru \
  --output results.json --format json

# Parse with jq
cat results.json | jq '.mqm_score'
cat results.json | jq '.errors[] | select(.severity=="critical")'
```

### Performance Optimization

For large-scale processing:

```bash
# Use parallel processing
kttc batch --file translations.csv --parallel 8

# Optimize for speed (GigaChat is fastest)
kttc benchmark --source text.txt \
  --source-lang en --target-lang ru \
  --providers gigachat
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
- [Contributing Guide](https://github.com/kttc-ai/kttc/blob/main/CONTRIBUTING.md)
