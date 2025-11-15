# CLI Usage

KTTC provides a powerful command-line interface for translation quality checking.

## Basic Commands

### Check Single Translation

```bash
kttc check source.txt translation.txt --source-lang en --target-lang es
```

### Compare Multiple Translations

```bash
kttc check source.txt translation1.txt translation2.txt translation3.txt \
    --source-lang en --target-lang es
```

KTTC automatically detects when you provide multiple translations and compares them.

### Batch Processing

Process multiple translations from a CSV file:

```bash
kttc batch --file translations.csv
```

**CSV format:**

```csv
source,translation,source_lang,target_lang
"Hello","Hola","en","es"
"Goodbye","Adiós","en","es"
```

From directories:

```bash
kttc check source_dir/ translation_dir/ --source-lang en --target-lang es
```

## Options

### Language Options

```bash
--source-lang LANG    # Source language code (e.g., en, es, fr, zh, ru)
--target-lang LANG    # Target language code
```

### Quality Threshold

```bash
--threshold SCORE     # Minimum MQM score to pass (default: 95.0)
```

Example:

```bash
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --threshold 90.0
```

### Output Format

```bash
--output-format FORMAT    # json, yaml, text (default: text)
--output-file FILE        # Save results to file
```

Example:

```bash
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --output-format json \
    --output-file results.json
```

### Glossary

Use a custom glossary for terminology validation:

```bash
--glossary FILE    # Path to glossary JSON file
```

Example:

```bash
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --glossary my-glossary.json
```

**Glossary format:**

```json
{
  "terms": [
    {
      "source": "API",
      "target": "API",
      "context": "Keep as-is"
    }
  ]
}
```

### LLM Provider

Specify which LLM provider to use:

```bash
--provider PROVIDER    # openai, anthropic, gigachat, yandexgpt
```

Example:

```bash
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --provider anthropic
```

### Smart Routing

Enable smart routing for cost optimization:

```bash
--smart-routing    # Auto-select cheaper models for simple texts
```

This can reduce costs by up to 60%.

### Verbosity

Control output verbosity:

```bash
-v, --verbose     # Increase verbosity
-q, --quiet       # Decrease verbosity
```

## Advanced Commands

### Compare Translations Explicitly

```bash
kttc compare --source source.txt -t translation1.txt -t translation2.txt \
    --source-lang en --target-lang es
```

### Translate with QA

Translate text and automatically check quality:

```bash
kttc translate --text "Hello, world!" \
    --source-lang en --target-lang es \
    --provider openai
```

### Benchmark Providers

Compare different LLM providers:

```bash
kttc benchmark --source test.txt \
    --providers openai,anthropic,gigachat \
    --source-lang en --target-lang es
```

### Glossary Management

```bash
kttc glossary list                    # List all glossaries
kttc glossary show my-glossary        # Show glossary details
kttc glossary validate my-glossary    # Validate glossary format
```

## Exit Codes

KTTC uses standard exit codes for CI/CD integration:

- `0` - Success (quality threshold met)
- `1` - Failure (quality below threshold)
- `2` - Error (invalid input, missing API key, etc.)

Example in CI/CD:

```bash
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --threshold 95.0 || exit 1
```

## Examples

### Example 1: Basic Quality Check

```bash
kttc check en.txt es.txt --source-lang en --target-lang es
```

### Example 2: High-Quality Threshold with JSON Output

```bash
kttc check en.txt es.txt \
    --source-lang en --target-lang es \
    --threshold 98.0 \
    --output-format json \
    --output-file results.json
```

### Example 3: Batch Processing with Glossary

```bash
kttc batch --file batch.csv \
    --glossary technical-terms.json \
    --output-format json
```

### Example 4: Compare Three Translations

```bash
kttc check source.txt t1.txt t2.txt t3.txt \
    --source-lang en --target-lang es \
    --smart-routing
```

### Example 5: CI/CD Integration

```bash
#!/bin/bash
# CI script for translation quality check

kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --threshold 95.0 \
    --output-format json \
    --output-file qa-report.json

if [ $? -eq 0 ]; then
    echo "✅ Translation quality check passed"
else
    echo "❌ Translation quality check failed"
    exit 1
fi
```

## Getting Help

View help for any command:

```bash
kttc --help
kttc check --help
kttc batch --help
```
