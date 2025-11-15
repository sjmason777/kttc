# CLI Reference

Complete reference for KTTC command-line interface.

## Main Command

```bash
kttc [OPTIONS] COMMAND [ARGS]...
```

## Global Options

- `--version` - Show version and exit
- `--help` - Show help message and exit
- `-v, --verbose` - Increase verbosity
- `-q, --quiet` - Decrease verbosity

## Commands

### check

Check translation quality.

```bash
kttc check SOURCE TRANSLATION [OPTIONS]
```

**Arguments:**

- `SOURCE` - Source text file or directory
- `TRANSLATION` - Translation file(s) or directory

**Options:**

- `--source-lang LANG` - Source language code (e.g., en, es, fr)
- `--target-lang LANG` - Target language code
- `--threshold SCORE` - Minimum MQM score (default: 95.0)
- `--glossary FILE` - Path to glossary JSON file
- `--provider PROVIDER` - LLM provider (openai, anthropic, gigachat, yandexgpt)
- `--output-format FORMAT` - Output format (text, json, yaml)
- `--output-file FILE` - Save results to file
- `--smart-routing` - Enable complexity-based routing

**Examples:**

```bash
# Basic check
kttc check source.txt translation.txt --source-lang en --target-lang es

# With glossary and custom threshold
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --glossary terms.json \
    --threshold 98.0

# JSON output
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --output-format json \
    --output-file results.json
```

### batch

Process multiple translations from a file.

```bash
kttc batch [OPTIONS]
```

**Options:**

- `--file FILE` - Input CSV or JSON file with translations
- `--glossary FILE` - Path to glossary JSON file
- `--output-format FORMAT` - Output format (text, json, yaml)
- `--output-file FILE` - Save results to file
- `--parallel N` - Number of parallel workers

**CSV Format:**

```csv
source,translation,source_lang,target_lang
"Hello","Hola","en","es"
"Goodbye","Adiós","en","es"
```

**Examples:**

```bash
# Process batch from CSV
kttc batch --file translations.csv

# With parallel processing
kttc batch --file translations.csv --parallel 4

# Save results to JSON
kttc batch --file translations.csv \
    --output-format json \
    --output-file results.json
```

### compare

Compare multiple translations.

```bash
kttc compare --source SOURCE -t TRANS1 -t TRANS2 [OPTIONS]
```

**Options:**

- `--source FILE` - Source text file
- `-t, --translation FILE` - Translation file (can be used multiple times)
- `--source-lang LANG` - Source language code
- `--target-lang LANG` - Target language code
- `--output-format FORMAT` - Output format

**Example:**

```bash
kttc compare --source source.txt \
    -t translation1.txt \
    -t translation2.txt \
    -t translation3.txt \
    --source-lang en --target-lang es
```

### translate

Translate text with automatic QA.

```bash
kttc translate [OPTIONS]
```

**Options:**

- `--text TEXT` - Text to translate
- `--file FILE` - File to translate
- `--source-lang LANG` - Source language code
- `--target-lang LANG` - Target language code
- `--provider PROVIDER` - LLM provider
- `--glossary FILE` - Glossary file

**Examples:**

```bash
# Translate text
kttc translate --text "Hello, world!" \
    --source-lang en --target-lang es

# Translate file
kttc translate --file source.txt \
    --source-lang en --target-lang es \
    --provider anthropic
```

### benchmark

Benchmark different LLM providers.

```bash
kttc benchmark [OPTIONS]
```

**Options:**

- `--source FILE` - Source text file
- `--providers LIST` - Comma-separated provider list
- `--source-lang LANG` - Source language code
- `--target-lang LANG` - Target language code

**Example:**

```bash
kttc benchmark --source test.txt \
    --providers openai,anthropic,gigachat \
    --source-lang en --target-lang es
```

### glossary

Manage glossaries.

```bash
kttc glossary COMMAND [OPTIONS]
```

**Subcommands:**

- `list` - List all glossaries
- `show NAME` - Show glossary details
- `validate FILE` - Validate glossary format
- `create` - Create new glossary (interactive)

**Examples:**

```bash
# List glossaries
kttc glossary list

# Show specific glossary
kttc glossary show my-glossary

# Validate glossary
kttc glossary validate technical-terms.json
```

## Exit Codes

- `0` - Success (quality threshold met)
- `1` - Failure (quality below threshold)
- `2` - Error (invalid input, missing API key, etc.)

## Configuration

KTTC can be configured via:

1. Command-line options (highest priority)
2. `.kttc.yml` in current directory
3. `~/.kttc.yml` in home directory
4. Environment variables (lowest priority)

See [Configuration](../configuration.md) for details.

## Environment Variables

- `KTTC_OPENAI_API_KEY` - OpenAI API key
- `KTTC_ANTHROPIC_API_KEY` - Anthropic API key
- `KTTC_GIGACHAT_CLIENT_ID` - GigaChat client ID
- `KTTC_GIGACHAT_CLIENT_SECRET` - GigaChat client secret
- `KTTC_YANDEXGPT_API_KEY` - YandexGPT API key
- `KTTC_YANDEXGPT_FOLDER_ID` - YandexGPT folder ID
- `KTTC_DEFAULT_PROVIDER` - Default provider
- `KTTC_DEFAULT_THRESHOLD` - Default quality threshold

## Examples

### CI/CD Integration

```bash
#!/bin/bash
# Check translation quality in CI/CD

kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --threshold 95.0 \
    --output-format json \
    --output-file qa-report.json

if [ $? -eq 0 ]; then
    echo "✅ Translation quality check passed"
    exit 0
else
    echo "❌ Translation quality check failed"
    cat qa-report.json
    exit 1
fi
```

### Batch Processing with Glossary

```bash
kttc batch --file translations.csv \
    --glossary technical-terms.json \
    --parallel 8 \
    --output-format json \
    --output-file batch-results.json
```

### Compare Multiple Providers

```bash
kttc benchmark --source test.txt \
    --providers openai,anthropic \
    --source-lang en --target-lang es \
    --output-format json
```
