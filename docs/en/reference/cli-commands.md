# CLI Commands Reference

Complete reference for all KTTC command-line commands.

## kttc check

Smart translation quality checker with auto-detection.

### Syntax

```bash
kttc check SOURCE [TRANSLATIONS...] [OPTIONS]
```

### Auto-Detection Modes

`kttc check` automatically detects what you want to do:

| Input | Detected Mode | Behavior |
|-------|--------------|----------|
| `source.txt translation.txt` | Single check | Quality assessment |
| `source.txt trans1.txt trans2.txt` | Compare | Automatic comparison |
| `translations.csv` | Batch (file) | Process CSV/JSON |
| `source_dir/ trans_dir/` | Batch (directory) | Process directories |

### Options

#### Required (for single/compare modes)

- `--source-lang CODE` - Source language code (e.g., `en`)
- `--target-lang CODE` - Target language code (e.g., `ru`)

####Smart Features (enabled by default)

- `--smart-routing` / `--no-smart-routing` - Complexity-based model selection (default: enabled)
- `--glossary TEXT` - Glossaries to use: `auto` (default), `none`, or comma-separated names
- `--output PATH` - Auto-detects format from extension (`.json`, `.md`, `.html`)

#### Quality Control

- `--threshold FLOAT` - Minimum MQM score (default: 95.0)
- `--auto-correct` - Automatically fix detected errors
- `--correction-level light|full` - Correction level (default: `light`)

#### Model Selection

- `--provider openai|anthropic|gigachat|yandex` - LLM provider
- `--auto-select-model` - Use optimal model for language pair
- `--show-routing-info` - Display complexity analysis

#### Output & Verbosity

- `--format text|json|markdown|html` - Output format (overrides auto-detection)
- `--verbose` - Show detailed output
- `--demo` - Demo mode (no API calls, simulated responses)

### Examples

**Single file check:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es
```

**Compare multiple translations (auto-detected):**

```bash
kttc check source.txt trans1.txt trans2.txt trans3.txt \
  --source-lang en \
  --target-lang ru
```

**Batch process CSV (auto-detected, languages from file):**

```bash
kttc check translations.csv
```

**Batch process directories:**

```bash
kttc check source_dir/ translation_dir/ \
  --source-lang en \
  --target-lang ru
```

**Auto-correction:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --auto-correct \
  --correction-level full
```

**HTML report (auto-detected from extension):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --output report.html
```

**Disable smart features:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --no-smart-routing \
  --glossary none
```

**Demo mode (no API calls):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es \
  --demo
```

---

## kttc batch

Batch process multiple translations.

### Syntax

**File mode:**

```bash
kttc batch --file FILE [OPTIONS]
```

**Directory mode:**

```bash
kttc batch --source-dir DIR --translation-dir DIR \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### Options

#### Mode Selection (mutually exclusive)

- `--file PATH` - Batch file (CSV, JSON, or JSONL)
- `--source-dir PATH` + `--translation-dir PATH` - Directory mode

#### Required (directory mode only)

- `--source-lang CODE` - Source language code
- `--target-lang CODE` - Target language code

#### Common Options

- `--threshold FLOAT` - Minimum MQM score (default: 95.0)
- `--output PATH` - Output report path (default: `report.json`)
- `--parallel INT` - Number of parallel workers (default: 4)
- `--glossary TEXT` - Glossaries to use
- `--smart-routing` - Enable complexity-based routing
- `--show-progress` / `--no-progress` - Show progress bar (default: show)
- `--verbose` - Verbose output
- `--demo` - Demo mode

#### File Mode Only

- `--batch-size INT` - Batch size for grouping

### Supported File Formats

**CSV:**

```csv
source,translation,source_lang,target_lang,domain
"Hello world","Hola mundo","en","es","general"
```

**JSON:**

```json
[
  {
    "source": "Hello world",
    "translation": "Hola mundo",
    "source_lang": "en",
    "target_lang": "es",
    "domain": "general"
  }
]
```

**JSONL:**

```jsonl
{"source": "Hello world", "translation": "Hola mundo", "source_lang": "en", "target_lang": "es"}
{"source": "Good morning", "translation": "Buenos días", "source_lang": "en", "target_lang": "es"}
```

### Examples

**Process CSV file:**

```bash
kttc batch --file translations.csv
```

**Process JSON with progress:**

```bash
kttc batch --file translations.json \
  --show-progress \
  --output results.json
```

**Directory mode:**

```bash
kttc batch \
  --source-dir ./source \
  --translation-dir ./translations \
  --source-lang en \
  --target-lang es \
  --parallel 8
```

---

## kttc compare

Compare multiple translations side by side.

### Syntax

```bash
kttc compare --source FILE \
  --translation FILE --translation FILE [...] \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### Options

- `--source PATH` - Source text file (required)
- `--translation PATH` - Translation file (can be specified multiple times, required)
- `--source-lang CODE` - Source language code (required)
- `--target-lang CODE` - Target language code (required)
- `--threshold FLOAT` - Quality threshold (default: 95.0)
- `--provider TEXT` - LLM provider
- `--verbose` - Show detailed comparison

### Examples

**Compare 3 translations:**

```bash
kttc compare \
  --source text.txt \
  --translation trans1.txt \
  --translation trans2.txt \
  --translation trans3.txt \
  --source-lang en \
  --target-lang ru \
  --verbose
```

---

## kttc translate

Translate text with automatic quality checking and refinement.

### Syntax

```bash
kttc translate --text TEXT \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### Options

- `--text TEXT` - Text to translate (or `@file.txt` for file input, required)
- `--source-lang CODE` - Source language code (required)
- `--target-lang CODE` - Target language code (required)
- `--threshold FLOAT` - Quality threshold for refinement (default: 95.0)
- `--max-iterations INT` - Maximum refinement iterations (default: 3)
- `--output PATH` - Output file path
- `--provider TEXT` - LLM provider
- `--verbose` - Verbose output

### Examples

**Translate inline text:**

```bash
kttc translate --text "Hello, world!" \
  --source-lang en \
  --target-lang es
```

**Translate from file:**

```bash
kttc translate --text @document.txt \
  --source-lang en \
  --target-lang ru \
  --output translated.txt
```

**With quality threshold:**

```bash
kttc translate --text "Complex technical text" \
  --source-lang en \
  --target-lang zh \
  --threshold 98 \
  --max-iterations 5
```

---

## kttc benchmark

Benchmark multiple LLM providers.

### Syntax

```bash
kttc benchmark --source FILE \
  --source-lang CODE --target-lang CODE \
  --providers LIST [OPTIONS]
```

### Options

- `--source PATH` - Source text file (required)
- `--source-lang CODE` - Source language code (required)
- `--target-lang CODE` - Target language code (required)
- `--providers TEXT` - Comma-separated provider list (default: `gigachat,openai,anthropic`)
- `--threshold FLOAT` - Quality threshold (default: 95.0)
- `--output PATH` - Output file path (JSON)
- `--verbose` - Verbose output

### Examples

**Benchmark all providers:**

```bash
kttc benchmark \
  --source text.txt \
  --source-lang en \
  --target-lang ru \
  --providers gigachat,openai,anthropic
```

---

## kttc report

Generate formatted reports from QA results.

### Syntax

```bash
kttc report INPUT_FILE [OPTIONS]
```

### Options

- `--format markdown|html` - Output format (default: markdown)
- `--output PATH` - Output file path (auto-generated if not specified)

### Examples

**Generate Markdown report:**

```bash
kttc report results.json --format markdown -o report.md
```

**Generate HTML report:**

```bash
kttc report results.json --format html -o report.html
```

---

## kttc glossary

Manage terminology glossaries.

### Subcommands

#### list

List available glossaries:

```bash
kttc glossary list
```

#### show

Show glossary details:

```bash
kttc glossary show NAME
```

#### add

Add glossary entry:

```bash
kttc glossary add NAME \
  --source TEXT \
  --target TEXT \
  --lang-pair SRC-TGT
```

#### import

Import glossary from file:

```bash
kttc glossary import NAME \
  --file PATH \
  --format csv|json|tbx
```

### Examples

**List glossaries:**

```bash
kttc glossary list
```

**Show base glossary:**

```bash
kttc glossary show base
```

**Add term:**

```bash
kttc glossary add medical \
  --source "myocardial infarction" \
  --target "инфаркт миокарда" \
  --lang-pair en-ru
```

**Import from CSV:**

```bash
kttc glossary import technical \
  --file terms.csv \
  --format csv
```

---

## Global Options

Available for all commands:

- `--version`, `-v` - Show version and exit
- `--help` - Show help message

---

## Exit Codes

- `0` - Success (all translations passed quality threshold)
- `1` - Failure (one or more translations failed quality threshold)
- `130` - Interrupted by user (Ctrl+C)

---

## Environment Variables

- `KTTC_OPENAI_API_KEY` - OpenAI API key
- `KTTC_ANTHROPIC_API_KEY` - Anthropic API key
- `KTTC_GIGACHAT_CLIENT_ID` - GigaChat client ID
- `KTTC_GIGACHAT_CLIENT_SECRET` - GigaChat client secret
- `KTTC_YANDEX_API_KEY` - Yandex GPT API key
- `KTTC_YANDEX_FOLDER_ID` - Yandex GPT folder ID

---

## See Also

- [CLI Usage Guide](../guides/cli-usage.md) - Practical examples
- [Configuration](../guides/configuration.md) - Advanced configuration
- [API Reference](api-reference.md) - Python API
