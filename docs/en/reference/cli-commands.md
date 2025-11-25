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

- `--format text|json|markdown|html|xlsx` - Output format (overrides auto-detection)
- `--verbose` - Show detailed output
- `--demo` - Demo mode (no API calls, simulated responses)

#### MQM Profiles (NEW in v1.3)

- `--profile NAME` - Use MQM profile: `default`, `strict`, `minimal`, `legal`, `medical`, `marketing`, `literary`, `technical`, or path to custom YAML profile

Built-in profiles:

| Profile | Agents | Threshold | Use Case |
|---------|--------|-----------|----------|
| `default` | accuracy, fluency, terminology | 95% | General translation QA |
| `strict` | accuracy, fluency, terminology, hallucination, context | 98% | High-stakes content |
| `minimal` | accuracy, fluency, terminology | 90% | Quick checks |
| `legal` | accuracy, terminology, context, hallucination | 97% | Legal documents |
| `medical` | accuracy, terminology, hallucination, context | 98% | Medical/pharmaceutical |
| `marketing` | fluency, accuracy, context | 93% | Marketing/creative |
| `literary` | style_preservation, accuracy, context | 88% | Literary works |
| `technical` | accuracy, terminology, fluency, hallucination | 97% | Technical documentation |

#### Agent Selection (NEW in v1.3)

- `--agents PRESET|LIST` - Agent preset or comma-separated list

Presets:
- `minimal` - accuracy, fluency (2 agents)
- `default` - accuracy, fluency, terminology (3 agents)
- `full` - accuracy, fluency, terminology, hallucination, context (5 agents)

Available agents: `accuracy`, `fluency`, `terminology`, `hallucination`, `context`, `style_preservation`, `fluency_russian`, `fluency_chinese`, `fluency_hindi`, `fluency_persian`, `fluency_english`

#### Performance & Cost (NEW in v1.1)

- `--quick`, `-q` - Quick mode: single pass with 3 core agents (accuracy, fluency, terminology). Faster and cheaper for simple texts.
- `--show-cost` - Display token usage and estimated API cost after check.

#### Technical Documentation Mode (NEW in v1.2)

KTTC automatically detects technical documentation (CLI docs, API references, README files) and skips literary style analysis. This prevents false positives like "stream of consciousness" or "pleonastic patterns" for code-heavy content.

**Detection markers:**
- Code blocks (` ```bash `, ` ```python `)
- CLI options (`--option`, `-flag`)
- Markdown headers and tables
- Technical acronyms (API, CLI, SDK, HTTP)
- Environment variables (KTTC_*, API_KEY)

### MQM Severity Scoring (Updated in v1.1)

KTTC uses industry-standard severity multipliers for MQM scoring:

| Severity | Multiplier | Impact |
|----------|------------|--------|
| Neutral | 0x | No penalty (informational) |
| Minor | 1x | Noticeable but doesn't affect understanding |
| Major | 5x | Affects understanding or quality |
| Critical | 25x | Severe meaning change or unusable |

**Formula:** `Quality Score = 100 - (ETPT / word_count × normalization_factor)`

Where `ETPT = Σ(error_count × severity_multiplier)`

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

**Quick mode (faster, cheaper):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --quick
```

**Show token usage and cost:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --show-cost
```

Output:
```
✓ Translation passed quality check (MQM Score: 96.5)
💰 Tokens: 1,245 (in: 890, out: 355) | Calls: 5 | Cost: $0.0234
```

**Use MQM profile for legal documents:**

```bash
kttc check contract.txt contract_ru.txt \
  --source-lang en \
  --target-lang ru \
  --profile legal
```

**Use medical profile with XLSX output:**

```bash
kttc check medical.txt medical_es.txt \
  --source-lang en \
  --target-lang es \
  --profile medical \
  --output report.xlsx
```

**Select specific agents:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --agents accuracy,terminology,hallucination
```

**Use full agent preset:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang zh \
  --agents full
```

**Custom YAML profile:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --profile ./profiles/custom.yaml
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

## kttc proofread

Proofread a text file for grammar, spelling, and punctuation errors.

This is an alias for `kttc check --self` with a simpler interface.
Uses school curriculum rules and optional LLM for context-aware checking.

### Syntax

```bash
kttc proofread FILE --lang CODE [OPTIONS]
```

### Options

- `--lang CODE`, `-l CODE` - Language code (e.g., 'ru', 'en', 'zh') (required)
- `--threshold FLOAT` - Quality threshold (0-100, default: 95.0)
- `--output PATH`, `-o PATH` - Output report file path
- `--provider TEXT` - LLM provider for context-aware checking
- `--verbose` - Verbose output

### Supported Languages

en, ru, zh, hi, fa

### Examples

**Proofread a Russian article:**

```bash
kttc proofread article.md --lang ru
```

**Proofread with strict threshold:**

```bash
kttc proofread article.md --lang ru --threshold 98
```

**Save report:**

```bash
kttc proofread article.md --lang ru --output report.json --verbose
```

---

## kttc lint

Quick lint check for common errors (no LLM, fast).

Fast rule-based checking using school curriculum rules and patterns.
Does not use LLM - ideal for CI/CD pipelines and pre-commit hooks.

### Syntax

```bash
kttc lint FILE --lang CODE [OPTIONS]
```

### Options

- `--lang CODE`, `-l CODE` - Language code (e.g., 'ru', 'en', 'zh') (required)
- `--strict` - Strict mode: fail on any error
- `--fix` - Show suggestions for fixing errors

### Exit Codes

- `0` - No errors found
- `1` - Errors found

### Examples

**Quick lint check:**

```bash
kttc lint article.md --lang ru
```

**Strict mode (fail on any error):**

```bash
kttc lint article.md --lang ru --strict
```

**Show fix suggestions:**

```bash
kttc lint article.md --lang ru --fix
```

---

## kttc report

Generate formatted reports from QA results.

### Syntax

```bash
kttc report INPUT_FILE [OPTIONS]
```

### Options

- `--format markdown|html`, `-f markdown|html` - Output format (default: markdown)
- `--output PATH`, `-o PATH` - Output file path (auto-generated if not specified)

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

Manage terminology glossaries with support for both project-local and user-global storage.

### Storage Locations

KTTC supports two-tier glossary storage:

- **Project glossaries** (default): `./glossaries/` - Stored in current project, can be version-controlled
- **User glossaries** (with `--user` flag): `~/.kttc/glossaries/` - Global glossaries available across all projects

**Search priority**: Project glossaries are checked first, then user glossaries.

### Subcommands

#### list

List all available glossaries from both locations:

```bash
kttc glossary list
```

Shows: name, location (project/user), term count, and file path.

#### show

Show glossary contents:

```bash
kttc glossary show NAME [OPTIONS]
```

**Options:**
- `--lang-pair SRC-TGT` - Filter by language pair (e.g., `en-ru`)
- `--limit N` - Limit number of entries shown

#### create

Create a new glossary from CSV or JSON file:

```bash
kttc glossary create NAME --from-csv FILE
# or
kttc glossary create NAME --from-json FILE
```

**Options:**
- `--from-csv PATH` - Create from CSV file (required if not using `--from-json`)
- `--from-json PATH` - Create from JSON file (required if not using `--from-csv`)
- `--user` - Save to user directory (`~/.kttc/glossaries/`) instead of project directory

**CSV format** (required columns):

```csv
source,target,source_lang,target_lang,context,notes
API,API,en,es,Keep as-is,Technical term
database,base de datos,en,es,,
```

**JSON format:**

```json
{
  "metadata": {
    "name": "technical",
    "description": "Technical terminology",
    "version": "1.0.0"
  },
  "entries": [
    {
      "source": "API",
      "target": "API",
      "source_lang": "en",
      "target_lang": "es",
      "context": "Keep as-is",
      "notes": "Technical term"
    }
  ]
}
```

#### merge

Merge multiple glossaries into one:

```bash
kttc glossary merge GLOSSARY1 GLOSSARY2 [...] --output NAME [OPTIONS]
```

**Options:**
- `--output NAME` - Output glossary name (required)
- `--user` - Save merged glossary to user directory

#### export

Export glossary to CSV or JSON:

```bash
kttc glossary export NAME [OPTIONS]
```

**Options:**
- `--format csv|json` - Export format (default: csv)
- `--output PATH` - Output file path (default: `{name}.{format}`)

#### validate

Validate glossary file format:

```bash
kttc glossary validate FILE
```

Checks for:
- Required fields (source, target, source_lang, target_lang)
- Duplicate entries
- Empty values
- Valid language codes

### Examples

**List all glossaries (project + user):**

```bash
kttc glossary list
```

Output:
```
📚 Project Glossaries (./glossaries/):
  • base (120 terms) - ./glossaries/base.json
  • technical (45 terms) - ./glossaries/technical.json

📚 User Glossaries (~/.kttc/glossaries/):
  • personal (30 terms) - ~/.kttc/glossaries/personal.json
```

**Create project glossary from CSV:**

```bash
kttc glossary create medical --from-csv medical-terms.csv
```

Saves to `./glossaries/medical.json` (can be committed to git).

**Create global user glossary:**

```bash
kttc glossary create personal --from-csv my-terms.csv --user
```

Saves to `~/.kttc/glossaries/personal.json` (available in all projects).

**Show glossary with filtering:**

```bash
kttc glossary show base --lang-pair en-ru --limit 10
```

**Merge multiple glossaries:**

```bash
kttc glossary merge base technical medical --output combined
```

Creates `./glossaries/combined.json` with all terms from three glossaries.

**Merge to user directory:**

```bash
kttc glossary merge base technical --output my-combined --user
```

Creates `~/.kttc/glossaries/my-combined.json`.

**Export to CSV:**

```bash
kttc glossary export technical --format csv --output technical-export.csv
```

**Validate glossary file:**

```bash
kttc glossary validate my-glossary.csv
```

Output:
```
✓ All required columns present
✓ No duplicate entries found
✓ All language codes valid
✓ No empty values
✅ Glossary is valid
```

### Using Glossaries in Translation Checks

Reference glossaries by name in `kttc check`:

```bash
# Auto-detect 'base' glossary (searches project, then user)
kttc check source.txt trans.txt --source-lang en --target-lang ru --glossary auto

# Use specific glossaries (comma-separated)
kttc check source.txt trans.txt --source-lang en --target-lang ru --glossary base,technical,medical

# Disable glossaries
kttc check source.txt trans.txt --source-lang en --target-lang ru --glossary none
```

**Search order**: KTTC searches for glossaries in project directory first, then user directory.

## kttc terminology

Access linguistic reference glossaries and validators for translation quality assessment.

### Subcommands

#### list

List all available linguistic reference glossaries.

```bash
kttc terminology list
```

**Options:**
- `--lang CODE`, `-l CODE` - Filter by language code

**Examples:**

```bash
kttc terminology list
kttc terminology list --lang ru
kttc terminology list --lang zh
```

#### show

Show contents of a linguistic reference glossary.

```bash
kttc terminology show LANGUAGE CATEGORY [OPTIONS]
```

**Arguments:**
- `LANGUAGE` - Language code (e.g., en, ru, zh)
- `CATEGORY` - Glossary category (e.g., mqm_core, russian_cases)

**Options:**
- `--limit N`, `-n N` - Maximum entries to display (default: 50)
- `--format FORMAT`, `-f FORMAT` - Output format: table or json (default: table)

**Examples:**

```bash
kttc terminology show en mqm_core
kttc terminology show ru russian_cases
kttc terminology show zh chinese_classifiers --limit 20
kttc terminology show en mqm_core --format json
```

#### search

Search across all terminology glossaries.

```bash
kttc terminology search QUERY [OPTIONS]
```

**Arguments:**
- `QUERY` - Search query

**Options:**
- `--lang CODE`, `-l CODE` - Filter by language code
- `--case-sensitive`, `-c` - Case-sensitive search

**Examples:**

```bash
kttc terminology search "mistranslation"
kttc terminology search "genitive" --lang ru
kttc terminology search "classifier" --lang zh
```

#### validate-error

Validate an MQM error type against the terminology glossary.

```bash
kttc terminology validate-error ERROR_TYPE [OPTIONS]
```

**Arguments:**
- `ERROR_TYPE` - MQM error type to validate

**Options:**
- `--lang CODE`, `-l CODE` - Language code (default: en)

**Examples:**

```bash
kttc terminology validate-error mistranslation
kttc terminology validate-error grammar --lang ru
kttc terminology validate-error untranslated --lang en
```

#### validators

List available language-specific validators.

```bash
kttc terminology validators
```

**Examples:**

```bash
kttc terminology validators
```

---
---

## Global Options

Available for all commands:

- `--ui-lang CODE`, `-L CODE` - UI language for CLI output (en, ru, zh, hi, fa) or 'auto' for system detection
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
