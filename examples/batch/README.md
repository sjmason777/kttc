# Batch Processing Examples

This directory contains example batch files for testing KTTC's batch processing capabilities.

## File Formats

### CSV Format (`translations.csv`)

Simple CSV format with required columns:

```csv
source,translation,source_lang,target_lang,domain
"Hello, world!","¡Hola, mundo!",en,es,general
...
```

**Required columns:**
- `source` - Source text
- `translation` - Translation text
- `source_lang` - Source language code (ISO 639-1)
- `target_lang` - Target language code (ISO 639-1)

**Optional columns:**
- `domain` - Domain category (general, technical, medical, legal, etc.)
- `context` - JSON string with additional context

**Usage:**
```bash
kttc batch --file examples/batch/translations.csv --output report.json
```

---

### JSON Format (`translations.json`)

Array of translation objects:

```json
[
  {
    "source": "Hello, world!",
    "translation": "¡Hola, mundo!",
    "source_lang": "en",
    "target_lang": "es",
    "domain": "general",
    "context": {"complexity": "simple"}
  },
  ...
]
```

**Required fields:**
- `source` (or `source_text`) - Source text
- `translation` - Translation text
- `source_lang` - Source language code
- `target_lang` - Target language code

**Optional fields:**
- `domain` - Domain category
- `context` - Object with additional context

**Usage:**
```bash
kttc batch --file examples/batch/translations.json --batch-size 50
```

---

### JSONL Format (`translations.jsonl`)

One JSON object per line (JSON Lines):

```jsonl
{"source": "Hello, world!", "translation": "¡Hola, mundo!", "source_lang": "en", "target_lang": "es"}
{"source": "Goodbye", "translation": "Adiós", "source_lang": "en", "target_lang": "es"}
```

**Benefits:**
- Streamable for very large files
- Easy to append new translations
- Can process line-by-line

**Usage:**
```bash
kttc batch --file examples/batch/translations.jsonl --parallel 4
```

---

## Advanced Options

### Grouping by Language Pair

Automatically groups translations by language pair for optimal processing:

```bash
kttc batch --file translations.csv --group-by language-pair
```

### Batch Size Control

Process translations in batches of specified size:

```bash
kttc batch --file translations.json --batch-size 25
```

### Parallel Processing

Control number of parallel workers:

```bash
kttc batch --file translations.csv --parallel 8
```

### Output Formats

Generate reports in different formats:

```bash
# JSON report (default)
kttc batch --file translations.csv --output report.json

# Generate markdown report from JSON
kttc report report.json --format markdown --output report.md

# Generate HTML report
kttc report report.json --format html --output report.html
```

---

## Example Workflow

1. **Prepare batch file:**
   ```bash
   # Create CSV with your translations
   cat > my_translations.csv << EOF
   source,translation,source_lang,target_lang,domain
   "API key","Clave API",en,es,technical
   "User authentication","Autenticación de usuario",en,es,technical
   EOF
   ```

2. **Run batch evaluation:**
   ```bash
   kttc batch --file my_translations.csv --output results.json
   ```

3. **View results:**
   ```bash
   # Generate human-readable report
   kttc report results.json --format markdown

   # View in terminal
   cat results.md
   ```

4. **Filter by status:**
   ```bash
   # Extract only failed translations
   jq '.files[] | select(.status == "fail")' results.json
   ```

---

## Tips

- **Large files:** Use JSONL format for files with 1000+ translations
- **Mixed language pairs:** File can contain multiple language pairs - they'll be grouped automatically
- **Domain-specific:** Use `domain` field to enable domain-specific quality thresholds
- **Resume processing:** If interrupted, batch processing can resume from last checkpoint
- **Cost optimization:** Use `--batch-size` to balance speed vs. API cost

---

## Sample Data

The example files contain 10 translations across:
- 2 language pairs (EN→ES, EN→RU)
- 5 domains (general, technical, medical, legal)
- Mix of complexity levels

Perfect for testing batch processing features!
