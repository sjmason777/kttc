# KTTC CLI Demo Commands

This directory contains demo files and commands to test the new compact CLI formatter.

## Demo Files

### Source Files
- `source_en.txt` - Short English text (pangram)
- `source_en_realistic.txt` - **Realistic text** (AI/ML in healthcare, ~1300 chars, 4 paragraphs)

### Translation Files
- `translation_ru_bad.txt` - Bad translation with mixed languages
- `translation_ru_realistic_bad.txt` - **Realistic bad translation** (many errors: terminology, grammar, transliteration)
- `translation_ru_realistic_medium.txt` - **Medium quality translation** (some errors)
- `translation_ru_realistic_good.txt` - **Good quality translation** (minimal errors)

### Batch Files
- `batch_demo.csv` - CSV file with 5 translation pairs for batch processing

## Quick Start

Run all demos interactively:
```bash
bash examples/cli/demo_commands.sh
```

Or run individual commands below ↓

---

## Individual Commands

### 1. CHECK - Compact Mode (Default)
**Quick quality check, ~10-15 lines output**

```bash
python3.11 -m kttc check \
  examples/cli/source_en_realistic.txt \
  examples/cli/translation_ru_realistic_bad.txt \
  --source-lang en --target-lang ru \
  --demo
```

**Expected output:**
```
Translation Quality Check: en → ru

● ✗ FAIL  |  MQM: 75.2/100  |  Errors: 12 (C:2 M:5 m:5)
● Metrics: chrF: 62.3 | BLEU: 35.8 | TER: 58.9 | Rule-based: 72/100

Issues Found
Category      Severity   Description
Terminology   CRITICAL   Incorrect term: "искусственный разум"
Accuracy      CRITICAL   Mistranslation of "healthcare providers"
...
```

---

### 2. CHECK - Verbose Mode
**Detailed output with all metrics, ~25-35 lines**

```bash
python3.11 -m kttc check \
  examples/cli/source_en_realistic.txt \
  examples/cli/translation_ru_realistic_bad.txt \
  --source-lang en --target-lang ru \
  --verbose \
  --demo
```

**Shows additional details:**
- Configuration panel
- Step-by-step progress
- Confidence scores
- Agent agreement metrics
- Per-agent scores
- Domain detection
- NLP insights

---

### 3. CHECK - Good Translation (Should PASS)

```bash
python3.11 -m kttc check \
  examples/cli/source_en_realistic.txt \
  examples/cli/translation_ru_realistic_good.txt \
  --source-lang en --target-lang ru \
  --demo
```

**Expected output:**
```
Translation Quality Check: en → ru

● ✓ PASS  |  MQM: 96.8/100  |  Errors: 1 (C:0 M:0 m:1)
● Metrics: chrF: 88.5 | BLEU: 72.3 | TER: 91.2 | Rule-based: 98/100

Issues Found
Category   Severity   Description
Fluency    MINOR      Minor stylistic improvement possible
```

---

### 4. COMPARE - Multiple Translations (Compact)
**Compare 3 translations side-by-side**

```bash
python3.11 -m kttc compare \
  --source examples/cli/source_en_realistic.txt \
  --translation examples/cli/translation_ru_realistic_good.txt \
  --translation examples/cli/translation_ru_realistic_medium.txt \
  --translation examples/cli/translation_ru_realistic_bad.txt \
  --source-lang en --target-lang ru \
  --demo
```

**Expected output:**
```
Translation Comparison: en → ru

● Compared: 3 translations  |  Avg MQM: 84.3  |  Best: translation_ru_realistic_good (96.8)

Translation                        MQM    Errors      Status
translation_ru_realistic_good      96.8   C:0 M:0 m:1  ✓
translation_ru_realistic_medium    86.5   C:0 M:3 m:4  ✗
translation_ru_realistic_bad       75.2   C:2 M:5 m:5  ✗
```

---

### 5. COMPARE - Verbose Mode

```bash
python3.11 -m kttc compare \
  --source examples/cli/source_en_realistic.txt \
  --translation examples/cli/translation_ru_realistic_good.txt \
  --translation examples/cli/translation_ru_realistic_bad.txt \
  --source-lang en --target-lang ru \
  --verbose \
  --demo
```

**Shows detailed errors for each translation**

---

### 6. BATCH - Process CSV File

```bash
python3.11 -m kttc batch \
  --file examples/cli/batch_demo.csv \
  --output /tmp/batch_results.json \
  --demo
```

**Expected output:**
```
Batch Processing Complete

● ✓ Total: 5  |  Passed: 3  |  Failed: 2  |  Pass rate: 60%
● Avg MQM: 87.2/100  |  Total errors: 8
```

---

### 7. BENCHMARK - Compare Providers

```bash
python3.11 -m kttc benchmark \
  --source examples/cli/source_en.txt \
  --source-lang en --target-lang ru \
  --providers openai,anthropic \
  --demo
```

**Expected output:**
```
Provider Benchmark: en → ru

● Tested: 2 providers  |  Avg MQM: 92.5  |  Avg time: 3.2s  |  Best: anthropic

Provider    MQM    Errors      Time   Status
anthropic   94.5   C:0 M:1 m:0  2.8s   ✓
openai      90.5   C:0 M:2 m:1  3.5s   ✓

✓ Recommendation: Use anthropic (MQM: 94.5, Time: 2.8s)
```

---

### 8. OUTPUT Formats

**JSON output:**
```bash
python3.11 -m kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_bad.txt \
  --source-lang en --target-lang ru \
  --output /tmp/report.json \
  --demo

cat /tmp/report.json | jq
```

**Markdown output:**
```bash
python3.11 -m kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_bad.txt \
  --source-lang en --target-lang ru \
  --output /tmp/report.md \
  --demo

cat /tmp/report.md
```

**HTML output:**
```bash
python3.11 -m kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_bad.txt \
  --source-lang en --target-lang ru \
  --output /tmp/report.html \
  --demo

open /tmp/report.html  # macOS
# or
xdg-open /tmp/report.html  # Linux
```

---

## Testing Different Scenarios

### Test PASS scenario
```bash
python3.11 -m kttc check \
  examples/cli/source_en_realistic.txt \
  examples/cli/translation_ru_realistic_good.txt \
  --source-lang en --target-lang ru \
  --demo
```

### Test FAIL scenario
```bash
python3.11 -m kttc check \
  examples/cli/source_en_realistic.txt \
  examples/cli/translation_ru_realistic_bad.txt \
  --source-lang en --target-lang ru \
  --demo
```

### Test with reference translation
```bash
python3.11 -m kttc check \
  examples/cli/source_en_realistic.txt \
  examples/cli/translation_ru_realistic_medium.txt \
  --source-lang en --target-lang ru \
  --reference examples/cli/translation_ru_realistic_good.txt \
  --demo
```

---

## Key Features to Observe

### Compact Mode (Default)
- ✅ Fits on one screen (~10-15 lines)
- ✅ All essential information visible
- ✅ No scrolling needed
- ✅ Quick scan of status and metrics
- ✅ Error table with key details

### Verbose Mode
- ✅ Detailed configuration
- ✅ Step-by-step progress
- ✅ Confidence and agreement metrics
- ✅ Per-agent scores
- ✅ Domain detection
- ✅ Full error descriptions

### Consistency
- ✅ Same format across all commands
- ✅ Unified color scheme
- ✅ Predictable structure
- ✅ Easy to parse visually

---

## Notes

- **Demo Mode**: All commands use `--demo` flag to simulate API responses without actual API calls
- **Real Usage**: Remove `--demo` and set API keys in `.env` file
- **Terminal Width**: Recommended 100+ columns for best display
- **Colors**: Terminal must support ANSI colors

---

## Comparison: Before vs After

### Before (OLD formatter, ~28 lines):
```
✓ Translation Quality Check
Evaluating translation quality with multi-agent AI system

┌─ KTTC Configuration ─────────┐
│ Source File:     source.txt   │
│ Translation:     trans.txt    │
│ Languages:       en → ru      │
│ Threshold:       95           │
└───────────────────────────────┘

✓ Step 1/3: Linguistic analysis complete
✓ Step 2/3: Quality assessment complete
✓ Step 3/3: Report ready

┌─ Quality Assessment Report ──┐
│ Status:          ✗ FAIL      │
│ MQM Score:       87.5/100    │
│ Total Issues:    3            │
│ Confidence:      0.85 (high) │
│ Agent Agreement: 90%          │
│ Issue Breakdown: C:1 M:1 m:1 │
└───────────────────────────────┘

[Large error table...]
```

### After (NEW formatter, ~12 lines):
```
Translation Quality Check: en → ru

● ✗ FAIL  |  MQM: 87.5/100  |  Errors: 3 (C:1 M:1 m:1)
● Metrics: chrF: 68.5 | BLEU: 42.3 | TER: 71.9 | Rule-based: 85/100

Issues Found
Category   Severity   Description
Accuracy   CRITICAL   Incorrect translation
Grammar    MAJOR      Case agreement error
Fluency    MINOR      Awkward phrasing
```

**Result: 60% less vertical space, same information!**
