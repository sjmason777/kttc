# 🎭 KTTC Demo Mode

Test the **Smart CLI** without spending tokens on API calls!

---

## What is Demo Mode?

Demo mode lets you test KTTC's new **Hybrid Format** without making real API calls. It uses simulated responses that mimic real LLM outputs.

### ✅ Perfect for:
- Testing the **smart auto-detection** features
- Verifying file paths and arguments
- Seeing how the beautiful UI looks
- Learning the new simplified syntax
- **Saving money** - Zero API token usage!
- Testing smart defaults (routing, glossary, auto-format)

### 🎯 New in Hybrid Format:
- Auto-detects mode (single/compare/batch)
- Smart routing enabled by default
- Auto-glossary detection
- Auto-format from file extension

---

## How to Use Demo Mode

Add the `--demo` flag to any `kttc check` command:

```bash
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru \
  --demo \
  --verbose
```

**Output:**
```
🎭 Demo mode: Using simulated responses (no API calls)

╭──────────────────────── 📊 Quality Assessment Report ────────────────────────╮
│                                                                              │
│    Status:          ✓ PASS                                                   │
│    MQM Score:       100.00/100                                               │
│    Errors Found:    0                                                        │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Use Cases

### 1. Test Auto-Detection (New!)

```bash
# Test single file mode
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --demo

# Test compare mode (auto-detected!)
kttc check source.txt trans1.txt trans2.txt \
  --source-lang en --target-lang ru \
  --demo

# Test batch mode (auto-detected!)
kttc check translations.csv --demo
```

### 2. Test Smart Defaults

```bash
# See smart routing in action
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --demo --verbose

# Output shows:
# 🎯 Mode: single
# 📚 Glossary: base  ← Auto-detected!
# 🧠 Smart routing: enabled ← Default!
```

### 3. Test Auto-Format Detection

```bash
# HTML output (auto-detected from .html)
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --output report.html \
  --demo

# Markdown output (auto-detected from .md)
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --output report.md \
  --demo
```

### 4. Test Disabling Smart Features

```bash
# Turn off smart defaults
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --no-smart-routing \
  --glossary none \
  --demo --verbose

# Output shows features disabled
```

---

## What Demo Mode Does

✅ **Affects:**
- API calls (simulated, no real LLM requests)
- Token usage (zero cost)
- QA analysis (returns demo data)

❌ **Does NOT affect:**
- Extension status checks (always shown)
- File loading and validation
- Configuration display
- UI/output formatting

## Limitations

Demo mode has some limitations:

1. **Simulated Responses** - Not real QA analysis, just demo data
2. **No Real Errors** - Won't detect actual translation issues
3. **Fixed Scores** - Always returns MQM 100.00
4. **No Persistence** - JSON output will contain demo data
5. **Extension warnings still shown** - You'll see missing extension info even in demo mode

---

## When to Use Real Mode

Use real API calls (without `--demo`) when you need:

- ✅ Actual translation quality assessment
- ✅ Real error detection and MQM scoring
- ✅ Production QA workflows
- ✅ Reliable metrics for decision-making

---

## Cost Comparison

### Demo Mode
```bash
kttc check file.txt trans.txt \
  --source-lang en --target-lang ru \
  --demo
```
**Cost:** $0.00 (no API calls)

### Real Mode with Cheapest Model
```bash
# Using Claude Haiku (cheapest Anthropic model)
export KTTC_DEFAULT_MODEL=claude-3-5-haiku-20241022

kttc check file.txt trans.txt \
  --source-lang en --target-lang ru \
  --provider anthropic
```
**Cost:** ~$0.001-0.01 per check (depending on text size)

---

## Quick Reference

```bash
# Demo mode - no API calls
kttc check SOURCE TRANSLATION [OPTIONS] --demo

# Real mode - uses configured provider
kttc check SOURCE TRANSLATION [OPTIONS]

# Test different providers in demo mode
kttc check SOURCE TRANSLATION --provider openai --demo [OPTIONS]
kttc check SOURCE TRANSLATION --provider anthropic --demo [OPTIONS]
kttc check SOURCE TRANSLATION --provider gigachat --demo [OPTIONS]
```

---

## Tips

1. **Use demo for learning** - Experiment with flags without cost
2. **Test file paths** - Verify files load before using real API
3. **Check configuration** - Use `--verbose` to see settings
4. **Switch to real** - Remove `--demo` when ready for actual QA

---

## Support

For real QA analysis, see:
- `TESTING_GUIDE.md` - Setup for real API testing
- `CLI_USAGE.md` - Full command documentation
- `.env.example` - Configuration template

Happy testing! 🧪✨
