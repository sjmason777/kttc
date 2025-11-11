# 🎭 KTTC Demo Mode

Test the CLI without spending tokens on API calls.

---

## What is Demo Mode?

Demo mode allows you to test KTTC CLI commands without making real API calls to LLM providers. It uses simulated responses that mimic real LLM outputs, so you can:

- ✅ Test CLI features and UI
- ✅ Verify file paths and arguments
- ✅ See how the output looks
- ✅ Learn command syntax
- ✅ **Save money** - No API token usage!
- ℹ️ **Note:** Extension status is always shown, regardless of demo mode

---

## How to Use Demo Mode

Add the `--demo` flag to any `kttc check` command:

```bash
kttc check \
  --demo \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru \
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

### 1. Testing CLI Installation

```bash
# Quick test that everything works
kttc check --demo \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_good.txt \
  --source-lang en --target-lang ru
```

### 2. Learning Command Syntax

```bash
# Try different flags without API cost
kttc check --demo \
  --source my_source.txt \
  --translation my_translation.txt \
  --source-lang en --target-lang ru \
  --threshold 90 \
  --verbose
```

### 3. Testing File Paths

```bash
# Verify your files are loaded correctly
kttc check --demo \
  --source path/to/source.txt \
  --translation path/to/translation.txt \
  --source-lang en --target-lang es \
  --verbose  # Shows file loading details
```

### 4. UI/Output Testing

```bash
# See how verbose output looks
kttc check --demo \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_good.txt \
  --source-lang en --target-lang ru \
  --verbose
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
kttc check --demo --source file.txt --translation trans.txt \
  --source-lang en --target-lang ru
```
**Cost:** $0.00 (no API calls)

### Real Mode with Cheapest Model
```bash
# Using Claude Haiku (cheapest Anthropic model)
export KTTC_DEFAULT_MODEL=claude-3-5-haiku-20241022

kttc check --source file.txt --translation trans.txt \
  --source-lang en --target-lang ru \
  --provider anthropic
```
**Cost:** ~$0.001-0.01 per check (depending on text size)

---

## Quick Reference

```bash
# Demo mode - no API calls
kttc check --demo [options...]

# Real mode - uses configured provider
kttc check [options...]

# Test different providers in demo mode
kttc check --demo --provider openai [options...]
kttc check --demo --provider anthropic [options...]
kttc check --demo --provider gigachat [options...]
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
