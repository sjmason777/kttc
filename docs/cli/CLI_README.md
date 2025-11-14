# KTTC CLI - Smart & Beautiful Terminal Interface

## Overview

KTTC features a modern, intelligent CLI with **Hybrid Format** - one smart command that auto-detects what you want to do!

## ✨ What's New in Hybrid Format

### 🎯 Smart `check` Command
One command to rule them all! Auto-detects mode based on your input:
- **Single file** → Quality check
- **Multiple files** → Comparison mode
- **CSV/JSON** → Batch processing
- **Directory** → Batch processing

### 🚀 Smart Defaults (Auto-Enabled)
- ✅ **Smart routing** - Saves money by using cheaper models for simple texts
- ✅ **Auto-glossary** - Automatically uses `base` glossary if it exists
- ✅ **Auto-format** - Detects output format from file extension (.html, .md, .json)

### 🎨 User Experience
- **Beautiful visuals** - Rich panels, tables, and progress bars
- **Auto-detection** - No need to remember which command to use
- **Smart defaults** - Works great out of the box
- **Backwards compatible** - Old commands (`batch`, `compare`) still work
- **CI/CD friendly** - Exit codes and JSON output

## Quick Start

### Installation

```bash
# Install KTTC
pip install kttc

# Or for development
python3.11 -m pip install -e ".[dev]"

# Verify installation
kttc --help
```

### Your First Command

```bash
# 🎯 Smart check - it figures out what you want!
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru

# That's it! Smart routing, glossary auto-detection enabled by default
```

### Compare Translations (Auto-Detected)

```bash
# Just add more files - compare mode activates automatically!
kttc check \
  examples/cli/source_en.txt \
  examples/cli/translation_ru_good.txt \
  examples/cli/translation_ru_bad.txt \
  --source-lang en \
  --target-lang ru

# Shows comparison table automatically
```

### Batch Process (Auto-Detected)

```bash
# CSV file? Batch mode activated!
kttc check examples/batch/translations.csv

# Or use directories
kttc check source_dir/ translation_dir/ \
  --source-lang en --target-lang ru
```

### Output Example

```
╭────────────── ✓ Translation Quality Check ──────────────╮
│ Evaluating translation quality with multi-agent AI      │
│ system                                                   │
╰──────────────────────────────────────────────────────────╯

╭─────────────── KTTC Configuration ────────────────────╮
│ Source File          examples/cli/source_en.txt          │
│ Translation File     examples/cli/translation_ru_good... │
│ Languages            en → ru                             │
│ Quality Threshold    95.0                                │
╰──────────────────────────────────────────────────────────╯

ℹ Running multi-agent QA system...

╭────────────── 📊 Quality Assessment Report ──────────────╮
│ Status:       ✓ PASS                                     │
│ MQM Score:    96.50/100                                  │
│ Errors Found: 2                                          │
│ Error Breakdown: Critical: 0 | Major: 0 | Minor: 2      │
╰──────────────────────────────────────────────────────────╯
```

## Architecture

### Tech Stack

- **Typer** - CLI framework (from creator of FastAPI)
- **Rich** - Beautiful terminal formatting
- **Pydantic** - Data validation
- **asyncio** - Concurrent operations

### Project Structure

```
kttc/
├── src/kttc/
│   ├── cli/
│   │   ├── main.py              # Main CLI app
│   │   ├── ui.py                # Rich UI components
│   │   └── commands/
│   │       ├── benchmark.py     # Provider comparison
│   │       └── compare.py       # Translation comparison
│   ├── core/                    # Core QA logic
│   ├── agents/                  # Multi-agent system
│   ├── llm/                     # LLM providers
│   └── metrics/                 # Quality metrics
├── examples/cli/                # Example files
└── docs/CLI_USAGE.md           # Full documentation
```

## Command Overview

### 🎯 check - Smart Quality Check (Hybrid)

The `check` command is your **one-stop solution** - it auto-detects the mode:

```bash
# Single file check
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru

# Compare mode (2+ translations) - AUTO-DETECTED!
kttc check source.txt trans1.txt trans2.txt \
  --source-lang en --target-lang ru

# Batch mode (CSV) - AUTO-DETECTED!
kttc check translations.csv

# Batch mode (directories) - AUTO-DETECTED!
kttc check source_dir/ trans_dir/ \
  --source-lang en --target-lang ru
```

**🚀 Smart Defaults (Auto-Enabled):**
- ✅ Smart routing (--no-smart-routing to disable)
- ✅ Auto-glossary detection (--glossary none to disable)
- ✅ Auto-format from extension (--format to override)

**Features:**
- MQM scoring with multi-agent QA
- Error categorization (critical/major/minor)
- Auto-correction support
- Multiple output formats (text/json/markdown/html)

### 2. translate - AI Translation with TEaR Loop

```bash
kttc translate --text "Hello world" \
  --source-lang en --target-lang ru \
  --threshold 95 --max-iterations 3
```

**Features:**
- TEaR loop (Translate-Estimate-Refine)
- Iterative quality improvement
- Auto-stop when threshold met
- Built-in quality validation

### 3. glossary - Manage Translation Glossaries

```bash
# List available glossaries
kttc glossary list

# Show glossary contents
kttc glossary show base

# Create new glossary
kttc glossary create my-terms --from-csv terms.csv
```

**Features:**
- Multiple glossary support
- Auto-detection in check command
- CSV/JSON import/export
- Version control friendly

---

## 🔄 Legacy Commands (Still Available)

These commands still work for backwards compatibility. However, we recommend using the smart `check` command instead.

### compare - Dedicated Comparison

```bash
kttc compare --source text.txt \
  --translation trans1.txt --translation trans2.txt \
  --source-lang en --target-lang ru
```

💡 **New way:** `kttc check text.txt trans1.txt trans2.txt --source-lang en --target-lang ru`

**Features:**
- Side-by-side comparison
- Quality ranking
- Detailed error analysis
- Best translation selection

### batch - Dedicated Batch Processing

```bash
kttc batch --source-dir ./sources \
  --translation-dir ./translations \
  --source-lang en --target-lang ru --parallel 8

# Or with file
kttc batch --file translations.csv
```

💡 **New way:** `kttc check translations.csv` or `kttc check source_dir/ trans_dir/`

**Features:**
- Parallel processing
- Progress tracking
- Aggregated reports
- CI/CD integration

### benchmark - Provider Comparison

```bash
kttc benchmark --source text.txt \
  --source-lang en --target-lang ru \
  --providers gigachat,openai,anthropic
```

**Features:**
- COMET + MQM scoring
- Performance metrics
- Cost comparison
- Best provider recommendation

## Configuration

### Environment Variables

```bash
# .env file
KTTC_OPENAI_API_KEY=sk-...
KTTC_ANTHROPIC_API_KEY=sk-ant-...
KTTC_GIGACHAT_CLIENT_ID=...
KTTC_GIGACHAT_CLIENT_SECRET=...
```

### Default Settings

```bash
KTTC_DEFAULT_LLM_PROVIDER=gigachat
KTTC_DEFAULT_MODEL=gpt-4
KTTC_DEFAULT_TEMPERATURE=0.3
```

## Examples

See `examples/cli/README.md` for comprehensive examples:

- Basic quality checks
- Provider benchmarking
- Translation comparison
- Batch processing
- CI/CD integration
- Scripting with JSON

## Visual Features

### Color Coding

- 🟢 **Green** - Pass, success, high scores
- 🔴 **Red** - Fail, errors, low scores
- 🟡 **Yellow** - Warnings, medium scores
- 🔵 **Cyan** - Info, neutral content

### Progress Indicators

- ⏳ **Spinners** - Indefinite tasks
- ━━━━ **Progress bars** - Batch operations
- ✓/✗ **Status icons** - Completion status
- ⏱️ **Time tracking** - Duration metrics

### Tables

```
╭─────────── Provider Benchmark ───────────╮
│ Provider  │ COMET │ MQM   │ Duration    │
├───────────┼───────┼───────┼─────────────┤
│ gigachat  │ 91.20 │ 92.50 │ 1.2s        │
│ openai    │ 93.40 │ 94.80 │ 2.1s        │
╰───────────────────────────────────────────╯
```

## Comparison with Other Tools

### vs Claude Code

- ✅ Same React Ink-inspired visual design
- ✅ Real-time progress indicators
- ✅ Beautiful error reporting
- ✅ Interactive commands

### vs Strix

- ✅ Rich panels and tables
- ✅ Textual-ready architecture
- ✅ Signal handling
- ✅ Telemetry integration

### Best Practices 2025

- ✅ Progressive discovery (guides users)
- ✅ Context inference (smart defaults)
- ✅ Modern terminal capabilities (24-bit color)
- ✅ Mouse support ready (via Textual)
- ✅ Pager integration for long output
- ✅ Pipeline-friendly (--format json)

## Development Roadmap

### Current (Phase 1)

- ✅ Rich UI components
- ✅ All core commands
- ✅ Beautiful visual output
- ✅ Documentation

### Future (Phase 2)

- ⏳ **Textual TUI** - Full interactive mode
- ⏳ **Interactive comparison** - Side-by-side with keyboard
- ⏳ **Watch mode** - Auto-check on file changes
- ⏳ **Plugin system** - Custom commands
- ⏳ **Web mode** - Terminal + browser (Textual feature)

## CI/CD Integration

```bash
# .github/workflows/translation-qa.yml
- name: Check translation quality
  run: |
    kttc check \
      --source src.txt --translation tgt.txt \
      --source-lang en --target-lang ru \
      --threshold 95 --output results.json

    # Fail pipeline if quality < threshold
    exit $?
```

## Performance

- **Parallel processing** - Use `--parallel N` for batch
- **Async operations** - All LLM calls are async
- **Progress tracking** - Real-time feedback
- **Optimized metrics** - COMET caching

## Support

- 📖 **Full docs**: `docs/CLI_USAGE.md`
- 💡 **Examples**: `examples/cli/`
- 🐛 **Issues**: GitHub Issues
- 💬 **Email**: dev@kt.tc

## Credits

Inspired by:
- [Strix](https://github.com/usestrix/strix) - Beautiful Python TUI
- [Claude Code](https://github.com/anthropics/claude-code) - React Ink excellence
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal output
- [Typer](https://github.com/tiangolo/typer) - Modern CLI framework

## License

Apache 2.0

---

**Try it now:**

```bash
python3.11 -m kttc --help
```

Happy translating! ✨
