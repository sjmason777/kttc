# KTTC CLI - Beautiful Terminal Interface

## Overview

KTTC now features a beautiful, modern CLI built with Rich and Typer, inspired by industry-leading tools like Strix and Claude Code.

## Key Features

### ✨ Beautiful Visual Output

- **Rich panels** with color-coded status indicators
- **Tables** with syntax highlighting and Unicode borders
- **Progress bars** for long-running operations
- **Spinners** for real-time feedback
- **Error details** with severity color coding

### 🚀 Powerful Commands

1. **`check`** - Quality check single translation
2. **`translate`** - Translate with auto-refinement
3. **`batch`** - Process multiple files in parallel
4. **`benchmark`** - Compare LLM providers
5. **`compare`** - Compare multiple translations
6. **`report`** - Generate formatted reports

### 🎨 User Experience

- **Auto-completion** support
- **Detailed help** for every command
- **Colored output** (green=pass, red=fail)
- **Verbose mode** for debugging
- **CI/CD friendly** (exit codes, JSON output)

## Quick Start

### Installation

```bash
# Install with dev dependencies
python3.11 -m pip install -e ".[dev]"

# Verify installation
kttc --help
```

### First Command

```bash
# Check translation quality
kttc check \
  --source examples/cli/source_en.txt \
  --translation examples/cli/translation_ru_good.txt \
  --source-lang en \
  --target-lang ru \
  --verbose
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

### 1. check - Quality Check

```bash
kttc check --source src.txt --translation tgt.txt \
  --source-lang en --target-lang ru --threshold 95 --verbose
```

**Features:**
- MQM scoring
- Error categorization
- Auto-correction
- Multiple output formats

### 2. benchmark - Provider Comparison

```bash
kttc benchmark --source text.txt \
  --source-lang en --target-lang ru \
  --providers gigachat,openai,anthropic \
  --reference ref.txt
```

**Features:**
- COMET + MQM scoring
- Performance metrics
- Cost comparison
- Best provider recommendation

### 3. compare - Translation Comparison

```bash
kttc compare --source text.txt \
  --translation trans1.txt --translation trans2.txt \
  --source-lang en --target-lang ru --verbose
```

**Features:**
- Side-by-side comparison
- Quality ranking
- Detailed error analysis
- Best translation selection

### 4. translate - AI Translation

```bash
kttc translate --text "Hello world" \
  --source-lang en --target-lang ru \
  --threshold 95 --max-iterations 3
```

**Features:**
- TEaR loop (Translate-Estimate-Refine)
- Iterative improvement
- Quality convergence
- Auto-stop on threshold

### 5. batch - Batch Processing

```bash
kttc batch --source-dir ./sources \
  --translation-dir ./translations \
  --source-lang en --target-lang ru --parallel 8
```

**Features:**
- Parallel processing
- Progress tracking
- Aggregated reports
- CI/CD integration

### 6. report - Report Generation

```bash
kttc report results.json --format html --output report.html
```

**Features:**
- HTML reports
- Markdown export
- JSON data
- Custom styling

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
