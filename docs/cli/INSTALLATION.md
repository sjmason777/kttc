# KTTC Installation Guide

## Quick Start

### Minimal Installation (Fast ⚡ ~30 seconds)

For basic translation QA with LLM agents only:

```bash
pip install kttc
```

**Includes:**
- ✅ CLI commands (check, translate, batch)
- ✅ LLM providers (OpenAI, Anthropic, GigaChat)
- ✅ Multi-agent QA system
- ✅ MQM scoring
- ✅ Beautiful terminal UI

**Size:** ~50MB

---

### Full Installation (Complete 🚀 ~5-10 minutes)

For all features including ML metrics:

```bash
pip install kttc[all]
```

**Includes everything above plus:**
- ✅ COMET metric (~2GB models)
- ✅ Sentence transformers (~500MB)
- ✅ BLEU and classical metrics
- ✅ Benchmark datasets
- ✅ Web UI server

**Size:** ~2.5GB total

---

## Smart Dependency Management

### Auto-Install on First Use

KTTC will prompt you to install missing dependencies when needed:

```bash
$ kttc benchmark --source text.txt ...

⚠️  COMET models not found!

╭─────────────── 📦 Missing Dependencies ───────────────╮
│                                                        │
│  The 'benchmark' command requires ML models (~2.5GB)  │
│                                                        │
│  Required packages:                                   │
│  • unbabel-comet (COMET metric, ~2GB)                 │
│  • sentence-transformers (~500MB)                     │
│  • sacrebleu (BLEU metric, lightweight)               │
│                                                        │
│  Estimated download size: ~2500MB                     │
│                                                        │
│  Installation options:                                │
│  1. Auto-install now (recommended)                    │
│  2. Manual: pip install kttc[metrics]                 │
│  3. Skip (limited functionality)                      │
│                                                        │
╰────────────────────────────────────────────────────────╯

[?] Install missing dependencies now? (Y/n): Y

📥 Installing benchmark dependencies...
━━━━━━━━━━━━━━━━━━━━━━━━━━ 35% ⏱️ 2m 15s remaining

✓ Successfully installed benchmark dependencies!
```

## Installation Speed Optimization

### Why is it slow?

- **ML models are large:** COMET models are ~2GB
- **First-time download:** Models cached after first download
- **Dependencies resolution:** pip checks version compatibility

### Speed it up:

#### 1. Use minimal install first

```bash
# Fast install (30 seconds)
pip install kttc

# Add features later as needed
pip install kttc[metrics]  # When you need COMET
```

#### 2. Use pip cache

```bash
# Models download once, then cached
# Location: ~/.cache/huggingface/

# Check cache size
du -sh ~/.cache/huggingface/
```

#### 3. Offline mode

After first download, models work offline:

```bash
# Works offline after initial download
export HF_DATASETS_OFFLINE=1
kttc benchmark ...
```

#### 4. Pre-download models

```bash
# Download models separately
python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
```

#### 5. Use faster pip resolver

```bash
# Use new pip resolver
pip install --use-feature=fast-deps kttc[all]
```

---

## Installation Scenarios

### Scenario 1: Quick Testing

**Goal:** Try KTTC quickly

```bash
pip install kttc
kttc check src.txt tgt.txt \
  --source-lang en --target-lang ru
```

**Time:** 30 seconds install + instant run

---

### Scenario 2: Production QA

**Goal:** High-quality evaluation with COMET

```bash
pip install kttc[metrics]

# First run downloads models (~5-10 min)
kttc check src.txt tgt.txt \
  --source-lang en --target-lang ru

# Subsequent runs are instant (models cached)
```

**Time:** 5-10 min first run, instant after

---

### Scenario 3: Provider Comparison

**Goal:** Benchmark multiple LLM providers

```bash
pip install kttc[benchmark]

kttc benchmark --source text.txt \
  --source-lang en --target-lang ru \
  --providers gigachat,openai,anthropic
```

**Time:** 5-10 min install, 2-5 min per benchmark

---

### Scenario 4: CI/CD Pipeline

**Goal:** Automated quality checks

```bash
# Dockerfile
FROM python:3.11-slim

# Install minimal KTTC
RUN pip install kttc

# Pre-download models (optional, for faster CI)
RUN pip install kttc[metrics] && \
    python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"

# Run checks
CMD kttc check /data/src.txt /data/tgt.txt --source-lang en --target-lang es
```

**Time:** Build once, run fast

---

## Troubleshooting

### Issue 1: Slow Installation

**Symptom:** `pip install` takes 10+ minutes

**Solutions:**

```bash
# Option 1: Install minimal first
pip install kttc  # Fast
pip install kttc[metrics]  # Slow but separate

# Option 2: Use pip cache
pip install --cache-dir ~/.pip-cache kttc[all]

# Option 3: Increase pip timeout
pip install --timeout 300 kttc[all]
```

---

### Issue 2: Network Issues

**Symptom:** Download fails or times out

**Solutions:**

```bash
# Option 1: Retry with increased timeout
pip install --timeout 600 --retries 10 kttc[metrics]

# Option 2: Use mirrors (if in China/Russia)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kttc

# Option 3: Download models manually
wget https://huggingface.co/Unbabel/wmt22-comet-da/resolve/main/pytorch_model.bin
```

---

### Issue 3: Disk Space

**Symptom:** "No space left on device"

**Solutions:**

```bash
# Check space
df -h

# Clean pip cache
pip cache purge

# Use minimal install
pip install kttc  # Only 50MB

# Clear huggingface cache
rm -rf ~/.cache/huggingface/
```

---

### Issue 4: Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'comet'`

**Solutions:**

```bash
# Reinstall with metrics
pip install --force-reinstall kttc[metrics]

# Or let KTTC auto-install
kttc benchmark ...  # Will prompt to install
```

---

## Platform-Specific Instructions

### macOS

```bash
# Use Python 3.11+
brew install python@3.11

# Install KTTC
python3.11 -m pip install kttc[all]
```

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.11
sudo apt install python3.11 python3.11-pip

# Install KTTC
python3.11 -m pip install kttc[all]
```

### Windows

```bash
# Install Python 3.11 from python.org

# Install KTTC
py -3.11 -m pip install kttc[all]
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Minimal install
RUN pip install kttc

# Or full install
RUN pip install kttc[all]

CMD ["kttc", "--help"]
```

---

## Verification

### Check Installation

```bash
# Verify KTTC installed
kttc --version

# Check available commands
kttc --help

# Test basic functionality
echo "Hello" > test.txt
echo "Привет" > test_ru.txt
kttc check test.txt test_ru.txt \
  --source-lang en --target-lang ru
```

### Check Optional Dependencies

```bash
# Check if metrics available
python -c "from kttc.utils.dependencies import has_metrics; print(has_metrics())"

# Check if webui available
python -c "from kttc.utils.dependencies import has_webui; print(has_webui())"
```

---

## Upgrading

### Upgrade KTTC

```bash
# Upgrade to latest version
pip install --upgrade kttc

# Upgrade with all dependencies
pip install --upgrade kttc[all]
```

### Upgrade Models

```bash
# Models auto-update when needed
# Or force update:
rm -rf ~/.cache/huggingface/hub/
kttc benchmark ...  # Re-downloads models
```

---

## Uninstallation

### Remove KTTC

```bash
# Uninstall package
pip uninstall kttc

# Remove cached models (optional)
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/kttc/
```

---

## Summary

| Install Command | Size | Time | Features |
|----------------|------|------|----------|
| `pip install kttc` | ~50MB | 30s | ✅ CLI, LLM agents, MQM |
| `pip install kttc[metrics]` | ~2.5GB | 5-10min | ✅ + COMET, BLEU |
| `pip install kttc[webui]` | ~60MB | 1min | ✅ + Web interface |
| `pip install kttc[benchmark]` | ~2.5GB | 5-10min | ✅ + Datasets |
| `pip install kttc[all]` | ~2.5GB | 5-10min | ✅ Everything |

**Recommendation:** Start with `pip install kttc`, add features as needed.

---

## Next Steps

After installation:

1. **Configure API keys** - See [Configuration Guide](./CONFIGURATION.md)
2. **Try examples** - See `examples/cli/README.md`
3. **Read CLI guide** - See [CLI Usage Guide](./CLI_USAGE.md)
4. **Run first check** - `kttc check --help`

Happy translating! ✨
