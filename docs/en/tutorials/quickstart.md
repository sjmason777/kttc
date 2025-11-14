# Quickstart Guide

Welcome to KTTC! This tutorial will guide you through your first translation quality check in under 5 minutes.

## What You'll Learn

- How to install KTTC
- How to set up your API key
- How to check translation quality
- How to interpret the results

## Prerequisites

- Python 3.11 or higher
- An OpenAI or Anthropic API key

## Step 1: Installation

Install KTTC using pip:

```bash
pip install kttc
```

This installs the core package (~50MB). For language-specific enhancements:

```bash
# English grammar checking (requires Java 17+)
pip install kttc[english]

# Chinese NLP features
pip install kttc[chinese]

# All language helpers
pip install kttc[all-languages]
```

## Step 2: Set Your API Key

Set your LLM provider API key:

```bash
# OpenAI (recommended for beginners)
export KTTC_OPENAI_API_KEY="sk-..."

# Or Anthropic
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

**Tip:** Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

## Step 3: Create Test Files

Create a source text file:

```bash
echo "Hello, world! This is a test." > source.txt
```

Create a translation file:

```bash
echo "¡Hola, mundo! Esto es una prueba." > translation.txt
```

## Step 4: Run Your First Quality Check

Run KTTC's smart check command:

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es
```

**Note:** `kttc check` auto-detects the operation mode:
- Single file → quality check
- Multiple translations → automatic comparison
- CSV/JSON → batch processing

## Step 5: Understand the Results

KTTC will output:

```
✓ Translation Quality Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Step 1/3: Linguistic analysis complete
✓ Step 2/3: Quality assessment complete
✓ Step 3/3: Report ready

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Quality Assessment Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ MQM Score: 96.5 (PASS - Excellent Quality)

📊 5 agents analyzed translation
⚠️  Found 2 minor issues, 0 major, 0 critical
✓ Quality threshold met (≥95.0)
```

### Understanding the MQM Score

- **95-100:** Excellent (production-ready)
- **90-94:** Good (minor fixes needed)
- **80-89:** Acceptable (revision needed)
- **<80:** Poor (significant rework required)

## What's Next?

Now that you've run your first quality check, explore:

- [Batch Processing](../guides/batch-processing.md) - Process multiple translations
- [Auto-Correction](../guides/auto-correction.md) - Automatically fix detected errors
- [Glossaries](../guides/glossary-management.md) - Use custom terminology
- [Smart Routing](../guides/smart-routing.md) - Optimize costs with intelligent model selection

## Troubleshooting

### "API key not found" Error

Make sure you've set the environment variable:

```bash
echo $KTTC_OPENAI_API_KEY
```

If empty, set it again and try in the same terminal session.

### "Module not found" Error

Ensure you installed KTTC:

```bash
pip install kttc
```

For language-specific features, install the extras:

```bash
pip install kttc[english]  # For LanguageTool
```

### Python Version Error

KTTC requires Python 3.11+. Check your version:

```bash
python3 --version
```

If you have 3.11 installed, use it explicitly:

```bash
python3.11 -m pip install kttc
```

## Demo Mode

Want to try KTTC without API calls?

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es \
  --demo
```

This uses simulated responses so you can explore the CLI without costs.
