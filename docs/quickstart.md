# Quick Start

This guide will get you up and running with KTTC in 5 minutes.

## Prerequisites

- Python 3.11 or higher installed
- At least one LLM provider API key (OpenAI, Anthropic, etc.)

If you haven't installed KTTC yet, see the [Installation](installation.md) guide.

## 1. Set Your API Key

Set your API key as an environment variable:

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

Or use a different provider:

```bash
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

## 2. Check a Single Translation

Create two text files:

```bash
echo "Hello, world! How are you today?" > source.txt
echo "¡Hola, mundo! ¿Cómo estás hoy?" > translation.txt
```

Check the translation quality:

```bash
kttc check source.txt translation.txt --source-lang en --target-lang es
```

**Output:**

```
✅ MQM Score: 96.5 (PASS - Excellent Quality)
📊 5 agents analyzed translation
⚠️  Found 2 minor issues, 0 major, 0 critical
✓ Quality threshold met (≥95.0)
```

## 3. Compare Multiple Translations

You can compare multiple translations of the same source:

```bash
echo "Bonjour, monde!" > translation_fr1.txt
echo "Salut, monde!" > translation_fr2.txt

kttc check source.txt translation_fr1.txt translation_fr2.txt \
    --source-lang en --target-lang fr
```

KTTC will automatically compare them and show which one is better.

## 4. Batch Processing

For processing multiple translations at once, create a CSV file:

```csv
source,translation,source_lang,target_lang
"Hello","Hola","en","es"
"Goodbye","Adiós","en","es"
"Thank you","Gracias","en","es"
```

Process the batch:

```bash
kttc batch --file translations.csv
```

## 5. Use a Glossary

Create a glossary file for custom terminology:

```json
{
  "terms": [
    {
      "source": "API",
      "target": "API",
      "context": "Keep as-is, do not translate"
    },
    {
      "source": "cloud",
      "target": "nube",
      "context": "Technology context"
    }
  ]
}
```

Save as `my-glossary.json` and use it:

```bash
kttc check source.txt translation.txt \
    --source-lang en --target-lang es \
    --glossary my-glossary.json
```

## 6. Python API

Use KTTC in your Python code:

```python
import asyncio
from kttc.agents import AgentOrchestrator
from kttc.llm import OpenAIProvider
from kttc.core import TranslationTask

async def check_translation():
    # Initialize LLM provider
    llm = OpenAIProvider(api_key="sk-...")

    # Create orchestrator
    orchestrator = AgentOrchestrator(llm)

    # Define translation task
    task = TranslationTask(
        source_text="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es",
    )

    # Evaluate
    report = await orchestrator.evaluate(task)

    # Print results
    print(f"MQM Score: {report.mqm_score}")
    print(f"Status: {report.status}")
    print(f"Issues found: {len(report.issues)}")

    for issue in report.issues:
        print(f"  - {issue.severity}: {issue.description}")

# Run
asyncio.run(check_translation())
```

## What's Next?

- [CLI Usage](cli-usage.md) - Learn all CLI commands and options
- [Configuration](configuration.md) - Configure KTTC for your needs
- [Supported Providers](providers.md) - Learn about different LLM providers
- [API Reference](api/core.md) - Explore the Python API

## Common Issues

### "No API key found"

Make sure you've set the environment variable:

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

Or create a `.env` file in your project directory.

### "ModuleNotFoundError: No module named 'kttc'"

Install KTTC:

```bash
pip install kttc
```

### Rate limits

If you hit rate limits, you can:
- Use a different provider
- Enable smart routing to use cheaper models
- Add delays between requests

See [Configuration](configuration.md) for more details.
