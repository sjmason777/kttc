# KTTC API Documentation

API reference documentation for programmatic usage of KTTC.

## Overview

KTTC provides both a CLI interface and a Python API for translation quality assessment. This documentation covers the Python API for developers who want to integrate KTTC into their applications.

## Quick Start

```python
import asyncio
from kttc.agents.orchestrator import AgentOrchestrator
from kttc.llm.openai_provider import OpenAIProvider
from kttc.core.models import TranslationTask

async def check_quality():
    # Setup LLM provider
    llm = OpenAIProvider(api_key="your-api-key")

    # Create orchestrator
    orchestrator = AgentOrchestrator(llm)

    # Create task
    task = TranslationTask(
        source_text="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es",
    )

    # Evaluate
    report = await orchestrator.evaluate(task)

    print(f"MQM Score: {report.mqm_score}")
    print(f"Status: {report.status}")
    print(f"Errors: {len(report.errors)}")

# Run
asyncio.run(check_quality())
```

## Core Modules

### [Models](models.md)

Pydantic models for data validation:
- `TranslationTask` - Input data for evaluation
- `ErrorAnnotation` - Individual quality issue
- `QAReport` - Complete quality assessment report

### [Agents](agents.md)

Quality assessment agents:
- `BaseAgent` - Abstract base class
- `AccuracyAgent` - Semantic accuracy evaluation
- `FluencyAgent` - Grammatical fluency evaluation
- `TerminologyAgent` - Terminology consistency evaluation
- `AgentOrchestrator` - Coordinates multiple agents

### [LLM Providers](llm-providers.md)

LLM integration layer:
- `BaseLLMProvider` - Abstract interface
- `OpenAIProvider` - OpenAI/GPT integration
- `AnthropicProvider` - Anthropic/Claude integration
- `YandexProvider` - YandexGPT integration
- `GigaChatProvider` - GigaChat integration

### [MQM Scoring](mqm.md)

Quality metrics calculation:
- `MQMScorer` - Multidimensional Quality Metrics scoring engine

## Examples

See [examples directory](../../examples/) for complete examples:
- [Basic Usage](../../examples/basic_usage.py)
- [Batch Processing](../../examples/batch_processing.py)
- [Custom Agent](../../examples/custom_agent.py)

## API Reference

Detailed API reference for each module is available in the respective documentation files.

---

**Last Updated:** November 10, 2025
