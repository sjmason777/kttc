# Core API

The core API provides the main classes and functions for translation quality checking.

## Basic Usage

```python
import asyncio
from kttc.agents import AgentOrchestrator
from kttc.llm import OpenAIProvider
from kttc.core import TranslationTask

async def check_translation():
    # Initialize provider
    llm = OpenAIProvider(api_key="sk-...")

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

    # Access results
    print(f"MQM Score: {report.mqm_score}")
    print(f"Issues: {len(report.errors)}")

    for error in report.errors:
        print(f"  {error.severity}: {error.description}")

asyncio.run(check_translation())
```

## Core Classes

### TranslationTask

::: kttc.core.TranslationTask
    options:
      show_source: false
      heading_level: 4

Main class for representing a translation task.

**Example:**

```python
from kttc.core import TranslationTask

task = TranslationTask(
    source_text="Hello, world!",
    translation="¡Hola, mundo!",
    source_lang="en",
    target_lang="es",
)
```

### QAReport

::: kttc.core.QAReport
    options:
      show_source: false
      heading_level: 4

Represents the results of quality analysis.

**Example:**

```python
# Report is returned by AgentOrchestrator
report = await orchestrator.evaluate(task)

print(f"MQM Score: {report.mqm_score}")
print(f"Total Errors: {len(report.errors)}")
```

### ErrorAnnotation

::: kttc.core.ErrorAnnotation
    options:
      show_source: false
      heading_level: 4

Represents a translation quality issue.

### ErrorSeverity

::: kttc.core.ErrorSeverity
    options:
      show_source: false
      heading_level: 4

Enumeration of error severity levels.

## Advanced Usage

### Using Glossary

```python
from kttc.core import Glossary, TermEntry

# Create glossary
glossary = Glossary(
    name="Technical Terms",
    version="1.0",
    terms=[
        TermEntry(
            source="API",
            target="API",
            context="Keep as-is"
        ),
        TermEntry(
            source="cloud",
            target="nube",
            context="Technology context"
        )
    ]
)

# Use in task
task = TranslationTask(
    source_text="Cloud API documentation",
    translation="Documentación de API en la nube",
    source_lang="en",
    target_lang="es",
    glossary=glossary
)

report = await orchestrator.evaluate(task)
```

### Batch Processing

```python
from kttc.core import BatchFileParser

async def process_batch():
    # Parse batch file
    parser = BatchFileParser()
    tasks = parser.parse_csv("translations.csv")

    # Evaluate each task
    reports = []
    for task in tasks:
        report = await orchestrator.evaluate(task)
        reports.append(report)

    # Calculate average MQM
    avg_mqm = sum(r.mqm_score for r in reports) / len(reports)
    print(f"Average MQM Score: {avg_mqm:.2f}")
```

### MQM Scoring

```python
from kttc.core import MQMScorer

# Create scorer
scorer = MQMScorer()

# Calculate MQM score from errors
errors = [...]  # List of ErrorAnnotation objects
mqm_score = scorer.calculate_score(errors, total_words=10)

print(f"MQM Score: {mqm_score:.2f}")
```
