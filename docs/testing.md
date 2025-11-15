# Testing

KTTC has a comprehensive test suite covering unit tests, integration tests, and end-to-end tests.

## Running Tests

### All Tests

```bash
python3.11 -m pytest
```

### Unit Tests Only

Unit tests don't require API keys or external dependencies:

```bash
python3.11 -m pytest tests/unit/
```

### Integration Tests

Integration tests require API keys:

```bash
# Set API keys first
export KTTC_OPENAI_API_KEY="sk-..."
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."

# Run integration tests
python3.11 -m pytest tests/integration/
```

### Specific Test File

```bash
python3.11 -m pytest tests/unit/test_core.py
```

### Specific Test Function

```bash
python3.11 -m pytest tests/unit/test_core.py::test_translation_task_creation
```

### With Coverage

```bash
# Generate coverage report
python3.11 -m pytest --cov=kttc --cov-report=html

# View report
open htmlcov/index.html
```

## Test Structure

```
tests/
├── unit/               # Unit tests (no external dependencies)
│   ├── test_core.py
│   ├── test_agents.py
│   └── test_llm.py
├── integration/        # Integration tests (require API keys)
│   ├── test_openai_integration.py
│   ├── test_anthropic_integration.py
│   └── test_gigachat_integration.py
└── e2e/               # End-to-end tests
    └── test_cli.py
```

## Writing Tests

### Unit Test Example

```python
import pytest
from kttc.core import TranslationTask, QualityReport

def test_translation_task_creation():
    """Test creating a translation task."""
    task = TranslationTask(
        source_text="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es"
    )

    assert task.source_text == "Hello, world!"
    assert task.translation == "¡Hola, mundo!"
    assert task.source_lang == "en"
    assert task.target_lang == "es"

def test_quality_report_passed():
    """Test quality report pass/fail logic."""
    report = QualityReport(
        mqm_score=96.5,
        threshold=95.0,
        issues=[]
    )

    assert report.passed
    assert report.status == "PASS"
```

### Async Test Example

```python
import pytest
from kttc.agents import AgentOrchestrator
from kttc.llm import MockProvider

@pytest.mark.asyncio
async def test_agent_orchestrator():
    """Test agent orchestrator evaluation."""
    # Use mock provider for unit tests
    provider = MockProvider()
    orchestrator = AgentOrchestrator(provider)

    task = TranslationTask(
        source_text="Hello",
        translation="Hola",
        source_lang="en",
        target_lang="es"
    )

    report = await orchestrator.evaluate(task)

    assert report.mqm_score >= 0
    assert report.mqm_score <= 100
    assert isinstance(report.issues, list)
```

### Integration Test Example

```python
import pytest
import os
from kttc.agents import AgentOrchestrator
from kttc.llm import OpenAIProvider

@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_integration():
    """Test real OpenAI API integration."""
    # Skip if no API key
    api_key = os.getenv("KTTC_OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not set")

    provider = OpenAIProvider(api_key=api_key)
    orchestrator = AgentOrchestrator(provider)

    task = TranslationTask(
        source_text="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es"
    )

    report = await orchestrator.evaluate(task)

    assert report.mqm_score > 90  # Should be good translation
    assert report.passed
```

### Parametrized Tests

```python
@pytest.mark.parametrize("source,translation,expected_score", [
    ("Hello", "Hola", 95.0),
    ("Goodbye", "Adiós", 95.0),
    ("Thank you", "Gracias", 95.0),
])
def test_simple_translations(source, translation, expected_score):
    """Test various simple translations."""
    task = TranslationTask(
        source_text=source,
        translation=translation,
        source_lang="en",
        target_lang="es"
    )
    # Test logic here
```

### Fixtures

```python
@pytest.fixture
def mock_provider():
    """Fixture for mock LLM provider."""
    return MockProvider()

@pytest.fixture
def sample_task():
    """Fixture for sample translation task."""
    return TranslationTask(
        source_text="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es"
    )

def test_with_fixtures(mock_provider, sample_task):
    """Test using fixtures."""
    orchestrator = AgentOrchestrator(mock_provider)
    # Use fixtures in test
```

## Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_unit():
    """Unit test."""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test."""
    pass

@pytest.mark.slow
def test_slow():
    """Slow test."""
    pass
```

Run specific markers:

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Mocking

### Mock LLM Provider

```python
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_with_mock():
    """Test with mocked provider."""
    mock_provider = Mock()
    mock_provider.complete = AsyncMock(return_value="Mocked response")

    # Use mock in test
    response = await mock_provider.complete(messages=[])
    assert response == "Mocked response"
```

### Mock External API

```python
import httpx
from unittest.mock import patch

@pytest.mark.asyncio
async def test_api_call():
    """Test with mocked HTTP client."""
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"result": "success"}
        )

        # Test code that makes API call
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com")
            assert response.json()["result"] == "success"
```

## Coverage Requirements

- **Minimum coverage:** 80%
- **Target coverage:** 90%+
- **Critical paths:** 100%

Check coverage:

```bash
# Generate coverage report
pytest --cov=kttc --cov-report=term-missing

# Show lines not covered
pytest --cov=kttc --cov-report=html
open htmlcov/index.html
```

## Continuous Integration

Tests run automatically on:
- Every push to main
- Every pull request
- Daily scheduled runs

See `.github/workflows/ci.yml` for CI configuration.

## Test Best Practices

### 1. Test One Thing

```python
# Good
def test_task_creation():
    """Test task creation."""
    task = TranslationTask(...)
    assert task.source_text == "Hello"

def test_task_validation():
    """Test task validation."""
    with pytest.raises(ValueError):
        TranslationTask(source_text="", ...)

# Bad
def test_everything():
    """Test everything."""
    task = TranslationTask(...)
    assert task.source_text == "Hello"
    with pytest.raises(ValueError):
        TranslationTask(source_text="", ...)
```

### 2. Use Descriptive Names

```python
# Good
def test_translation_fails_with_empty_source():
    """Test that empty source text raises ValueError."""
    pass

# Bad
def test_1():
    """Test."""
    pass
```

### 3. Arrange-Act-Assert

```python
def test_example():
    """Test example."""
    # Arrange - set up test data
    task = TranslationTask(...)

    # Act - perform action
    result = process_task(task)

    # Assert - verify result
    assert result.success
```

### 4. Don't Test Implementation Details

```python
# Good - test behavior
def test_translation_quality():
    """Test translation quality is evaluated correctly."""
    report = evaluate_translation(...)
    assert report.mqm_score > 90

# Bad - test implementation
def test_internal_method():
    """Test internal method."""
    obj = MyClass()
    assert obj._internal_method() == "something"
```

## Debugging Tests

### Run with verbose output

```bash
pytest -v
```

### Show print statements

```bash
pytest -s
```

### Drop into debugger on failure

```bash
pytest --pdb
```

### Run last failed tests

```bash
pytest --lf
```

### Run only failed tests

```bash
pytest --failed-first
```
