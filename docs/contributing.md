# Contributing

We welcome contributions to KTTC! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/kttc.git
cd kttc
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Running Tests

```bash
# Run all tests
python3.11 -m pytest

# Run specific test file
python3.11 -m pytest tests/unit/test_core.py

# Run with coverage
python3.11 -m pytest --cov=kttc --cov-report=html

# Run integration tests (requires API keys)
python3.11 -m pytest tests/integration/
```

### Code Quality Checks

KTTC uses multiple tools to ensure code quality:

```bash
# Format code with Black
python3.11 -m black src/ tests/

# Check formatting
python3.11 -m black --check src/ tests/

# Lint with Ruff
python3.11 -m ruff check src/ tests/

# Fix linting issues
python3.11 -m ruff check src/ tests/ --fix

# Type checking with MyPy
python3.11 -m mypy src/

# Run all checks (what CI/CD runs)
pre-commit run --all-files
```

### Code Style Guidelines

- **Line length:** 100 characters
- **Formatting:** Use Black (configured in `pyproject.toml`)
- **Imports:** Use isort (configured in `pyproject.toml`)
- **Type hints:** All functions must have type hints
- **Docstrings:** Use Google style docstrings
- **Language:** All code, comments, and documentation in English

Example:

```python
def evaluate_translation(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
) -> QualityReport:
    """Evaluate translation quality.

    Args:
        source: Source text
        translation: Translation to evaluate
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'es')

    Returns:
        Quality report with MQM score and issues

    Raises:
        ValueError: If language codes are invalid
    """
    # Implementation
    pass
```

## Making Changes

### Adding a New Feature

1. **Create an issue** describing the feature
2. **Discuss the approach** in the issue
3. **Implement the feature** with tests
4. **Add documentation** to the relevant docs
5. **Submit a pull request**

### Fixing a Bug

1. **Create an issue** describing the bug (if not exists)
2. **Write a test** that reproduces the bug
3. **Fix the bug**
4. **Verify** the test now passes
5. **Submit a pull request**

### Adding Tests

Tests should be placed in the appropriate directory:

- `tests/unit/` - Unit tests (no external dependencies)
- `tests/integration/` - Integration tests (require API keys)
- `tests/e2e/` - End-to-end tests

Example test:

```python
import pytest
from kttc.core import TranslationTask

def test_translation_task_creation():
    """Test creating a translation task."""
    task = TranslationTask(
        source_text="Hello",
        translation="Hola",
        source_lang="en",
        target_lang="es"
    )

    assert task.source_text == "Hello"
    assert task.translation == "Hola"
    assert task.source_lang == "en"
    assert task.target_lang == "es"

@pytest.mark.asyncio
async def test_translation_evaluation():
    """Test evaluating a translation."""
    # Your async test here
    pass
```

## Pull Request Process

### Before Submitting

1. **Run all tests:** `pytest`
2. **Run quality checks:** `pre-commit run --all-files`
3. **Update documentation** if needed
4. **Add changelog entry** in `CHANGELOG.md`

### Submitting

1. **Push to your fork**
2. **Create pull request** on GitHub
3. **Fill in PR template** with:
   - Description of changes
   - Related issue number
   - Testing done
   - Screenshots (if UI changes)

### PR Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing done

## Checklist
- [ ] Code follows style guidelines
- [ ] Added/updated tests
- [ ] Added/updated documentation
- [ ] Changelog updated
```

## Coding Standards

### Type Checking

All code must pass strict type checking:

```python
# Good
def process_text(text: str) -> str:
    return text.upper()

# Bad
def process_text(text):
    return text.upper()
```

### Error Handling

Use specific exceptions:

```python
# Good
if not api_key:
    raise ValueError("API key is required")

# Bad
if not api_key:
    raise Exception("Error")
```

### Async/Await

Use async/await for I/O operations:

```python
# Good
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Bad
def fetch_data(url: str) -> dict:
    response = requests.get(url)
    return response.json()
```

## Documentation

### Updating Docs

Documentation is in the `docs/` directory using MkDocs:

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Serve docs locally
mkdocs serve

# View at http://127.0.0.1:8000
```

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description.

    Longer description if needed. Can span multiple
    lines and paragraphs.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When this happens
        TypeError: When that happens

    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release commit
4. Tag release: `git tag v0.2.0`
5. Push tags: `git push --tags`
6. GitHub Actions will build and publish to PyPI

## Getting Help

- **Questions:** Open a [Discussion](https://github.com/kttc-ai/kttc/discussions)
- **Bugs:** Open an [Issue](https://github.com/kttc-ai/kttc/issues)
- **Chat:** Join our Discord (coming soon)

## Code of Conduct

Please read our [Code of Conduct](https://github.com/kttc-ai/kttc/blob/main/CODE_OF_CONDUCT.md) before contributing.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
