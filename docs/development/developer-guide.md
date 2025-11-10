# KTTC Developer Guide

Welcome to the KTTC developer documentation! This guide will help you set up your development environment and contribute to KTTC.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Architecture](#project-architecture)
3. [Development Workflow](#development-workflow)
4. [Testing](#testing)
5. [Code Quality](#code-quality)
6. [Contributing](#contributing)
7. [Release Process](#release-process)

---

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)
- An API key for OpenAI or Anthropic (for testing)

### Clone the Repository

```bash
git clone https://github.com/kttc-ai/kttc.git
cd kttc
```

### Create Virtual Environment

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n kttc python=3.11
conda activate kttc
```

### Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs:
- Core dependencies (typer, rich, pydantic, etc.)
- Development tools (pytest, black, ruff, mypy)
- Pre-commit hooks

### Setup Pre-commit Hooks

```bash
pre-commit install
```

This automatically runs code quality checks before each commit.

### Verify Setup

```bash
# Run tests
pytest

# Check code quality
black --check src/ tests/
ruff check src/ tests/
mypy src/kttc --strict

# Test CLI
kttc --help
```

---

## Project Architecture

### Directory Structure

```
kttc/
├── src/
│   └── kttc/              # Main package
│       ├── __init__.py
│       ├── __main__.py    # Entry point
│       │
│       ├── cli/           # CLI interface (Typer)
│       │   ├── main.py    # Main CLI app
│       │   └── __init__.py
│       │
│       ├── agents/        # QA agents
│       │   ├── base.py              # Base agent class
│       │   ├── orchestrator.py     # Agent coordinator
│       │   ├── accuracy.py         # Accuracy agent
│       │   ├── fluency.py          # Fluency agent
│       │   ├── terminology.py      # Terminology agent
│       │   └── parser.py           # LLM response parser
│       │
│       ├── core/          # Core business logic
│       │   ├── models.py           # Pydantic models
│       │   └── mqm.py              # MQM scoring engine
│       │
│       ├── llm/           # LLM integrations
│       │   ├── base.py             # Base LLM provider
│       │   ├── openai_provider.py  # OpenAI implementation
│       │   ├── anthropic_provider.py # Claude implementation
│       │   ├── yandex_provider.py  # Yandex implementation
│       │   ├── gigachat_provider.py # GigaChat implementation
│       │   └── prompts/            # Prompt templates
│       │
│       └── utils/         # Utilities
│           └── config.py           # Configuration management
│
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests (future)
│   ├── e2e/               # End-to-end tests (future)
│   └── performance/       # Performance benchmarks
│
├── docs/                  # Documentation
│   ├── guides/            # User guides
│   ├── development/       # Developer docs
│   └── api/               # API documentation
│
├── .github/
│   └── workflows/         # CI/CD workflows
│
├── pyproject.toml         # Project configuration
├── README.md
└── CONTRIBUTING.md
```

### Key Components

#### 1. **CLI Layer** (`src/kttc/cli/`)

Built with [Typer](https://typer.tiangolo.com/), provides command-line interface:

```python
# src/kttc/cli/main.py
import typer

app = typer.Typer()

@app.command()
def check(source: str, translation: str, ...):
    """Check translation quality"""
    # Implementation
```

#### 2. **Agent System** (`src/kttc/agents/`)

Multi-agent architecture for parallel quality assessment:

- **BaseAgent**: Abstract base class for all agents
- **Orchestrator**: Coordinates multiple agents using `asyncio.gather()`
- **Specialized Agents**: Accuracy, Fluency, Terminology

```python
# src/kttc/agents/base.py
class BaseAgent(ABC):
    @abstractmethod
    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate translation and return errors"""
```

#### 3. **LLM Provider Layer** (`src/kttc/llm/`)

Abstract interface for different LLM providers:

```python
# src/kttc/llm/base.py
class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion"""
```

Implementations:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Yandex (YandexGPT)
- GigaChat

#### 4. **MQM Scoring** (`src/kttc/core/mqm.py`)

Multidimensional Quality Metrics scoring engine:

```python
class MQMScorer:
    def calculate_score(self, errors: list[ErrorAnnotation], word_count: int) -> float:
        """Calculate MQM score based on errors and word count"""
```

**Formula:**
```
MQM Score = 100 - (total_penalty / word_count * 1000)

where:
  total_penalty = Σ (error_severity × category_weight)
```

---

## Development Workflow

### Branch Strategy

We use **Git Flow**:

- `main` - production-ready code
- `develop` - integration branch
- `feature/*` - feature branches
- `bugfix/*` - bug fix branches
- `hotfix/*` - emergency fixes

### Creating a Feature

1. **Create branch from develop**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/add-comet-metrics
   ```

2. **Make changes and commit**:
   ```bash
   # Make changes
   git add .
   git commit -m "feat: add COMET metrics integration"
   ```

3. **Run tests and quality checks**:
   ```bash
   pytest
   black src/ tests/
   ruff check src/ tests/
   mypy src/kttc --strict
   ```

4. **Push and create PR**:
   ```bash
   git push origin feature/add-comet-metrics
   gh pr create --title "Add COMET metrics integration" --base develop
   ```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process or auxiliary tool changes

**Examples:**
```bash
feat(agents): add Style agent for stylistic evaluation
fix(mqm): correct scoring calculation for critical errors
docs(guides): update user guide with batch processing examples
test(orchestrator): add tests for agent coordination
refactor(llm): simplify provider initialization
perf(parser): optimize error parsing with regex
chore(deps): update pydantic to 2.5.0
```

---

## Testing

### Test Organization

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   ├── test_agents_*.py
│   ├── test_models.py
│   ├── test_mqm.py
│   └── test_llm_*.py
│
├── integration/           # Integration tests (future)
│   └── test_orchestrator.py
│
├── e2e/                   # End-to-end tests (future)
│   └── test_workflows.py
│
└── performance/           # Performance benchmarks
    └── test_benchmarks.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/kttc --cov-report=html

# Run specific test file
pytest tests/unit/test_agents_accuracy.py

# Run specific test
pytest tests/unit/test_mqm.py::test_calculate_score_no_errors

# Run with verbose output
pytest -v

# Run performance benchmarks
pytest tests/performance/ -v -s -m benchmark
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_mqm.py
import pytest
from kttc.core.mqm import MQMScorer
from kttc.core.models import ErrorAnnotation, ErrorSeverity


def test_calculate_score_no_errors():
    """Test MQM score with no errors (should be 100)."""
    scorer = MQMScorer()
    score = scorer.calculate_score(errors=[], word_count=100)
    assert score == 100.0


def test_calculate_score_single_minor_error():
    """Test MQM score with one minor error."""
    scorer = MQMScorer()
    errors = [
        ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(10, 15),
            description="Minor grammar issue",
        )
    ]
    score = scorer.calculate_score(errors, word_count=100)

    # Minor error (1 point) × fluency weight (0.8) = 0.8 penalty
    # 0.8 / 100 * 1000 = 8 points off
    assert score == pytest.approx(92.0)
```

#### Async Test Example

```python
# tests/unit/test_agents_accuracy.py
def test_accuracy_agent_evaluate(mock_llm, sample_task):
    """Test accuracy agent evaluation."""
    async def run_test():
        agent = AccuracyAgent(mock_llm)
        errors = await agent.evaluate(sample_task)
        assert len(errors) >= 0
        assert all(isinstance(e, ErrorAnnotation) for e in errors)

    asyncio.run(run_test())
```

### Test Fixtures

Use pytest fixtures for common test data:

```python
# tests/conftest.py
import pytest
from kttc.core.models import TranslationTask


@pytest.fixture
def sample_task():
    """Sample translation task for testing."""
    return TranslationTask(
        source_text="Hello, world!",
        translation="¡Hola, mundo!",
        source_lang="en",
        target_lang="es",
    )


@pytest.fixture
def mock_llm(mocker):
    """Mock LLM provider."""
    mock = mocker.Mock()
    mock.complete = mocker.AsyncMock(return_value="ERROR: none\nNo errors found.")
    return mock
```

### Coverage Goals

- **Unit tests**: ≥ 90% coverage
- **Integration tests**: ≥ 80% coverage
- **Overall**: ≥ 80% coverage

Current coverage: **99.9%** ✅

---

## Code Quality

### Code Style

We use **Black** for code formatting:

```bash
# Format all code
black src/ tests/

# Check formatting without making changes
black --check src/ tests/
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py311']
```

### Linting

We use **Ruff** (fast Python linter):

```bash
# Lint all code
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]
```

### Type Checking

We use **mypy** with strict mode:

```bash
# Type check
mypy src/kttc --strict
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
```

### Pre-commit Hooks

All checks run automatically before commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

Run manually:
```bash
pre-commit run --all-files
```

---

## Contributing

### Contribution Workflow

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run quality checks**
6. **Submit a pull request**

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

### Pull Request Checklist

Before submitting a PR:

- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`black --check`)
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy --strict`)
- [ ] Coverage ≥ 80% (`pytest --cov`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Commit messages follow convention

### Code Review Process

1. **Automated checks** run on PR creation
2. **Maintainer review** (1-2 business days)
3. **Address feedback** if needed
4. **Merge** when approved

---

## Release Process

See [RELEASE.md](../../docs/RELEASE.md) for full release process.

### Quick Release Steps

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Commit and tag**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to 0.2.0"
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin main --tags
   ```
4. **Create GitHub Release**
5. **Workflow automatically publishes to PyPI**

---

## Additional Resources

### Documentation

- [User Guide](../guides/user-guide.md) - End-user documentation
- [API Documentation](../api/README.md) - API reference
- [GitHub Actions Guide](github-actions.md) - CI/CD integration

### External Resources

- [Typer Documentation](https://typer.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pytest Documentation](https://docs.pytest.org/)
- [MQM Framework](https://themqm.org/)

### Getting Help

- **GitHub Issues**: https://github.com/kttc-ai/kttc/issues
- **GitHub Discussions**: https://github.com/kttc-ai/kttc/discussions
- **Email**: dev@kttc.ai

---

## Appendix

### Common Development Tasks

#### Adding a New Agent

1. Create agent file: `src/kttc/agents/new_agent.py`
2. Inherit from `BaseAgent`
3. Implement `evaluate()` method
4. Add prompt template in `src/kttc/llm/prompts/`
5. Register in orchestrator
6. Write tests in `tests/unit/test_agents_new.py`

#### Adding a New LLM Provider

1. Create provider file: `src/kttc/llm/new_provider.py`
2. Inherit from `BaseLLMProvider`
3. Implement `complete()` and `stream()` methods
4. Add configuration in `src/kttc/utils/config.py`
5. Write tests in `tests/unit/test_llm_new.py`

#### Adding a New Command

1. Add command function in `src/kttc/cli/main.py`
2. Use `@app.command()` decorator
3. Add command logic
4. Write tests in `tests/unit/test_cli_new.py`
5. Update user guide

---

**Last Updated:** November 10, 2025
**Version:** 0.1.0
