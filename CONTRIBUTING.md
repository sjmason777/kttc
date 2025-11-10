# Contributing to KTTC

Thank you for your interest in contributing to KTTC!

## Development Setup

### Prerequisites
- Python 3.11+
- Git
- OpenAI or Anthropic API key

### Setup Environment

```bash
# Clone repository
git clone git@github.com:kttc-ai/kttc.git
cd kttc

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kttc

# Run specific tests
pytest tests/unit/test_agents.py -v

# Run in parallel
pytest -n auto
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Development Workflow

1. **Create branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write tests first (TDD)
   - Follow code style (Black, Ruff)
   - Add type hints
   - Update documentation

3. **Run checks**
   ```bash
   pre-commit run --all-files
   pytest
   ```

4. **Commit**
   ```bash
   git commit -m "feat: add new feature"
   ```

   Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation
   - `test:` - Tests
   - `chore:` - Maintenance

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8
- Use type hints everywhere
- Write docstrings (Google style)
- Keep functions small and focused
- Test coverage > 80%

## Project Structure

```
kttc/
├── src/kttc/           # Source code
│   ├── cli/            # CLI commands
│   ├── agents/         # QA agents
│   ├── core/           # Core logic
│   └── llm/            # LLM providers
├── tests/              # Tests
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/               # Documentation
```

## Questions?

Open an issue or contact: dev@kttc.ai

---

Thank you for contributing! 🎉
