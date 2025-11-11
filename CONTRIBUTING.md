# Contributing to KTTC

Thank you for your interest in contributing to KTTC (Knowledge Translation Transmutation Core)! We welcome contributions from the community.

## Table of Contents

- [Contributor License Agreement](#contributor-license-agreement)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Contributor License Agreement

**IMPORTANT**: By submitting a contribution to this project, you agree to the following terms:

### Grant of Copyright License

You hereby grant to KTTC AI (https://github.com/kttc-ai) and to recipients of software distributed by KTTC AI a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contributions and such derivative works.

### Grant of Patent License

You hereby grant to KTTC AI and to recipients of software distributed by KTTC AI a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the work, where such license applies only to those patent claims licensable by you that are necessarily infringed by your contribution(s) alone or by combination of your contribution(s) with the work to which such contribution(s) was submitted.

### Representations

You represent that:
1. You are legally entitled to grant the above licenses
2. Each of your contributions is your original creation
3. Your contribution submissions include complete details of any third-party license or other restriction of which you are aware

### Submissions on Behalf of Your Employer

If you are making contributions as part of your employment, you represent that you have received permission from your employer to make contributions, or that your employer has waived such rights for your contributions to KTTC AI.

### Agreement

**By submitting a pull request or contribution to this project, you acknowledge that you have read this Contributor License Agreement and agree to its terms.**

---

**Note for Corporate Contributors**: If you are contributing on behalf of a company, we may require a Corporate Contributor License Agreement (CCLA). Please contact us at the project repository.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, KTTC version)
- Any relevant logs or error messages

### Suggesting Enhancements

We welcome feature requests! Please create an issue with:
- A clear description of the feature
- Use cases and benefits
- Any relevant examples or mockups

### Contributing Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following our coding standards
4. Write or update tests as needed
5. Ensure all quality checks pass
6. Commit your changes (see commit message guidelines below)
7. Push to your fork
8. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Git

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/kttc.git
cd kttc

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
python3.11 -m pip install -e ".[dev]"

# Install pre-commit hooks (automatic)
# The hook is already configured in .git/hooks/pre-commit
```

### Running Tests

```bash
# Run all tests
python3.11 -m pytest

# Run specific test suite
python3.11 -m pytest tests/unit/
python3.11 -m pytest tests/integration/

# Run with coverage
python3.11 -m pytest --cov=kttc --cov-report=html
```

## Code Quality Standards

All code contributions must meet strict quality standards enforced by automated checks.

### Pre-commit Hook

A pre-commit hook is installed that automatically runs before each commit to ensure code quality.

#### What It Checks:

1. **Black** - Code formatting
   - Ensures consistent code style
   - Runs in check mode (won't modify files)

2. **Ruff** - Python linter
   - Catches common bugs and code smells
   - Enforces best practices

3. **MyPy** - Static type checker (src/ files only)
   - Runs in `--strict` mode
   - Ensures type safety

#### Hook is Already Installed

The pre-commit hook is located at `.git/hooks/pre-commit` and runs automatically.

#### Example Output

**When code has issues:**
```
🔍 Running pre-commit checks...

📝 Files to check:
  - src/kttc/example.py

▶️  Running Black...
✗ Black: Code needs formatting
  Run: python3.11 -m black src/kttc/example.py

❌ Pre-commit checks FAILED
```

**When code is clean:**
```
🔍 Running pre-commit checks...

📝 Files to check:
  - src/kttc/example.py

▶️  Running Black...
✓ Black: Code is formatted

▶️  Running Ruff...
✓ Ruff: No linting issues

▶️  Running MyPy (strict mode)...
✓ MyPy: No type errors

✅ All pre-commit checks PASSED
```

### Fixing Issues

If the pre-commit hook fails, fix the issues before committing:

```bash
# Fix formatting
python3.11 -m black src/ tests/ examples/

# Fix linting issues  
python3.11 -m ruff check --fix src/ tests/ examples/

# Check types
python3.11 -m mypy --strict src/
```

### Skipping Checks (Not Recommended)

In rare cases, you can skip the hook:

```bash
git commit --no-verify
```

**WARNING:** This should only be used in exceptional circumstances.

## Manual Quality Checks

You can run checks manually before committing:

```bash
# Run all checks
python3.11 -m black --check src/ tests/ examples/
python3.11 -m ruff check src/ tests/ examples/
python3.11 -m mypy --strict src/

# Auto-fix issues
python3.11 -m black src/ tests/ examples/
python3.11 -m ruff check --fix src/ tests/ examples/
```

## Code Style Requirements

### English Only

- ALL code, comments, documentation, and commit messages MUST be in English
- No Cyrillic (Russian) text allowed in source code
- See `claude.md` for details

### Type Hints

- All functions must have type hints (strict mode)
- Use modern Python 3.11+ syntax: `str | None` instead of `Optional[str]`

### Formatting

- Follow Black's code style (line length: 100)
- Imports sorted by Ruff
- Docstrings in Google style

### Testing

- Write tests for new features
- Maintain >80% code coverage
- All tests must pass before commit

## Python Version

This project requires **Python 3.11+**

Always use:
```bash
python3.11 -m <command>
```

NOT:
```bash
python3 -m <command>  # Wrong - this is Python 3.9
```

See `claude.md` for details.

## Pull Request Process

1. **Update Documentation**: Update README.md or other docs if needed
2. **Add Tests**: Ensure new features have appropriate test coverage (>80%)
3. **Follow Coding Standards**: Use Black for formatting, Ruff for linting, MyPy for type checking
4. **Write Clear Commit Messages**: Use conventional commit format (see below)
5. **Keep PRs Focused**: One feature or fix per PR
6. **Pass All Quality Checks**: Pre-commit hook must pass

### Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(agents): add context-aware agent for cultural nuances

Implement new agent that analyzes cultural context in translations
and suggests culturally appropriate alternatives.

Closes #123
```

## Coding Standards

### Documentation

- Write clear, self-documenting code
- Add docstrings to all public modules, functions, classes, and methods (Google style)
- Use meaningful variable and function names
- Keep functions focused and small
- Add inline comments for complex logic

### Best Practices

- Write unit tests for new functionality
- Test edge cases and error conditions
- Use descriptive test names
- Avoid deep nesting (max 3-4 levels)
- Follow the Single Responsibility Principle

## Questions?

- Check `README.md` for project overview
- Check `CLAUDE.md` for development guidelines
- Open an issue on GitHub

---

Thank you for contributing to KTTC! Your contributions help make translation QA better for everyone.
