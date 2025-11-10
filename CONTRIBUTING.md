# Contributing to KTTC

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

## Questions?

- Check `README.md` for project overview
- Check `claude.md` for development guidelines
- Open an issue on GitHub
