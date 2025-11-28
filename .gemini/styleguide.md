# KTTC Code Review Style Guide

## Project Overview

KTTC (Knowledge Translation Transmutation Core) is a Python library for translation quality assurance using LLM agents. The codebase uses async patterns extensively and follows modern Python practices.

## Language Requirements

- **All code, comments, docstrings, and documentation MUST be in English only**
- No Russian (Cyrillic) or other non-English text in source code
- Exception: Test data files may contain multilingual content for testing purposes

## Python Standards

### Version and Type Hints
- Python 3.11+ required
- Use modern union syntax: `str | None` instead of `Optional[str]`
- Use `list[str]` instead of `List[str]`
- All public functions must have type hints

### Code Style
- Follow PEP-8 guidelines
- Use Black formatter with default settings
- Line length: 88 characters (Black default)
- Use Ruff for linting

### Async Patterns
- Use `async/await` for I/O-bound operations
- Prefer `aiohttp` for HTTP requests
- Use `asyncio.gather()` for concurrent operations

## Security Requirements

- **Never commit API keys, secrets, or credentials**
- Use environment variables with `KTTC_` prefix for configuration
- No hardcoded URLs to external services (use config)
- Validate all external input
- Use `defusedxml` for XML parsing (not `xml.etree`)

## Testing Requirements

- Unit tests required for new features
- Tests go in `tests/unit/` or `tests/integration/`
- Use `pytest` with async support
- Mock external API calls in unit tests

## Documentation

- Public APIs must have docstrings
- Use Google-style docstrings format
- Update README.md for user-facing changes

## Dependencies

- Add new dependencies to `pyproject.toml`
- Prefer lightweight dependencies
- Document optional dependencies in `[project.optional-dependencies]`

## What to Flag in Reviews

1. Missing type hints on public functions
2. Hardcoded credentials or API keys
3. Synchronous I/O in async contexts
4. Missing error handling for external calls
5. Non-English comments or docstrings
6. Security vulnerabilities (SQL injection, command injection, etc.)
7. Breaking changes without version bump
