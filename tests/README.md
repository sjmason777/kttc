# KTTC Test Suite

Modern, fast, and rigorous test suite following 2025 best practices.

## Test Structure (Test Pyramid)

```
tests/
├── unit/                    # 58% (37 tests) - Fast (~1.5 sec total)
│   ├── test_cli.py          # CLI argument parsing, output formats (13 tests)
│   ├── test_agents.py       # Agent logic with mocked LLM (12 tests)
│   └── test_orchestrator.py # Orchestration and MQM scoring (12 tests)
├── integration/             # 31% (20 tests) - Medium speed (~1.9 sec)
│   └── test_cli_flow.py     # Full CLI workflows with real components (20 tests)
├── e2e/                     # 11% (7 tests) - Slow (run with --run-e2e)
│   └── test_real_api.py     # Real Anthropic API calls (7 tests)
├── conftest.py              # Common fixtures and mocks
└── pytest.ini               # Test markers and configuration
```

## Quick Start

```bash
# Run all fast tests (unit + integration, ~2 seconds)
pytest tests/ -m "not e2e" -v

# Run unit tests only (fastest, ~1.5 seconds)
pytest tests/unit/ -v

# Run integration tests only (~1.9 seconds)
pytest tests/integration/ -v

# Run E2E tests (slow, requires KTTC_ANTHROPIC_API_KEY)
export KTTC_ANTHROPIC_API_KEY="your-key-here"
pytest tests/e2e/ --run-e2e -v

# Run specific test file
pytest tests/unit/test_cli.py -v

# Run tests by marker
pytest -m unit         # Unit tests only
pytest -m integration  # Integration tests only
pytest -m e2e --run-e2e  # E2E tests only

# Run with coverage (fast tests only)
pytest tests/ -m "not e2e" --cov=kttc --cov-report=html

# Run all tests including E2E (requires API key)
pytest tests/ --run-e2e -v
```

## Test Markers

- `@pytest.mark.unit` - Fast, isolated tests with mocks
- `@pytest.mark.integration` - Multiple components, medium speed
- `@pytest.mark.e2e` - End-to-end tests with real API calls
- `@pytest.mark.slow` - Any test taking >5 seconds

## Current Status

**Total Tests:** ✅ 64 tests (57 fast, 7 E2E)

| Test Type    | Count | Percentage | Speed   | Status |
|--------------|-------|------------|---------|--------|
| Unit         | 37    | 58%        | ~1.5s   | ✅ PASS |
| Integration  | 20    | 31%        | ~1.9s   | ✅ PASS |
| E2E          | 7     | 11%        | Slow*   | ✅ PASS |

**\*E2E tests require `--run-e2e` flag and real API key**

### Unit Test Breakdown
| Component    | Tests | Status |
|--------------|-------|--------|
| CLI          | 13    | ✅ PASS |
| Agents       | 12    | ✅ PASS |
| Orchestrator | 12    | ✅ PASS |

### Integration Test Breakdown
| Component              | Tests | Status |
|------------------------|-------|--------|
| CLI Integration Flow   | 4     | ✅ PASS |
| Agent Pipeline         | 2     | ✅ PASS |
| Batch Processing       | 1     | ✅ PASS |
| Error Handling         | 3     | ✅ PASS |
| Language Pairs         | 4     | ✅ PASS |
| Output Formats         | 2     | ✅ PASS |
| Text Processing        | 3     | ✅ PASS |
| Performance            | 1     | ✅ PASS |

### E2E Test Breakdown
| Component                | Tests | Status |
|--------------------------|-------|--------|
| Real Anthropic API       | 3     | ✅ PASS |
| Real API Language Pairs  | 2     | ✅ PASS |
| Complex Scenarios        | 2     | ✅ PASS |

## Testing Philosophy

### Rigorous Testing
Tests are **strict** and **honest** - they find real bugs:
- Found incorrect LLM response formats (JSON vs ERROR_START/END)
- Validated Pydantic model constraints
- Test actual behavior, not assumptions

### AAA Pattern
All tests follow the Arrange-Act-Assert pattern:
```python
# Arrange - Setup
agent = AccuracyAgent(mock_llm)

# Act - Execute
errors = await agent.evaluate(task)

# Assert - Verify
assert len(errors) == 1
assert errors[0].category == "accuracy"
```

### Mocking Strategy
- **Unit tests:** Mock all external dependencies (LLM, file I/O)
- **Integration tests:** Mock only external APIs
- **E2E tests:** No mocks, real API calls

## Fixtures

### Available Fixtures (from conftest.py)
- `mock_llm` - Returns no errors
- `mock_llm_with_errors` - Returns accuracy errors
- `sample_translation_task` - Hello/Hola example
- `sample_translation_error` - ErrorAnnotation example
- `sample_qa_report` - QA report example
- `temp_text_files` - Temporary source/translation files
- `cli_runner` - Typer CLI test runner

## Adding New Tests

### 1. Unit Test Example
```python
@pytest.mark.unit
class TestMyFeature:
    @pytest.mark.asyncio
    async def test_feature_works(self, mock_llm: Any) -> None:
        # Arrange
        feature = MyFeature(mock_llm)

        # Act
        result = await feature.process()

        # Assert
        assert result is not None
```

### 2. Integration Test Example
```python
@pytest.mark.integration
class TestFullFlow:
    @pytest.mark.asyncio
    async def test_cli_to_report(self, temp_text_files: tuple[Path, Path]) -> None:
        # Test full flow from CLI to report generation
        ...
```

### 3. E2E Test Example
```python
@pytest.mark.e2e
class TestRealAPI:
    @pytest.mark.asyncio
    async def test_anthropic_translation_check(self) -> None:
        # Requires real API key - skipped unless --run-e2e
        ...
```

## CI/CD Integration

Tests run automatically in GitHub Actions:
- **Pre-commit:** Unit tests only (<2 seconds)
- **PR validation:** Unit + Integration (<5 minutes)
- **Nightly build:** All tests including E2E

## Core Principles (2025 Best Practices)

1. ✅ **Fast Feedback** - Unit tests <2 seconds
2. ✅ **Deterministic** - No flaky tests, no external dependencies
3. ✅ **Single Assert Focus** - Each test validates one thing
4. ✅ **Meaningful Names** - Test names describe behavior
5. ✅ **AAA Pattern** - Arrange, Act, Assert
6. ✅ **Mock External APIs** - Test units in isolation
7. ✅ **Strict Validation** - Tests find real errors

## Troubleshooting

### Slow Tests
```bash
# Check which tests are slow
pytest --durations=10

# Run only fast tests
pytest -m "not slow"
```

### Import Errors
```bash
# Install in editable mode
python3.11 -m pip install -e ".[dev]"
```

### Async Warnings
```bash
# Already configured in pytest.ini
# asyncio_mode = auto
```

---

**Last Updated:** 2025-11-15
**Test Count:** 64 tests (37 unit, 20 integration, 7 E2E)
**Total Time:** ~2 seconds (unit + integration), E2E tests slower
**Test Pyramid:** 58% unit / 31% integration / 11% E2E (target: 50/40/10)

## Summary

Successfully rebuilt KTTC test suite from scratch using 2025 best practices:
- **Before:** 941 tests taking 30+ minutes with heavy ML models
- **After:** 64 focused tests in ~2 seconds (99.9% faster!)
- **Quality:** Tests find real bugs (JSON vs ERROR_START/END format issue)
- **Coverage:** Complete test pyramid with unit, integration, and E2E tests
- **Maintainability:** Clear structure, AAA pattern, comprehensive fixtures
