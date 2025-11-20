# Integration Tests for Language Helpers

## ⚠️ IMPORTANT: CI/CD Configuration

**Hindi and Persian tests are marked as `@pytest.mark.integration` and `@pytest.mark.slow`**

These tests:
- Load heavy ML models (100-200MB each)
- Take 3-30 minutes to run
- Are **automatically skipped in CI/CD** (when `CI=true` or `GITHUB_ACTIONS=true`)
- Should be run manually or in separate integration test pipelines

## Why These Are Integration Tests (Not Unit Tests)

### Hindi Tests (`test_hindi_language_helper.py`)
- **Runtime:** 3-5 minutes (29 tests)
- **Dependencies:**
  - Stanza: 321MB Hindi models (POS, NER, lemmatization)
  - Indic NLP Library: tokenization
  - Spello: spell checking
- **Why slow:** Model loading + initialization

### Persian Tests (`test_persian_language_helper.py`)
- **Runtime:** 20-30 minutes (34 tests)
- **Dependencies:**
  - DadmaTools: ~10 heavy ML models
    - Tokenizer, POS tagger (98.8% accuracy)
    - Lemmatizer (89.9% accuracy)
    - Dependency parser (85.6% accuracy)
    - NER, sentiment analyzer
    - Spell checker, informal-to-formal converter
- **Why slow:** 10+ model downloads/loading + caching

## Running Integration Tests

### Run all integration tests:
```bash
pytest -m integration
```

### Run specific language tests:
```bash
# Hindi tests only (3-5 min)
pytest tests/unit/test_hindi_language_helper.py -m integration -v

# Persian tests only (20-30 min)
pytest tests/unit/test_persian_language_helper.py -m integration -v
```

### Skip slow tests in local development:
```bash
# Run all tests EXCEPT integration/slow tests
pytest -m "not integration and not slow"

# Standard unit tests only (fast!)
pytest tests/unit/ -m "not integration"
```

## CI/CD Configuration

Tests are **automatically skipped** in CI/CD via `pytestmark`:

```python
# In test files:
pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
    reason="Tests are slow - skip in CI/CD. Run manually with: pytest -m integration"
)
```

### To run in CI/CD (optional):
```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:  # Manual trigger

jobs:
  integration:
    runs-on: ubuntu-latest
    timeout-minutes: 45  # Allow time for model downloads
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev,hindi,persian]"
      - name: Run integration tests
        run: |
          # Override CI env to allow tests to run
          unset CI
          unset GITHUB_ACTIONS
          pytest -m integration -v --timeout=1800
```

## Test Markers

- `@pytest.mark.integration` - Tests requiring external services/heavy dependencies
- `@pytest.mark.slow` - Tests taking >10 seconds
- `@pytest.mark.unit` - Fast unit tests (< 1 second)

## Performance Optimization

### First run (slow - models download):
- Hindi: 3-5 min
- Persian: 20-30 min

### Subsequent runs (faster - cached models):
- Hindi: 2-3 min
- Persian: 15-20 min

### Cache locations:
- Stanza (Hindi): `~/stanza_resources/`
- DadmaTools (Persian): `~/.cache/dadmatools/`

## Test Coverage

### Hindi (29 tests):
- Basic instantiation and interface (2)
- Tokenization with positions (4)
- Morphological analysis (Stanza) (3)
- Word verification (anti-hallucination) (3)
- Error position validation (3)
- Grammar checking interface (2)
- Spell checking (Spello) (2)
- LLM enrichment data (2)
- Named entity extraction (Stanza NER) (2)
- Edge cases (Unicode, mixed script) (4)
- Integration workflows (2)

### Persian (34 tests):
- All Hindi test categories PLUS:
- Sentiment analysis (DadmaTools v2) (2)
- Informal-to-formal conversion (DadmaTools v2) (2)
- Additional integration tests (3)

## Found Bugs (Examples)

Tests successfully found real bugs:

1. **Empty string tokenization** - returned `[('', 0, 0)]` instead of `[]`
2. **Stanza network resilience** - failed without `download_method=REUSE_RESOURCES`
3. **Morphology feats parsing** - tried to call `.get()` on string

These are the types of bugs integration tests are designed to catch!

## Adding New Integration Tests

When adding tests for new language helpers:

1. **Mark tests appropriately:**
   ```python
   import os
   import pytest

   # Skip in CI/CD
   pytestmark = pytest.mark.skipif(
       os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
       reason="Language tests are slow - skip in CI/CD"
   )

   @pytest.mark.integration
   @pytest.mark.slow
   class TestNewLanguageHelper:
       ...
   ```

2. **Document expected runtime** in docstring
3. **Test real functionality** - not just mocks
4. **Verify model loading** works correctly

## Questions?

See main project documentation or contact the team.
