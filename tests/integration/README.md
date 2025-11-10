# Integration Tests

Integration tests for KTTC that perform **real requests** to LLM providers.

## ⚠️ Important

These tests:
- Make **real API requests**
- **Consume API credits/money**
- Require **real credentials**
- May take **several minutes**

## 📋 Requirements

### GigaChat Tests

To run GigaChat tests, you need credentials in `.env`:

```bash
KTTC_GIGACHAT_CLIENT_ID=your-client-id
KTTC_GIGACHAT_CLIENT_SECRET=your-client-secret
KTTC_GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

### OpenAI Tests (future)

```bash
KTTC_OPENAI_API_KEY=sk-...
```

### Anthropic Tests (future)

```bash
KTTC_ANTHROPIC_API_KEY=sk-ant-...
```

## 🚀 Running Tests

### Running ALL Integration Tests

```bash
# With Python 3.11
python3.11 -m pytest tests/integration/ -v

# With verbose output
python3.11 -m pytest tests/integration/ -v -s

# With coverage
python3.11 -m pytest tests/integration/ -v --cov=kttc
```

### Running GigaChat Tests

```bash
# All GigaChat tests
python3.11 -m pytest tests/integration/test_gigachat_integration.py -v

# Specific test group
python3.11 -m pytest tests/integration/test_gigachat_integration.py::TestGigaChatAuthentication -v

# Specific test
python3.11 -m pytest tests/integration/test_gigachat_integration.py::TestGigaChatAuthentication::test_authentication_success -v

# With print statements output
python3.11 -m pytest tests/integration/test_gigachat_integration.py -v -s
```

### Skipping Tests Without Credentials

Tests are automatically skipped if credentials are not configured:

```bash
python3.11 -m pytest tests/integration/test_gigachat_integration.py -v

# Output:
# test_authentication_success SKIPPED (GigaChat credentials not configured)
```

## 📊 What Is Being Tested

### test_gigachat_integration.py

#### TestGigaChatAuthentication
- ✅ OAuth 2.0 authentication
- ✅ Access token retrieval
- ✅ Token caching (30 minutes)
- ✅ Invalid credentials handling

#### TestGigaChatCompletion
- ✅ Basic completion requests
- ✅ Russian language support
- ✅ Parameters (temperature, max_tokens)
- ✅ Token limits

#### TestGigaChatStreaming
- ✅ Streaming mode
- ✅ Incremental chunk delivery
- ✅ Full response assembly

#### TestGigaChatErrorHandling
- ✅ Non-existent models
- ✅ Timeout handling
- ✅ Rate limits

#### TestGigaChatWithOrchestrator
- ✅ GOOD translation evaluation
- ✅ BAD translation detection
- ✅ Multiple errors
- ✅ RU→EN translation
- ✅ MQM scoring

#### TestGigaChatRealWorldScenarios
- ✅ Technical translations
- ✅ Business translations
- ✅ Batch evaluation

## 🧪 Manual Execution

You can run tests directly from Python:

```bash
python3.11 tests/integration/test_gigachat_integration.py
```

This will run a simplified version of the tests for quick verification.

## 📈 Expected Results

### Successful Run

```bash
$ python3.11 -m pytest tests/integration/test_gigachat_integration.py -v

tests/integration/test_gigachat_integration.py::TestGigaChatAuthentication::test_authentication_success PASSED
tests/integration/test_gigachat_integration.py::TestGigaChatCompletion::test_completion_simple PASSED
tests/integration/test_gigachat_integration.py::TestGigaChatStreaming::test_streaming_basic PASSED
...

====== 20 passed in 45.23s ======
```

### Skipped Tests (no credentials)

```bash
$ python3.11 -m pytest tests/integration/test_gigachat_integration.py -v

tests/integration/test_gigachat_integration.py::TestGigaChatAuthentication::test_authentication_success SKIPPED
...

====== 20 skipped in 0.05s ======
```

## ⏱️ Execution Time

Approximate time:
- **Authentication**: ~2-3 seconds
- **Completion**: ~3-5 seconds per test
- **Streaming**: ~4-6 seconds per test
- **Orchestrator**: ~10-15 seconds per test (3 agents)

**Total time**: ~3-5 minutes for all tests

## 💰 Cost

GigaChat (GIGACHAT_API_PERS) - **FREE**!

- Does not consume money
- Does not require a credit card
- Rate limits are sufficient for tests

OpenAI/Anthropic (future):
- ~$0.10-0.50 for a full test run
- Use cheap models (gpt-4o-mini)

## 🐛 Troubleshooting

### Error: Python version

```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

**Solution**: Use `python3.11` or `python3.12`

### Error: Module not found

```
ModuleNotFoundError: No module named 'kttc'
```

**Solution**:
```bash
python3.11 -m pip install -e ".[dev]"
```

### Error: Authentication failed

```
LLMAuthenticationError: GigaChat authentication failed
```

**Solution**:
1. Check the `.env` file
2. Ensure credentials are correct
3. Get new credentials at https://developers.sber.ru/studio

### Error: Timeout

```
LLMTimeoutError: GigaChat request timed out
```

**Solution**:
- This is normal for slow internet connections
- Tests use timeout=60s
- Try running again

### Tests Fail with Low MQM Scores

This may be normal! LLM models are non-deterministic:
- MQM scores can vary ±5-10 points
- Run tests 2-3 times
- If consistently failing - there may be a problem

## 📚 Adding New Tests

### Test Structure

```python
class TestMyFeature:
    """Test group description."""

    @skip_if_no_gigachat()
    async def test_my_feature(self, gigachat_provider: GigaChatProvider) -> None:
        """Test description."""
        # Arrange
        prompt = "Test prompt"

        # Act
        result = await gigachat_provider.complete(prompt)

        # Assert
        assert result is not None
        assert len(result) > 0

        print(f"\n✓ Test passed: {result[:50]}...")
```

### Fixtures

Available fixtures:
- `gigachat_provider` - GigaChatProvider with credentials
- `gigachat_orchestrator` - AgentOrchestrator with GigaChat

### Best Practices

1. **Use @skip_if_no_gigachat()** for tests requiring credentials
2. **Add print statements** for debugging (run with `-s`)
3. **Be lenient** with MQM scores (±5-10 points)
4. **Group related tests** into classes
5. **Document** what is being tested

## 🔗 See Also

- [../unit/](../unit/) - Unit tests (fast, no API)
- [../../test_kttc.py](../../test_kttc.py) - Manual test script
- [../../TESTING.md](../../TESTING.md) - General testing documentation

---

**Last updated**: November 10, 2025
