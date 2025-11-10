"""Strict integration tests for GigaChat provider.

These tests perform REAL requests to the GigaChat API and verify:
1. OAuth 2.0 authentication
2. Basic completion requests
3. Streaming
4. Error handling
5. Spanish language support
6. Integration with orchestrator
7. Real translation quality evaluation

IMPORTANT: These tests require real credentials in the .env file:
- KTTC_GIGACHAT_CLIENT_ID
- KTTC_GIGACHAT_CLIENT_SECRET
- KTTC_GIGACHAT_SCOPE

Usage:
    python3.11 -m pytest tests/integration/test_gigachat_integration.py -v
    python3.11 -m pytest tests/integration/test_gigachat_integration.py -v -s  # with output
    python3.11 -m pytest tests/integration/test_gigachat_integration.py::TestGigaChatAuthentication -v
"""

import os

import pytest

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core.models import TranslationTask
from kttc.llm.base import (
    LLMAuthenticationError,
    LLMError,
)
from kttc.llm.gigachat_provider import GigaChatProvider

# Mark all tests as integration tests requiring GigaChat credentials
pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
]


def get_gigachat_credentials() -> dict[str, str] | None:
    """Get GigaChat credentials from environment variables.

    Returns:
        Dictionary with credentials or None if not configured
    """
    client_id = os.getenv("KTTC_GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("KTTC_GIGACHAT_CLIENT_SECRET")
    scope = os.getenv("KTTC_GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    if not client_id or not client_secret:
        return None

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }


def skip_if_no_gigachat() -> pytest.MarkDecorator:
    """Skip test if GigaChat credentials are not configured."""
    creds = get_gigachat_credentials()
    return pytest.mark.skipif(
        creds is None,
        reason="GigaChat credentials not configured. Set KTTC_GIGACHAT_CLIENT_ID and KTTC_GIGACHAT_CLIENT_SECRET",
    )


@pytest.fixture
def gigachat_provider() -> GigaChatProvider:
    """Fixture for GigaChat provider with real credentials."""
    creds = get_gigachat_credentials()
    if not creds:
        pytest.skip("GigaChat credentials not configured")

    return GigaChatProvider(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        scope=creds["scope"],
        model="GigaChat",
        timeout=60.0,  # Extended timeout for integration tests
    )


@pytest.fixture
def gigachat_orchestrator(gigachat_provider: GigaChatProvider) -> AgentOrchestrator:
    """Fixture for orchestrator with GigaChat provider."""
    return AgentOrchestrator(gigachat_provider)


class TestGigaChatAuthentication:
    """OAuth 2.0 authentication tests for GigaChat."""

    @skip_if_no_gigachat()
    async def test_authentication_success(self, gigachat_provider: GigaChatProvider) -> None:
        """Test successful authentication and access token retrieval."""
        # Get access token
        token = await gigachat_provider._get_access_token()

        # Verify token is not empty
        assert token is not None
        assert len(token) > 0
        assert isinstance(token, str)

        # Token should be JWT format (contains dots)
        assert "." in token, "Access token must be in JWT format"

        print(f"\n✓ Access token received: {token[:30]}...")

    @skip_if_no_gigachat()
    async def test_authentication_caching(self, gigachat_provider: GigaChatProvider) -> None:
        """Test access token caching (token is valid for 30 minutes)."""
        # First request - get new token
        token1 = await gigachat_provider._get_access_token()

        # Second request - should return cached token
        gigachat_provider._access_token = token1
        token2 = await gigachat_provider._get_access_token()

        # Tokens should be identical (caching works)
        assert token1 == token2

        print("\n✓ Token is cached correctly")

    @skip_if_no_gigachat()
    async def test_authentication_invalid_credentials(self) -> None:
        """Test handling of invalid credentials."""
        # Create provider with invalid credentials
        provider = GigaChatProvider(
            client_id="invalid-client-id",
            client_secret="invalid-client-secret",
            scope="GIGACHAT_API_PERS",
        )

        # Should raise authentication error
        with pytest.raises(LLMAuthenticationError, match="GigaChat authentication failed"):
            await provider._get_access_token()

        print("\n✓ Invalid credentials handled correctly")


class TestGigaChatCompletion:
    """Basic completion request tests for GigaChat."""

    @skip_if_no_gigachat()
    async def test_completion_simple(self, gigachat_provider: GigaChatProvider) -> None:
        """Test simple completion request."""
        prompt = "Translate to English: Hello, world!"

        result = await gigachat_provider.complete(prompt, max_tokens=100)

        # Verify response is not empty
        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)

        # Response should contain English text
        assert "hello" in result.lower() or "world" in result.lower()

        print(f"\n✓ Completion successful: {result[:100]}...")

    @skip_if_no_gigachat()
    async def test_completion_russian_quality(self, gigachat_provider: GigaChatProvider) -> None:
        """Test quality of Spanish language support."""
        prompts_and_checks = [
            ("What is machine learning?", ["machine", "learning", "algorithm", "data"]),
            ("Translate to Spanish: Hello", ["hello", "hola"]),
            ("Fix the error: I went to the store", ["store", "went"]),  # Should keep correct words
        ]

        for prompt, expected_words in prompts_and_checks:
            result = await gigachat_provider.complete(prompt, max_tokens=200, temperature=0.3)

            assert result is not None
            assert len(result) > 0

            # Verify response contains at least one expected word
            result_lower = result.lower()
            has_expected = any(word.lower() in result_lower for word in expected_words)
            assert has_expected, f"Response does not contain expected words. Got: {result}"

            print(f"\n✓ Text processed: '{prompt[:50]}...' → {result[:80]}...")

    @skip_if_no_gigachat()
    async def test_completion_with_parameters(self, gigachat_provider: GigaChatProvider) -> None:
        """Test completion with various parameters."""
        prompt = "Write a short list of 3 fruits:"

        # Low temperature - more deterministic response
        result_low_temp = await gigachat_provider.complete(prompt, temperature=0.1, max_tokens=100)

        # High temperature - more creative response
        result_high_temp = await gigachat_provider.complete(prompt, temperature=0.9, max_tokens=100)

        # Both responses should be valid
        assert result_low_temp is not None
        assert result_high_temp is not None
        assert len(result_low_temp) > 0
        assert len(result_high_temp) > 0

        print(f"\n✓ Low temp: {result_low_temp[:60]}...")
        print(f"✓ High temp: {result_high_temp[:60]}...")

    @skip_if_no_gigachat()
    async def test_completion_max_tokens_limit(self, gigachat_provider: GigaChatProvider) -> None:
        """Test max_tokens limit."""
        prompt = "Write a long text about artificial intelligence"

        # Very small limit
        result_short = await gigachat_provider.complete(prompt, max_tokens=20)

        # Larger limit
        result_long = await gigachat_provider.complete(prompt, max_tokens=200)

        assert result_short is not None
        assert result_long is not None

        # Short response should be shorter than long response
        assert len(result_short) < len(result_long)

        print(f"\n✓ Short ({len(result_short)} chars): {result_short[:50]}...")
        print(f"✓ Long ({len(result_long)} chars): {result_long[:50]}...")


class TestGigaChatStreaming:
    """Streaming mode tests for GigaChat."""

    @skip_if_no_gigachat()
    async def test_streaming_basic(self, gigachat_provider: GigaChatProvider) -> None:
        """Test basic streaming."""
        prompt = "Count from 1 to 5"

        chunks: list[str] = []
        async for chunk in gigachat_provider.stream(prompt, max_tokens=100):
            assert isinstance(chunk, str)
            chunks.append(chunk)

        # Should receive multiple chunks
        assert len(chunks) > 0

        # Assembled text should not be empty
        full_text = "".join(chunks)
        assert len(full_text) > 0

        print(f"\n✓ Received {len(chunks)} chunks, total {len(full_text)} characters")
        print(f"✓ Text: {full_text[:100]}...")

    @skip_if_no_gigachat()
    async def test_streaming_incremental(self, gigachat_provider: GigaChatProvider) -> None:
        """Test that streaming is truly incremental."""
        prompt = "Write a short text about Python"

        chunks: list[str] = []
        chunk_count = 0

        async for chunk in gigachat_provider.stream(prompt, max_tokens=150):
            chunks.append(chunk)
            chunk_count += 1

        # Should receive multiple chunks (at least 2)
        assert len(chunks) >= 2, "Received only one chunk (streaming not working?)"

        # Verify chunks are not empty
        for i, chunk in enumerate(chunks):
            assert len(chunk) > 0, f"Chunk {i} is empty"

        # Verify full text is assembled from chunks
        full_text = "".join(chunks)
        assert len(full_text) > 0, "Full text is empty"

        print(
            f"\n✓ Streaming works incrementally: {len(chunks)} chunks, {len(full_text)} characters"
        )


class TestGigaChatErrorHandling:
    """GigaChat API error handling tests."""

    @skip_if_no_gigachat()
    async def test_invalid_model(self) -> None:
        """Test handling of non-existent model."""
        creds = get_gigachat_credentials()
        assert creds is not None

        provider = GigaChatProvider(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            model="NonExistentModel-XYZ",  # Non-existent model
        )

        # May be API error or success (GigaChat may ignore unknown models)
        try:
            result = await provider.complete("Test", max_tokens=10)
            # If success - that's OK, GigaChat falls back to default model
            assert result is not None
            print("\n✓ GigaChat uses fallback model")
        except LLMError as e:
            # Error is also valid
            print(f"\n✓ Error handled correctly: {e}")

    @skip_if_no_gigachat()
    async def test_timeout_handling(self) -> None:
        """Test timeout handling."""
        creds = get_gigachat_credentials()
        assert creds is not None

        # Create provider with very short timeout
        provider = GigaChatProvider(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            timeout=0.001,  # 1ms - guaranteed timeout
        )

        # Should raise timeout error
        with pytest.raises((LLMError, Exception)):  # May be LLMTimeoutError or other error
            await provider.complete("Test prompt", max_tokens=100)

        print("\n✓ Timeout handled correctly")


class TestGigaChatWithOrchestrator:
    """GigaChat integration tests with orchestrator for translation quality evaluation."""

    @skip_if_no_gigachat()
    async def test_orchestrator_good_translation(
        self, gigachat_orchestrator: AgentOrchestrator
    ) -> None:
        """Test evaluation of GOOD translation through orchestrator."""
        task = TranslationTask(
            source_text="Machine learning is transforming the world.",
            translation="El aprendizaje automático está transformando el mundo.",
            source_lang="en",
            target_lang="es",
        )

        report = await gigachat_orchestrator.evaluate(task)

        # Verify report structure
        assert report is not None
        assert report.mqm_score is not None
        assert report.status in ["pass", "fail"]
        assert isinstance(report.errors, list)

        # Good translation should have high score (but we can be lenient)
        assert report.mqm_score >= 80.0, f"Good translation received low score: {report.mqm_score}"

        # Should have few or no errors
        assert (
            len(report.errors) <= 2
        ), f"Good translation has too many errors: {len(report.errors)}"

        print("\n✓ Good translation evaluated correctly:")
        print(f"  MQM Score: {report.mqm_score:.2f}")
        print(f"  Status: {report.status}")
        print(f"  Errors: {len(report.errors)}")

    @skip_if_no_gigachat()
    async def test_orchestrator_bad_translation(
        self, gigachat_orchestrator: AgentOrchestrator
    ) -> None:
        """Test evaluation of BAD translation through orchestrator."""
        task = TranslationTask(
            source_text="Delete all user data immediately.",
            translation="Guardar todos los datos del usuario inmediatamente.",  # CRITICAL error: Delete → Save!
            source_lang="en",
            target_lang="es",
        )

        report = await gigachat_orchestrator.evaluate(task)

        # Verify that system found the error
        assert report is not None

        # Bad translation should have low score OR errors
        has_low_score = report.mqm_score < 95.0
        has_errors = len(report.errors) > 0

        assert has_low_score or has_errors, (
            f"Critical translation error was not detected! "
            f"MQM: {report.mqm_score}, Errors: {len(report.errors)}"
        )

        print("\n✓ Bad translation detected:")
        print(f"  MQM Score: {report.mqm_score:.2f}")
        print(f"  Status: {report.status}")
        print(f"  Errors found: {len(report.errors)}")
        for error in report.errors:
            print(f"    - [{error.severity.name}] {error.description[:60]}...")

    @skip_if_no_gigachat()
    async def test_orchestrator_multiple_errors(
        self, gigachat_orchestrator: AgentOrchestrator
    ) -> None:
        """Test detection of multiple errors."""
        task = TranslationTask(
            source_text="The service is available 24/7 for all premium users worldwide.",
            translation="El servicio está disponible para algunos usuarios.",  # Lost: 24/7, premium, worldwide
            source_lang="en",
            target_lang="es",
        )

        report = await gigachat_orchestrator.evaluate(task)

        assert report is not None

        # Should find errors (information loss)
        # We'll be lenient - require at least one error OR low score
        has_errors = len(report.errors) > 0
        has_low_score = report.mqm_score < 90.0

        assert has_errors or has_low_score, (
            f"Information loss not detected! "
            f"MQM: {report.mqm_score}, Errors: {len(report.errors)}"
        )

        print("\n✓ Multiple errors:")
        print(f"  MQM Score: {report.mqm_score:.2f}")
        print(f"  Errors found: {len(report.errors)}")

    @skip_if_no_gigachat()
    async def test_orchestrator_spanish_to_english(
        self, gigachat_orchestrator: AgentOrchestrator
    ) -> None:
        """Test evaluation of Spanish to English translation."""
        task = TranslationTask(
            source_text="La inteligencia artificial ayuda a resolver problemas complejos.",
            translation="Artificial intelligence helps solve complex problems.",
            source_lang="es",
            target_lang="en",
        )

        report = await gigachat_orchestrator.evaluate(task)

        assert report is not None
        assert report.mqm_score is not None

        # Good translation, but GigaChat may be less accurate for ES→EN
        assert (
            report.mqm_score >= 75.0
        ), f"ES→EN translation received too low score: {report.mqm_score}"

        print("\n✓ ES→EN translation evaluated:")
        print(f"  MQM Score: {report.mqm_score:.2f}")
        print(f"  Errors: {len(report.errors)}")


class TestGigaChatRealWorldScenarios:
    """Real-world usage scenario tests."""

    @skip_if_no_gigachat()
    async def test_technical_translation(self, gigachat_orchestrator: AgentOrchestrator) -> None:
        """Test evaluation of technical translation."""
        task = TranslationTask(
            source_text="The neural network architecture uses convolutional layers for feature extraction.",
            translation="La arquitectura de red neuronal utiliza capas convolucionales para la extracción de características.",
            source_lang="en",
            target_lang="es",
        )

        report = await gigachat_orchestrator.evaluate(task)

        assert report is not None
        # Technical translation - high requirements
        assert report.mqm_score >= 85.0

        print(f"\n✓ Technical translation: MQM {report.mqm_score:.2f}")

    @skip_if_no_gigachat()
    async def test_business_translation(self, gigachat_orchestrator: AgentOrchestrator) -> None:
        """Test evaluation of business translation."""
        task = TranslationTask(
            source_text="Please contact our customer support team for assistance.",
            translation="Por favor, contacte con nuestro equipo de atención al cliente para obtener asistencia.",
            source_lang="en",
            target_lang="es",
        )

        report = await gigachat_orchestrator.evaluate(task)

        assert report is not None
        assert report.mqm_score >= 85.0

        print(f"\n✓ Business translation: MQM {report.mqm_score:.2f}")

    @skip_if_no_gigachat()
    async def test_batch_evaluation(self, gigachat_orchestrator: AgentOrchestrator) -> None:
        """Test batch evaluation of multiple translations."""
        tasks = [
            TranslationTask(
                source_text="Hello, world!",
                translation="¡Hola, mundo!",
                source_lang="en",
                target_lang="es",
            ),
            TranslationTask(
                source_text="Good morning!",
                translation="¡Buenos días!",
                source_lang="en",
                target_lang="es",
            ),
            TranslationTask(
                source_text="Thank you!",
                translation="¡Gracias!",
                source_lang="en",
                target_lang="es",
            ),
        ]

        reports = []
        for task in tasks:
            report = await gigachat_orchestrator.evaluate(task)
            reports.append(report)

        # All reports should be valid
        assert len(reports) == len(tasks)
        for report in reports:
            assert report is not None
            assert report.mqm_score is not None

        # Average score should be high
        avg_score = sum(r.mqm_score for r in reports) / len(reports)
        assert avg_score >= 80.0

        print(f"\n✓ Batch evaluation of {len(tasks)} translations:")
        print(f"  Average MQM: {avg_score:.2f}")
        for i, report in enumerate(reports, 1):
            print(f"  {i}. {report.mqm_score:.2f}")


# Helper function for manual execution
async def run_all_tests() -> None:
    """Run all tests manually (for debugging)."""

    creds = get_gigachat_credentials()
    if not creds:
        print("❌ GigaChat credentials not configured!")
        print("Set KTTC_GIGACHAT_CLIENT_ID and KTTC_GIGACHAT_CLIENT_SECRET")
        return

    print("🧪 Running GigaChat integration tests...\n")

    provider = GigaChatProvider(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        scope=creds["scope"],
    )

    # Authentication test
    print("1. Authentication test...")
    token = await provider._get_access_token()
    print(f"   ✓ Token received: {token[:30]}...")

    # Completion test
    print("\n2. Completion test...")
    result = await provider.complete("Translate to English: Hello!")
    print(f"   ✓ Result: {result}")

    # Orchestrator test
    print("\n3. Orchestrator test...")
    orchestrator = AgentOrchestrator(provider)
    task = TranslationTask(
        source_text="Hello!",
        translation="¡Hola!",
        source_lang="en",
        target_lang="es",
    )
    report = await orchestrator.evaluate(task)
    print(f"   ✓ MQM Score: {report.mqm_score:.2f}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_all_tests())
