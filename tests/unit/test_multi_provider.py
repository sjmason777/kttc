# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for multi-provider orchestration module."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from kttc.llm.base import BaseLLMProvider, LLMError, LLMRateLimitError
from kttc.llm.multi_provider.aggregators import (
    ConsensusAggregator,
    FirstSuccessAggregator,
    MajorityVoteAggregator,
    ProviderResponse,
    QualityScoreAggregator,
    WeightedVoteAggregator,
    get_aggregator,
)
from kttc.llm.multi_provider.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)
from kttc.llm.multi_provider.config import (
    DEFAULT_PROVIDER_CONFIGS,
    OrchestratorConfig,
    get_default_config_for_provider,
)
from kttc.llm.multi_provider.orchestrator import (
    MultiProviderOrchestrator,
)
from kttc.llm.multi_provider.rate_limiter import (
    RateLimitExceededError,
    TokenAwareRateLimiter,
    TokenBucket,
)

# =============================================================================
# Mock Provider
# =============================================================================


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        response: str = "mock response",
        delay: float = 0.0,
        fail_count: int = 0,
        fail_with: type[Exception] = LLMError,
    ):
        super().__init__()
        self.response = response
        self.delay = delay
        self.fail_count = fail_count
        self.fail_with = fail_with
        self.call_count = 0

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        self.call_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.call_count <= self.fail_count:
            raise self.fail_with(f"Mock error {self.call_count}")

        self._usage.add_usage(input_tokens=100, output_tokens=50)
        return self.response

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield self.response


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self) -> None:
        """Circuit breaker should start in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_opens_after_failures(self) -> None:
        """Circuit should open after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(name="test", config=config)

        # Record failures
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_rejects_when_open(self) -> None:
        """Circuit should reject requests when open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        breaker = CircuitBreaker(name="test", config=config)

        await breaker.record_failure()

        with pytest.raises(CircuitOpenError):
            async with breaker:
                pass

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self) -> None:
        """Success should reset consecutive failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(name="test", config=config)

        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.metrics.consecutive_failures == 2

        await breaker.record_success()
        assert breaker.metrics.consecutive_failures == 0
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_records_success(self) -> None:
        """Context manager should record success on normal exit."""
        breaker = CircuitBreaker(name="test")

        async with breaker:
            pass

        assert breaker.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_context_manager_records_failure(self) -> None:
        """Context manager should record failure on exception."""
        breaker = CircuitBreaker(name="test")

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker.metrics.failed_calls == 1

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        """get_status should return meaningful data."""
        breaker = CircuitBreaker(name="test")
        await breaker.record_success()

        status = breaker.get_status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["metrics"]["successful_calls"] == 1


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestTokenBucket:
    """Tests for TokenBucket."""

    @pytest.mark.asyncio
    async def test_initial_capacity(self) -> None:
        """Bucket should start at full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10

    @pytest.mark.asyncio
    async def test_consume_reduces_tokens(self) -> None:
        """Consuming should reduce token count."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        success = await bucket.consume(3)
        assert success
        assert bucket.tokens == 7

    @pytest.mark.asyncio
    async def test_consume_fails_when_empty(self) -> None:
        """Consuming should fail when not enough tokens."""
        bucket = TokenBucket(capacity=2, refill_rate=0.1)

        success = await bucket.consume(5)
        assert not success


class TestTokenAwareRateLimiter:
    """Tests for TokenAwareRateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self) -> None:
        """Should acquire and release properly."""
        limiter = TokenAwareRateLimiter(
            name="test",
            rpm_limit=60,
            tpm_limit=100000,
            max_concurrent=5,
        )

        async with limiter.acquire(estimated_tokens=100):
            assert limiter.metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_concurrent_limit(self) -> None:
        """Should respect concurrent request limit."""
        limiter = TokenAwareRateLimiter(
            name="test",
            rpm_limit=100,
            tpm_limit=100000,
            max_concurrent=2,
        )

        # Acquire all slots
        ctx1 = limiter.acquire(estimated_tokens=100)
        await ctx1.__aenter__()
        ctx2 = limiter.acquire(estimated_tokens=100)
        await ctx2.__aenter__()

        # Third should timeout quickly
        with pytest.raises(RateLimitExceededError):
            async with limiter.acquire(estimated_tokens=100, timeout=0.1):
                pass

        await ctx1.__aexit__(None, None, None)
        await ctx2.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        """get_status should return meaningful data."""
        limiter = TokenAwareRateLimiter(name="test")

        status = limiter.get_status()
        assert status["name"] == "test"
        assert "limits" in status
        assert "current" in status


# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for configuration."""

    def test_default_provider_priority(self) -> None:
        """Yandex should be first, GigaChat second."""
        yandex = DEFAULT_PROVIDER_CONFIGS["yandex"]
        gigachat = DEFAULT_PROVIDER_CONFIGS["gigachat"]
        openai = DEFAULT_PROVIDER_CONFIGS["openai"]

        assert yandex.priority < gigachat.priority
        assert gigachat.priority < openai.priority

    def test_get_default_config(self) -> None:
        """Should return correct default config."""
        config = get_default_config_for_provider("yandex")
        assert config.priority == 1

        config = get_default_config_for_provider("unknown")
        assert config.priority == 100

    def test_orchestrator_fallback_chain(self) -> None:
        """Fallback chain should have Yandex first."""
        config = OrchestratorConfig()
        assert config.fallback_chain[0] == "yandex"
        assert config.fallback_chain[1] == "gigachat"


# =============================================================================
# Orchestrator Tests
# =============================================================================


class TestMultiProviderOrchestrator:
    """Tests for MultiProviderOrchestrator."""

    @pytest.mark.asyncio
    async def test_register_provider(self) -> None:
        """Should register providers correctly."""
        orchestrator = MultiProviderOrchestrator()
        provider = MockLLMProvider()

        orchestrator.register_provider("yandex", provider)

        assert "yandex" in orchestrator.get_available_providers()

    @pytest.mark.asyncio
    async def test_priority_routing(self) -> None:
        """Should use Yandex (priority 1) first."""
        orchestrator = MultiProviderOrchestrator()

        yandex = MockLLMProvider(response="yandex response")
        gigachat = MockLLMProvider(response="gigachat response")
        openai = MockLLMProvider(response="openai response")

        orchestrator.register_provider("yandex", yandex)
        orchestrator.register_provider("gigachat", gigachat)
        orchestrator.register_provider("openai", openai)

        result = await orchestrator.call("test prompt")

        assert result.success
        assert result.response == "yandex response"
        assert result.provider_name == "yandex"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self) -> None:
        """Should fallback to next provider on failure."""
        orchestrator = MultiProviderOrchestrator()

        # Yandex fails
        yandex = MockLLMProvider(fail_count=10)
        gigachat = MockLLMProvider(response="gigachat response")

        orchestrator.register_provider("yandex", yandex)
        orchestrator.register_provider("gigachat", gigachat)

        result = await orchestrator.call("test prompt")

        assert result.success
        assert result.response == "gigachat response"
        assert result.provider_name == "gigachat"

    @pytest.mark.asyncio
    async def test_ensemble_call(self) -> None:
        """Should call multiple providers in parallel."""
        orchestrator = MultiProviderOrchestrator()

        yandex = MockLLMProvider(response="yandex")
        gigachat = MockLLMProvider(response="gigachat")

        orchestrator.register_provider("yandex", yandex)
        orchestrator.register_provider("gigachat", gigachat)

        result = await orchestrator.ensemble_call(
            "test prompt",
            providers=["yandex", "gigachat"],
        )

        assert result.success
        assert result.successful_providers == 2
        assert len(result.provider_results) == 2

    @pytest.mark.asyncio
    async def test_race_call(self) -> None:
        """Should return fastest response."""
        orchestrator = MultiProviderOrchestrator()

        # Slower provider
        slow = MockLLMProvider(response="slow", delay=0.5)
        # Faster provider
        fast = MockLLMProvider(response="fast", delay=0.01)

        orchestrator.register_provider("slow", slow)
        orchestrator.register_provider("fast", fast)

        result = await orchestrator.race_call(
            "test prompt",
            providers=["slow", "fast"],
        )

        assert result.success
        assert result.response == "fast"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self) -> None:
        """Should retry on rate limit errors."""
        orchestrator = MultiProviderOrchestrator()

        # Fails first 2 times with rate limit
        provider = MockLLMProvider(
            response="success",
            fail_count=2,
            fail_with=LLMRateLimitError,
        )

        orchestrator.register_provider("yandex", provider)

        result = await orchestrator.call("test prompt")

        assert result.success
        assert result.retries == 2

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        """get_status should return meaningful data."""
        orchestrator = MultiProviderOrchestrator()
        orchestrator.register_provider("yandex", MockLLMProvider())

        status = orchestrator.get_status()
        assert "config" in status
        assert "providers" in status
        assert "yandex" in status["providers"]


# =============================================================================
# Aggregator Tests
# =============================================================================


class TestAggregators:
    """Tests for aggregation strategies."""

    def _make_responses(self, responses: list[tuple[str, str, float]]) -> list[ProviderResponse]:
        """Helper to create ProviderResponse list."""
        return [
            ProviderResponse(
                provider_name=name,
                response=response,
                quality_score=quality,
            )
            for name, response, quality in responses
        ]

    def test_first_success(self) -> None:
        """FirstSuccessAggregator should return first response."""
        aggregator = FirstSuccessAggregator()
        responses = self._make_responses(
            [
                ("yandex", "first", 0.8),
                ("gigachat", "second", 0.9),
            ]
        )

        result = aggregator.aggregate(responses)
        assert result.response == "first"

    def test_majority_vote(self) -> None:
        """MajorityVoteAggregator should return most common response."""
        aggregator = MajorityVoteAggregator(similarity_threshold=0.9)
        responses = self._make_responses(
            [
                ("yandex", "winner", 0.8),
                ("gigachat", "winner", 0.8),
                ("openai", "loser", 0.9),
            ]
        )

        result = aggregator.aggregate(responses)
        assert result.response == "winner"
        assert result.metadata is not None
        assert result.metadata.get("vote_count") == 2

    def test_weighted_vote(self) -> None:
        """WeightedVoteAggregator should weight by quality score."""
        aggregator = WeightedVoteAggregator()
        responses = [
            ProviderResponse("low", "A", weight=0.5, quality_score=0.5),
            ProviderResponse("high", "B", weight=1.0, quality_score=0.95),
        ]

        result = aggregator.aggregate(responses)
        assert result.response == "B"

    def test_quality_score(self) -> None:
        """QualityScoreAggregator should select highest quality."""
        aggregator = QualityScoreAggregator()
        responses = self._make_responses(
            [
                ("low", "bad", 0.5),
                ("high", "good", 0.95),
            ]
        )

        result = aggregator.aggregate(responses)
        assert result.response == "good"

    def test_consensus_reached(self) -> None:
        """ConsensusAggregator should detect consensus."""
        aggregator = ConsensusAggregator(min_agreement=0.5)
        responses = self._make_responses(
            [
                ("a", "same", 0.8),
                ("b", "same", 0.8),
                ("c", "different", 0.9),
            ]
        )

        result = aggregator.aggregate(responses)
        assert result.response == "same"
        assert result.metadata is not None
        assert result.metadata.get("consensus_reached") is True

    def test_consensus_not_reached(self) -> None:
        """ConsensusAggregator should fail when no consensus."""
        aggregator = ConsensusAggregator(min_agreement=0.9)
        responses = self._make_responses(
            [
                ("a", "one", 0.8),
                ("b", "two", 0.8),
                ("c", "three", 0.8),
            ]
        )

        result = aggregator.aggregate(responses)
        assert result.response is None
        assert result.metadata is not None
        assert result.metadata.get("consensus_reached") is False

    def test_get_aggregator_factory(self) -> None:
        """get_aggregator should return correct aggregator type."""
        assert isinstance(get_aggregator("first_success"), FirstSuccessAggregator)
        assert isinstance(get_aggregator("majority_vote"), MajorityVoteAggregator)
        assert isinstance(get_aggregator("weighted_vote"), WeightedVoteAggregator)
        assert isinstance(get_aggregator("quality_score"), QualityScoreAggregator)
        assert isinstance(get_aggregator("consensus"), ConsensusAggregator)

    def test_empty_responses(self) -> None:
        """Aggregators should handle empty input."""
        for aggregator in [
            FirstSuccessAggregator(),
            MajorityVoteAggregator(),
            WeightedVoteAggregator(),
            QualityScoreAggregator(),
            ConsensusAggregator(),
        ]:
            result = aggregator.aggregate([])
            assert result.response is None
            assert result.confidence == 0.0
