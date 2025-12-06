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

"""Multi-provider orchestrator for parallel LLM API calls.

Coordinates requests across multiple LLM providers with:
- Parallel execution via asyncio.gather
- Circuit breakers for fault tolerance
- Rate limiting (token-aware)
- Automatic fallbacks
- Result aggregation

Default provider priority:
1. Yandex GPT
2. GigaChat
3. OpenAI
4. Anthropic
5. Gemini

Example:
    >>> orchestrator = MultiProviderOrchestrator()
    >>> orchestrator.register_provider("yandex", yandex_provider)
    >>> orchestrator.register_provider("gigachat", gigachat_provider)
    >>>
    >>> # Single call with fallback
    >>> result = await orchestrator.call(prompt="Hello")
    >>>
    >>> # Parallel ensemble call
    >>> results = await orchestrator.ensemble_call(
    ...     prompt="Analyze this text",
    ...     providers=["yandex", "gigachat", "openai"]
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from ..base import BaseLLMProvider, LLMError, LLMRateLimitError, LLMTimeoutError
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError
from .config import (
    AggregationStrategy,
    FallbackBehavior,
    OrchestratorConfig,
    ProviderConfig,
    RetryConfig,
    RoutingStrategy,
    get_default_config_for_provider,
)
from .rate_limiter import RateLimitExceededError, TokenAwareRateLimiter

logger = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    """Result from a single provider call.

    Attributes:
        provider_name: Name of the provider
        success: Whether the call succeeded
        response: Response text (if successful)
        error: Error message (if failed)
        latency: Time taken in seconds
        tokens_used: Tokens consumed
        retries: Number of retries attempted
        metadata: Additional result metadata
    """

    provider_name: str
    success: bool
    response: str | None = None
    error: str | None = None
    latency: float = 0.0
    tokens_used: int = 0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Result from ensemble call to multiple providers.

    Attributes:
        aggregated_response: Final aggregated response
        provider_results: Results from each provider
        aggregation_strategy: Strategy used for aggregation
        successful_providers: Number of providers that succeeded
        total_latency: Total time for all calls
        metadata: Additional result metadata
    """

    aggregated_response: str | None
    provider_results: list[ProviderResult]
    aggregation_strategy: AggregationStrategy
    successful_providers: int
    total_latency: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if ensemble call was successful."""
        return self.aggregated_response is not None


@dataclass
class ProviderState:
    """Internal state for a registered provider."""

    provider: BaseLLMProvider
    config: ProviderConfig
    circuit_breaker: CircuitBreaker
    rate_limiter: TokenAwareRateLimiter
    latency_history: list[float] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0

    @property
    def average_latency(self) -> float:
        """Calculate average latency from recent history."""
        if not self.latency_history:
            return float("inf")
        # Use last 10 latencies
        recent = self.latency_history[-10:]
        return sum(recent) / len(recent)

    def record_latency(self, latency: float) -> None:
        """Record a latency measurement."""
        self.latency_history.append(latency)
        # Keep only last 100 measurements
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]


class MultiProviderOrchestrator:
    """Orchestrates calls across multiple LLM providers.

    Features:
    - Parallel execution with asyncio.gather
    - Circuit breakers prevent cascading failures
    - Token-aware rate limiting
    - Automatic fallbacks on failure
    - Multiple aggregation strategies
    - Latency-based and priority-based routing

    Default priority: Yandex → GigaChat → OpenAI → Anthropic → Gemini

    Example:
        >>> orchestrator = MultiProviderOrchestrator()
        >>>
        >>> # Register providers
        >>> orchestrator.register_provider("yandex", yandex_provider)
        >>> orchestrator.register_provider("gigachat", gigachat_provider)
        >>>
        >>> # Single call (uses priority routing with fallback)
        >>> result = await orchestrator.call("Translate: Hello")
        >>>
        >>> # Ensemble call (parallel to multiple providers)
        >>> ensemble = await orchestrator.ensemble_call(
        ...     prompt="Analyze this",
        ...     providers=["yandex", "gigachat"],
        ...     aggregation=AggregationStrategy.WEIGHTED_VOTE
        ... )
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            config: Orchestrator configuration (uses defaults if None)
        """
        self.config = config or OrchestratorConfig()
        self._providers: dict[str, ProviderState] = {}
        self._round_robin_index = 0

    def register_provider(
        self,
        name: str,
        provider: BaseLLMProvider,
        config: ProviderConfig | None = None,
    ) -> None:
        """Register a provider with the orchestrator.

        Args:
            name: Unique name for the provider
            provider: LLM provider instance
            config: Provider configuration (uses defaults based on name if None)
        """
        # Get default config if not provided
        if config is None:
            config = get_default_config_for_provider(name)

        # Create circuit breaker
        circuit_breaker = CircuitBreaker(
            name=f"circuit_{name}",
            config=CircuitBreakerConfig(
                failure_threshold=config.circuit_breaker_threshold,
                recovery_timeout=config.circuit_breaker_timeout,
            ),
        )

        # Create rate limiter
        rate_limiter = TokenAwareRateLimiter(
            name=f"rate_{name}",
            rpm_limit=config.rpm_limit,
            tpm_limit=config.tpm_limit,
            max_concurrent=config.max_concurrent,
        )

        self._providers[name] = ProviderState(
            provider=provider,
            config=config,
            circuit_breaker=circuit_breaker,
            rate_limiter=rate_limiter,
        )

        logger.info(f"Registered provider '{name}' with priority {config.priority}")

    def unregister_provider(self, name: str) -> None:
        """Unregister a provider.

        Args:
            name: Provider name to unregister
        """
        if name in self._providers:
            del self._providers[name]
            logger.info(f"Unregistered provider '{name}'")

    def get_available_providers(self) -> list[str]:
        """Get list of available (enabled and circuit not open) providers.

        Returns:
            List of available provider names
        """
        available = []
        for name, state in self._providers.items():
            if state.config.enabled and state.circuit_breaker.can_execute():
                available.append(name)
        return available

    def get_providers_by_priority(self) -> list[str]:
        """Get providers ordered by priority.

        Returns:
            List of provider names ordered by priority (lowest number first)
        """
        return sorted(
            self._providers.keys(),
            key=lambda n: self._providers[n].config.priority,
        )

    def _select_provider(self, strategy: RoutingStrategy | None = None) -> str | None:
        """Select a provider based on routing strategy.

        Args:
            strategy: Routing strategy (uses config default if None)

        Returns:
            Selected provider name, or None if none available
        """
        strategy = strategy or self.config.routing_strategy
        available = self.get_available_providers()

        if not available:
            return None

        if strategy == RoutingStrategy.PRIORITY:
            # Sort by priority and return first available
            sorted_providers = sorted(
                available,
                key=lambda n: self._providers[n].config.priority,
            )
            return sorted_providers[0] if sorted_providers else None

        if strategy == RoutingStrategy.ROUND_ROBIN:
            # Round-robin through available providers
            self._round_robin_index = (self._round_robin_index + 1) % len(available)
            return available[self._round_robin_index]

        if strategy == RoutingStrategy.LEAST_LATENCY:
            # Select provider with lowest average latency
            return min(
                available,
                key=lambda n: self._providers[n].average_latency,
            )

        if strategy == RoutingStrategy.LEAST_BUSY:
            # Select provider with most available rate limit capacity
            def get_capacity(name: str) -> float:
                limiter = self._providers[name].rate_limiter
                return limiter.concurrent_available / limiter.max_concurrent

            return max(available, key=get_capacity)

        if strategy == RoutingStrategy.COST_OPTIMIZED:
            # Select cheapest provider
            return min(
                available,
                key=lambda n: (
                    self._providers[n].config.cost_per_1k_input
                    + self._providers[n].config.cost_per_1k_output
                ),
            )

        if strategy == RoutingStrategy.RANDOM:
            # Weighted random selection
            weights = [self._providers[n].config.weight for n in available]
            return random.choices(available, weights=weights, k=1)[0]  # nosec B311

        return available[0] if available else None

    async def _call_provider_with_retry(
        self,
        name: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> ProviderResult:
        """Call a single provider with retry logic.

        Args:
            name: Provider name
            prompt: Prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider parameters

        Returns:
            ProviderResult with response or error
        """
        state = self._providers.get(name)
        if not state:
            return ProviderResult(
                provider_name=name,
                success=False,
                error=f"Provider '{name}' not registered",
            )

        if not state.config.enabled:
            return ProviderResult(
                provider_name=name,
                success=False,
                error=f"Provider '{name}' is disabled",
            )

        retry_config = RetryConfig(
            max_attempts=state.config.max_retries,
            initial_delay=state.config.retry_delay,
        )

        start_time = time.monotonic()
        last_error: Exception | None = None
        retries = 0

        for attempt in range(retry_config.max_attempts):
            try:
                # Check circuit breaker
                if not state.circuit_breaker.can_execute():
                    raise CircuitOpenError(f"Circuit breaker open for '{name}'")

                # Estimate tokens (rough: 4 chars per token)
                estimated_tokens = (len(prompt) + max_tokens * 4) // 4

                # Acquire rate limit
                async with state.rate_limiter.acquire(
                    estimated_tokens=estimated_tokens,
                    timeout=state.config.timeout,
                ):
                    # Make the actual call with circuit breaker
                    async with state.circuit_breaker:
                        response = await asyncio.wait_for(
                            state.provider.complete(
                                prompt=prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                **kwargs,
                            ),
                            timeout=state.config.timeout,
                        )

                # Success!
                latency = time.monotonic() - start_time
                state.record_latency(latency)
                state.success_count += 1

                # Get token usage
                tokens_used = 0
                if hasattr(state.provider, "usage"):
                    tokens_used = state.provider.usage.total_tokens

                return ProviderResult(
                    provider_name=name,
                    success=True,
                    response=response,
                    latency=latency,
                    tokens_used=tokens_used,
                    retries=retries,
                )

            except CircuitOpenError as e:
                # Circuit is open, don't retry
                return ProviderResult(
                    provider_name=name,
                    success=False,
                    error=str(e),
                    latency=time.monotonic() - start_time,
                )

            except RateLimitExceededError as e:
                # Rate limit exceeded, don't retry immediately
                last_error = e
                retries += 1

            except LLMRateLimitError as e:
                # Provider rate limit (429), retry with backoff
                last_error = e
                retries += 1
                if attempt < retry_config.max_attempts - 1:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(f"Provider '{name}' rate limited, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)

            except (TimeoutError, LLMTimeoutError) as e:
                # Timeout, retry
                last_error = e
                retries += 1
                if attempt < retry_config.max_attempts - 1:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(f"Provider '{name}' timed out, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)

            except LLMError as e:
                # Other LLM error, retry
                last_error = e
                retries += 1
                if attempt < retry_config.max_attempts - 1:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(f"Provider '{name}' error: {e}, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)

            except Exception as e:
                # Unexpected error, don't retry
                last_error = e
                break

        # All retries exhausted
        state.failure_count += 1
        latency = time.monotonic() - start_time

        return ProviderResult(
            provider_name=name,
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            latency=latency,
            retries=retries,
        )

    async def _execute_sequential_fallback(
        self,
        fallback_chain: list[str],
        selected: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> ProviderResult | None:
        """Execute fallback providers sequentially until one succeeds."""
        for fallback_provider in fallback_chain:
            logger.info(f"Falling back from '{selected}' to '{fallback_provider}'")
            result = await self._call_provider_with_retry(
                name=fallback_provider,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if result.success:
                return result
        return None

    async def _execute_parallel_fallback(
        self,
        fallback_chain: list[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> ProviderResult | None:
        """Execute all fallback providers in parallel and return first success."""
        if not fallback_chain:
            return None

        logger.info(f"Parallel fallback to: {fallback_chain}")
        tasks = [
            self._call_provider_with_retry(
                name=p,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            for p in fallback_chain
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, ProviderResult) and r.success:
                return r
        return None

    async def call(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        provider: str | None = None,
        routing_strategy: RoutingStrategy | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> ProviderResult:
        """Call a single provider with optional fallback.

        Default behavior: Uses Yandex first, falls back through priority chain.

        Args:
            prompt: Prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider: Specific provider to use (None for automatic selection)
            routing_strategy: Override routing strategy
            fallback: Whether to try fallback providers on failure
            **kwargs: Additional provider parameters

        Returns:
            ProviderResult with response or final error
        """
        selected = provider or self._select_provider(routing_strategy)

        if not selected:
            return ProviderResult(
                provider_name="none",
                success=False,
                error="No available providers",
            )

        result = await self._call_provider_with_retry(
            name=selected,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if result.success or not fallback:
            return result

        if self.config.fallback_behavior == FallbackBehavior.NONE:
            return result

        fallback_chain = [
            p for p in self.config.fallback_chain if p != selected and p in self._providers
        ]

        fallback_result: ProviderResult | None = None
        if self.config.fallback_behavior == FallbackBehavior.NEXT_PRIORITY:
            fallback_result = await self._execute_sequential_fallback(
                fallback_chain, selected, prompt, temperature, max_tokens, **kwargs
            )
        elif self.config.fallback_behavior == FallbackBehavior.ALL_PARALLEL:
            fallback_result = await self._execute_parallel_fallback(
                fallback_chain, prompt, temperature, max_tokens, **kwargs
            )

        return fallback_result if fallback_result else result

    async def ensemble_call(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        providers: list[str] | None = None,
        aggregation: AggregationStrategy | None = None,
        min_responses: int | None = None,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Call multiple providers in parallel and aggregate results.

        Args:
            prompt: Prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            providers: List of providers to call (None for all available)
            aggregation: Aggregation strategy
            min_responses: Minimum successful responses required
            **kwargs: Additional provider parameters

        Returns:
            EnsembleResult with aggregated response
        """
        start_time = time.monotonic()
        aggregation = aggregation or self.config.aggregation_strategy
        min_responses = min_responses or self.config.min_successful_responses

        # Select providers
        if providers:
            selected = [p for p in providers if p in self._providers]
        else:
            selected = self.get_available_providers()[: self.config.max_parallel_providers]

        if not selected:
            return EnsembleResult(
                aggregated_response=None,
                provider_results=[],
                aggregation_strategy=aggregation,
                successful_providers=0,
                total_latency=0.0,
                metadata={"error": "No providers available"},
            )

        logger.info(f"Ensemble call to providers: {selected}")

        # Call all providers in parallel
        tasks = [
            self._call_provider_with_retry(
                name=p,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            for p in selected
        ]

        results: list[ProviderResult] = []
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in raw_results:
            if isinstance(r, ProviderResult):
                results.append(r)
            elif isinstance(r, Exception):
                results.append(
                    ProviderResult(
                        provider_name="unknown",
                        success=False,
                        error=str(r),
                    )
                )

        # Log results if configured
        if self.config.log_all_responses:
            for r in results:
                logger.debug(
                    f"Provider '{r.provider_name}': "
                    f"success={r.success}, latency={r.latency:.2f}s"
                )

        successful = [r for r in results if r.success]
        total_latency = time.monotonic() - start_time

        # Check minimum responses
        if len(successful) < min_responses:
            return EnsembleResult(
                aggregated_response=None,
                provider_results=results,
                aggregation_strategy=aggregation,
                successful_providers=len(successful),
                total_latency=total_latency,
                metadata={
                    "error": f"Only {len(successful)} successful responses, "
                    f"minimum {min_responses} required"
                },
            )

        # Aggregate results
        aggregated = self._aggregate_results(successful, aggregation)

        return EnsembleResult(
            aggregated_response=aggregated,
            provider_results=results,
            aggregation_strategy=aggregation,
            successful_providers=len(successful),
            total_latency=total_latency,
        )

    def _aggregate_results(
        self,
        results: list[ProviderResult],
        strategy: AggregationStrategy,
    ) -> str | None:
        """Aggregate results from multiple providers.

        Args:
            results: List of successful provider results
            strategy: Aggregation strategy

        Returns:
            Aggregated response text
        """
        if not results:
            return None

        if strategy == AggregationStrategy.FIRST_SUCCESS:
            return results[0].response

        if strategy == AggregationStrategy.FASTEST:
            fastest = min(results, key=lambda r: r.latency)
            return fastest.response

        if strategy == AggregationStrategy.WEIGHTED_VOTE:
            # Weight responses by provider quality score
            weighted_responses: dict[str, float] = {}
            for r in results:
                if r.response:
                    state = self._providers.get(r.provider_name)
                    weight = state.config.quality_score if state else 0.5
                    if r.response in weighted_responses:
                        weighted_responses[r.response] += weight
                    else:
                        weighted_responses[r.response] = weight

            if weighted_responses:
                return max(weighted_responses, key=lambda k: weighted_responses[k])
            return results[0].response

        if strategy == AggregationStrategy.QUALITY_SCORE:
            # Return response from highest quality provider
            best = max(
                results,
                key=lambda r: (
                    self._providers[r.provider_name].config.quality_score
                    if r.provider_name in self._providers
                    else 0.5
                ),
            )
            return best.response

        # Default: first success
        return results[0].response

    async def race_call(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        providers: list[str] | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        """Race multiple providers, return first successful response.

        Cancels remaining tasks once first response arrives.

        Args:
            prompt: Prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            providers: List of providers to race (None for all available)
            **kwargs: Additional provider parameters

        Returns:
            First successful ProviderResult
        """
        # Select providers
        if providers:
            selected = [p for p in providers if p in self._providers]
        else:
            selected = self.get_available_providers()[: self.config.max_parallel_providers]

        if not selected:
            return ProviderResult(
                provider_name="none",
                success=False,
                error="No providers available",
            )

        logger.info(f"Race call to providers: {selected}")

        # Create tasks
        tasks = {
            asyncio.create_task(
                self._call_provider_with_retry(
                    name=p,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            ): p
            for p in selected
        }

        try:
            # Wait for first completed task
            while tasks:
                done, pending = await asyncio.wait(
                    tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    result = task.result()
                    if result.success:
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        return result
                    # Remove failed task and continue
                    del tasks[task]

        except Exception as e:
            # Cancel all tasks on error
            for task in tasks:
                task.cancel()
            return ProviderResult(
                provider_name="none",
                success=False,
                error=str(e),
            )

        # All tasks failed
        return ProviderResult(
            provider_name="none",
            success=False,
            error="All providers failed",
        )

    def get_status(self) -> dict[str, Any]:
        """Get detailed status of all providers.

        Returns:
            Dictionary with status of each provider
        """
        providers_status: dict[str, Any] = {}
        status: dict[str, Any] = {
            "config": {
                "routing_strategy": self.config.routing_strategy.value,
                "fallback_behavior": self.config.fallback_behavior.value,
                "aggregation_strategy": self.config.aggregation_strategy.value,
            },
            "providers": providers_status,
        }

        for name, state in self._providers.items():
            providers_status[name] = {
                "enabled": state.config.enabled,
                "priority": state.config.priority,
                "circuit_breaker": state.circuit_breaker.get_status(),
                "rate_limiter": state.rate_limiter.get_status(),
                "metrics": {
                    "success_count": state.success_count,
                    "failure_count": state.failure_count,
                    "average_latency": f"{state.average_latency:.3f}s",
                },
            }

        return status

    def reset_all(self) -> None:
        """Reset all circuit breakers and rate limiters."""
        for state in self._providers.values():
            state.circuit_breaker.reset()
            state.rate_limiter.reset()
            state.latency_history.clear()
            state.success_count = 0
            state.failure_count = 0
        logger.info("All providers reset")
