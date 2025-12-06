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

"""Configuration models for multi-provider orchestration.

Defines configuration for providers, routing strategies, and aggregation methods.
Provider priority is set with Russian providers first:
1. Yandex GPT (default)
2. GigaChat
3. OpenAI
4. Anthropic
5. Gemini
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderPriority(Enum):
    """Provider priority levels.

    Priority order (default):
    1. YANDEX - Russian market, good for Russian language
    2. GIGACHAT - Sber's model, excellent for Russian
    3. OPENAI - Global leader, good all-around
    4. ANTHROPIC - High quality, good reasoning
    5. GEMINI - Google's model, multilingual
    """

    YANDEX = 1  # Highest priority (default)
    GIGACHAT = 2  # Second priority
    OPENAI = 3
    ANTHROPIC = 4
    GEMINI = 5  # Lowest priority


class RoutingStrategy(Enum):
    """Strategy for selecting which provider to use.

    Strategies:
    - PRIORITY: Use providers in priority order (Yandex → GigaChat → others)
    - ROUND_ROBIN: Distribute requests evenly across providers
    - LEAST_LATENCY: Route to provider with lowest recent latency
    - LEAST_BUSY: Route to provider with most available capacity
    - COST_OPTIMIZED: Route to cheapest provider that meets quality threshold
    - RANDOM: Random selection with weights
    """

    PRIORITY = "priority"  # Default: Yandex first, GigaChat second
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    LEAST_BUSY = "least_busy"
    COST_OPTIMIZED = "cost_optimized"
    RANDOM = "random"


class AggregationStrategy(Enum):
    """Strategy for aggregating results from multiple providers.

    Strategies:
    - FIRST_SUCCESS: Return first successful response
    - FASTEST: Return fastest response (race)
    - MAJORITY_VOTE: Majority voting for categorical outputs
    - WEIGHTED_VOTE: Weighted voting based on provider trust scores
    - QUALITY_SCORE: Use quality scoring to pick best response
    - LLM_JUDGE: Use another LLM to evaluate and pick best
    - CONSENSUS: Require agreement between providers
    """

    FIRST_SUCCESS = "first_success"
    FASTEST = "fastest"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    QUALITY_SCORE = "quality_score"
    LLM_JUDGE = "llm_judge"
    CONSENSUS = "consensus"


class FallbackBehavior(Enum):
    """Behavior when primary provider fails.

    Behaviors:
    - NEXT_PRIORITY: Try next provider in priority order
    - ALL_PARALLEL: Try all remaining providers in parallel
    - NONE: No fallback, fail immediately
    """

    NEXT_PRIORITY = "next_priority"
    ALL_PARALLEL = "all_parallel"
    NONE = "none"


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider.

    Attributes:
        priority: Provider priority (lower = higher priority)
        weight: Weight for load balancing and voting (0.0-1.0)
        rpm_limit: Requests per minute limit
        tpm_limit: Tokens per minute limit
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds (exponential backoff)
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Seconds before attempting recovery
        cost_per_1k_input: Cost per 1K input tokens (USD)
        cost_per_1k_output: Cost per 1K output tokens (USD)
        quality_score: Estimated quality score (0.0-1.0) for this provider
        enabled: Whether this provider is enabled
        metadata: Additional provider-specific configuration
    """

    priority: int = 100  # Lower = higher priority
    weight: float = 1.0  # For load balancing
    rpm_limit: int = 60
    tpm_limit: int = 90000
    max_concurrent: int = 10
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    quality_score: float = 0.8
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# Default configurations for each provider
# Priority: Yandex (1) → GigaChat (2) → OpenAI (3) → Anthropic (4) → Gemini (5)
DEFAULT_PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    # Yandex GPT - DEFAULT PRIMARY PROVIDER
    "yandex": ProviderConfig(
        priority=1,  # Highest priority
        weight=1.0,
        rpm_limit=60,
        tpm_limit=100000,
        max_concurrent=10,
        timeout=60.0,
        max_retries=3,
        cost_per_1k_input=0.0004,  # Approximate
        cost_per_1k_output=0.0012,
        quality_score=0.85,
    ),
    # GigaChat - SECOND PRIORITY
    "gigachat": ProviderConfig(
        priority=2,  # Second priority
        weight=0.95,
        rpm_limit=60,
        tpm_limit=80000,
        max_concurrent=10,
        timeout=60.0,
        max_retries=3,
        cost_per_1k_input=0.0003,  # Approximate
        cost_per_1k_output=0.001,
        quality_score=0.83,
    ),
    # OpenAI - THIRD PRIORITY
    "openai": ProviderConfig(
        priority=3,
        weight=0.9,
        rpm_limit=60,
        tpm_limit=90000,
        max_concurrent=10,
        timeout=60.0,
        max_retries=3,
        cost_per_1k_input=0.0025,  # GPT-4o
        cost_per_1k_output=0.01,
        quality_score=0.90,
    ),
    # Anthropic - FOURTH PRIORITY
    "anthropic": ProviderConfig(
        priority=4,
        weight=0.85,
        rpm_limit=60,
        tpm_limit=100000,
        max_concurrent=10,
        timeout=60.0,
        max_retries=3,
        cost_per_1k_input=0.003,  # Claude 3.5 Sonnet
        cost_per_1k_output=0.015,
        quality_score=0.92,
    ),
    # Gemini - FIFTH PRIORITY
    "gemini": ProviderConfig(
        priority=5,
        weight=0.8,
        rpm_limit=60,
        tpm_limit=100000,
        max_concurrent=10,
        timeout=60.0,
        max_retries=3,
        cost_per_1k_input=0.00025,  # Gemini 2.0 Flash
        cost_per_1k_output=0.001,
        quality_score=0.85,
    ),
}


@dataclass
class OrchestratorConfig:
    """Configuration for multi-provider orchestrator.

    Attributes:
        routing_strategy: How to select providers for single calls
        fallback_behavior: What to do when primary fails
        aggregation_strategy: How to combine ensemble results
        default_timeout: Default request timeout
        enable_circuit_breakers: Whether to use circuit breakers
        enable_rate_limiting: Whether to use rate limiting
        enable_retries: Whether to retry failed requests
        retry_jitter: Add random jitter to retry delays
        max_parallel_providers: Max providers to call in parallel for ensemble
        min_successful_responses: Minimum successful responses for consensus
        log_all_responses: Log responses from all providers (for debugging)
        fallback_chain: Ordered list of provider names for fallback
                       Default: ["yandex", "gigachat", "openai", "anthropic", "gemini"]
    """

    routing_strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    fallback_behavior: FallbackBehavior = FallbackBehavior.NEXT_PRIORITY
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FIRST_SUCCESS
    default_timeout: float = 60.0
    enable_circuit_breakers: bool = True
    enable_rate_limiting: bool = True
    enable_retries: bool = True
    retry_jitter: bool = True
    max_parallel_providers: int = 5
    min_successful_responses: int = 1
    log_all_responses: bool = False
    fallback_chain: list[str] = field(
        default_factory=lambda: [
            "yandex",  # First fallback
            "gigachat",  # Second fallback
            "openai",  # Third fallback
            "anthropic",  # Fourth fallback
            "gemini",  # Fifth fallback
        ]
    )

    def get_provider_priority_order(self) -> list[str]:
        """Get providers ordered by priority.

        Returns:
            List of provider names in priority order
        """
        return self.fallback_chain.copy()


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay before first retry (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Exception types that should trigger retry
        retryable_status_codes: HTTP status codes that should trigger retry
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)
    retryable_status_codes: tuple[int, ...] = field(
        default_factory=lambda: (429, 500, 502, 503, 504)
    )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        import random

        delay = min(
            self.initial_delay * (self.exponential_base**attempt),
            self.max_delay,
        )

        if self.jitter:
            # Add jitter: 0.5x to 1.5x of calculated delay
            delay = delay * (0.5 + random.random())  # nosec B311 - not for crypto

        return delay


def get_default_config_for_provider(provider_name: str) -> ProviderConfig:
    """Get default configuration for a provider.

    Args:
        provider_name: Provider name (yandex, gigachat, openai, anthropic, gemini)

    Returns:
        Default ProviderConfig for the provider
    """
    return DEFAULT_PROVIDER_CONFIGS.get(
        provider_name.lower(),
        ProviderConfig(priority=100),  # Unknown providers get lowest priority
    )


@dataclass
class LLMSelectionConfig:
    """Configuration for LLM selection.

    Allows specifying which LLMs to use and how many.

    Attributes:
        count: Number of LLMs to use (1 = single provider, >1 = ensemble)
        providers: List of specific providers to use (None = use priority order)
        ensemble_mode: Whether to run providers in parallel for ensemble
        fallback_enabled: Whether to fall back to other providers on failure

    Examples:
        # Single Yandex (default)
        config = LLMSelectionConfig()

        # Single Anthropic
        config = LLMSelectionConfig(count=1, providers=["anthropic"])

        # Ensemble of 3 providers
        config = LLMSelectionConfig(count=3, providers=["yandex", "gigachat", "openai"])

        # All available providers
        config = LLMSelectionConfig(count=5)
    """

    count: int = 1  # Default: single LLM
    providers: list[str] | None = None  # None = use priority order (Yandex first)
    ensemble_mode: bool = False  # True = parallel calls, False = priority/fallback
    fallback_enabled: bool = True  # Enable fallback on failure

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.count = max(self.count, 1)
        self.count = min(self.count, 5)
        if self.providers and len(self.providers) < self.count:
            # If specified fewer providers than count, adjust count
            self.count = len(self.providers)

    def get_selected_providers(self) -> list[str]:
        """Get list of providers to use based on configuration.

        Returns:
            List of provider names in priority order
        """
        # Default priority order
        default_order = ["yandex", "gigachat", "openai", "anthropic", "gemini"]

        if self.providers:
            # Use specified providers in given order
            return self.providers[: self.count]
        # Use default priority order
        return default_order[: self.count]

    def to_orchestrator_config(self) -> OrchestratorConfig:
        """Convert to OrchestratorConfig.

        Returns:
            OrchestratorConfig based on this selection
        """
        selected = self.get_selected_providers()

        if self.count == 1 and not self.ensemble_mode:
            # Single provider mode
            return OrchestratorConfig(
                routing_strategy=RoutingStrategy.PRIORITY,
                fallback_behavior=(
                    FallbackBehavior.NEXT_PRIORITY
                    if self.fallback_enabled
                    else FallbackBehavior.NONE
                ),
                aggregation_strategy=AggregationStrategy.FIRST_SUCCESS,
                max_parallel_providers=1,
                fallback_chain=selected + [p for p in DEFAULT_FALLBACK_CHAIN if p not in selected],
            )
        # Ensemble mode
        return OrchestratorConfig(
            routing_strategy=RoutingStrategy.PRIORITY,
            fallback_behavior=FallbackBehavior.NONE,
            aggregation_strategy=AggregationStrategy.WEIGHTED_VOTE,
            max_parallel_providers=self.count,
            fallback_chain=selected,
        )


# Default fallback chain (Yandex first)
DEFAULT_FALLBACK_CHAIN: list[str] = [
    "yandex",
    "gigachat",
    "openai",
    "anthropic",
    "gemini",
]


def create_single_provider_config(provider: str = "yandex") -> LLMSelectionConfig:
    """Create config for single provider mode.

    Args:
        provider: Provider name (default: yandex)

    Returns:
        LLMSelectionConfig for single provider

    Examples:
        >>> config = create_single_provider_config()  # Yandex
        >>> config = create_single_provider_config("anthropic")  # Claude
        >>> config = create_single_provider_config("gigachat")  # GigaChat
    """
    return LLMSelectionConfig(count=1, providers=[provider])


def create_ensemble_config(
    count: int = 3,
    providers: list[str] | None = None,
) -> LLMSelectionConfig:
    """Create config for ensemble mode (parallel calls to multiple LLMs).

    Args:
        count: Number of LLMs to use
        providers: Specific providers (None = use priority order)

    Returns:
        LLMSelectionConfig for ensemble mode

    Examples:
        >>> config = create_ensemble_config(3)  # Top 3 by priority
        >>> config = create_ensemble_config(2, ["yandex", "anthropic"])
    """
    return LLMSelectionConfig(
        count=count,
        providers=providers,
        ensemble_mode=True,
        fallback_enabled=False,
    )
