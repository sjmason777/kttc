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

"""Multi-provider orchestration for parallel LLM API calls.

This module provides resilient, parallel execution across multiple LLM providers
with circuit breakers, rate limiting, fallbacks, and result aggregation.

Key components:
- CircuitBreaker: Prevents cascading failures by temporarily disabling failing providers
- TokenAwareRateLimiter: Rate limiting based on tokens per minute (not just requests)
- MultiProviderOrchestrator: Coordinates parallel calls with resilience patterns
- Aggregators: Strategies for combining results from multiple providers

Example:
    >>> from kttc.llm.multi_provider import MultiProviderOrchestrator, ProviderConfig
    >>> from kttc.llm import OpenAIProvider, AnthropicProvider
    >>>
    >>> orchestrator = MultiProviderOrchestrator()
    >>> orchestrator.register_provider(
    ...     "openai",
    ...     OpenAIProvider(api_key="sk-..."),
    ...     ProviderConfig(rpm_limit=60, tpm_limit=90000)
    ... )
    >>> orchestrator.register_provider(
    ...     "anthropic",
    ...     AnthropicProvider(api_key="sk-ant-..."),
    ...     ProviderConfig(rpm_limit=60, tpm_limit=100000)
    ... )
    >>>
    >>> # Parallel ensemble call
    >>> results = await orchestrator.ensemble_call(
    ...     prompt="Translate: Hello world",
    ...     providers=["openai", "anthropic"],
    ...     aggregation="weighted_vote"
    ... )
"""

from .aggregators import (
    BaseAggregator,
    FirstSuccessAggregator,
    MajorityVoteAggregator,
    QualityScoreAggregator,
    WeightedVoteAggregator,
)
from .circuit_breaker import CircuitBreaker, CircuitState
from .config import (
    DEFAULT_FALLBACK_CHAIN,
    AggregationStrategy,
    LLMSelectionConfig,
    OrchestratorConfig,
    ProviderConfig,
    RoutingStrategy,
    create_ensemble_config,
    create_single_provider_config,
)
from .orchestrator import MultiProviderOrchestrator, ProviderResult
from .rate_limiter import TokenAwareRateLimiter, TokenBucket

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    # Rate limiting
    "TokenAwareRateLimiter",
    "TokenBucket",
    # Configuration
    "ProviderConfig",
    "OrchestratorConfig",
    "AggregationStrategy",
    "RoutingStrategy",
    "LLMSelectionConfig",
    "create_single_provider_config",
    "create_ensemble_config",
    "DEFAULT_FALLBACK_CHAIN",
    # Orchestrator
    "MultiProviderOrchestrator",
    "ProviderResult",
    # Aggregators
    "BaseAggregator",
    "FirstSuccessAggregator",
    "WeightedVoteAggregator",
    "MajorityVoteAggregator",
    "QualityScoreAggregator",
]
