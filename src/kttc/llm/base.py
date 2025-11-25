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

"""Base abstract class for LLM providers.

Defines the interface that all LLM providers must implement.
Supports both OpenAI and Anthropic APIs.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Track token usage and costs for LLM API calls.

    Attributes:
        input_tokens: Total input/prompt tokens used
        output_tokens: Total output/completion tokens used
        total_tokens: Sum of input and output tokens
        estimated_cost_usd: Estimated cost in USD (based on model pricing)
        call_count: Number of API calls made
    """

    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    model_name: str = ""
    _costs_per_1k: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Pricing per 1K tokens (input, output) - November 2025 rates
    DEFAULT_PRICING: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            # OpenAI
            "gpt-4o": (0.0025, 0.01),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            # Anthropic
            "claude-3-5-sonnet": (0.003, 0.015),
            "claude-3-5-haiku": (0.0008, 0.004),
            "claude-3-opus": (0.015, 0.075),
            "claude-sonnet-4": (0.003, 0.015),
            # Default fallback
            "default": (0.002, 0.006),
        }
    )

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost in USD based on model pricing."""
        pricing = self.DEFAULT_PRICING.get(
            self.model_name, self.DEFAULT_PRICING.get("default", (0.002, 0.006))
        )
        input_cost = (self.input_tokens / 1000) * pricing[0]
        output_cost = (self.output_tokens / 1000) * pricing[1]
        return round(input_cost + output_cost, 6)

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Add usage from an API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.call_count += 1

    def reset(self) -> None:
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        return (
            f"Tokens: {self.total_tokens:,} "
            f"(in: {self.input_tokens:,}, out: {self.output_tokens:,}) | "
            f"Calls: {self.call_count} | "
            f"Cost: ${self.estimated_cost_usd:.4f}"
        )


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (OpenAI, Anthropic, etc.) must implement this interface.
    Supports both synchronous completion and streaming.

    Attributes:
        usage: TokenUsage instance tracking API usage and costs
    """

    def __init__(self) -> None:
        """Initialize provider with usage tracking."""
        self._usage = TokenUsage()

    @property
    def usage(self) -> TokenUsage:
        """Get current token usage statistics."""
        return self._usage

    def reset_usage(self) -> None:
        """Reset usage counters."""
        self._usage.reset()

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a single completion from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            The generated text response

        Raises:
            LLMError: If the API call fails
            TimeoutError: If the request times out

        Example:
            >>> provider = OpenAIProvider(api_key="...")
            >>> response = await provider.complete("Translate: Hello")
            >>> print(response)
            'Hola'
        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Yields:
            Text chunks as they are generated

        Example:
            >>> provider = OpenAIProvider(api_key="...")
            >>> async for chunk in provider.stream("Translate: Hello"):
            ...     print(chunk, end="", flush=True)
            Hola
        """
        pass


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when hitting rate limits."""

    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails."""

    pass
