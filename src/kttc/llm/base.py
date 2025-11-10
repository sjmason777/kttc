"""Base abstract class for LLM providers.

Defines the interface that all LLM providers must implement.
Supports both OpenAI and Anthropic APIs.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (OpenAI, Anthropic, etc.) must implement this interface.
    Supports both synchronous completion and streaming.
    """

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
        ...


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
