"""Anthropic (Claude) LLM provider implementation.

Implements the BaseLLMProvider interface for Anthropic's Claude API.
Supports Claude 3.5 Sonnet, Claude 3 Opus, and other models.
"""

from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from .base import (
    BaseLLMProvider,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider implementation.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku.
    Uses the official Anthropic Python SDK.

    Example:
        >>> provider = AnthropicProvider(
        ...     api_key="sk-ant-...",
        ...     model="claude-3-5-sonnet-20241022"
        ... )
        >>> response = await provider.complete("Translate: Hello")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: float = 30.0,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            timeout: Request timeout in seconds
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a single completion from Claude.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters (top_p, top_k, etc.)

        Returns:
            The generated text response

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Claude returns content as a list of content blocks
            if not response.content:
                raise LLMError("Anthropic returned empty response")

            # Get first text block
            first_block = response.content[0]
            if hasattr(first_block, "text"):
                text: str = first_block.text
                return text
            else:
                raise LLMError(f"Unexpected content type: {type(first_block)}")

        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}") from e
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(f"Anthropic request timed out: {e}") from e
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from Claude.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters

        Yields:
            Text chunks as they are generated

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}") from e
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(f"Anthropic request timed out: {e}") from e
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e
