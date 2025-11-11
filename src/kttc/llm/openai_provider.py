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

"""OpenAI LLM provider implementation.

Implements the BaseLLMProvider interface for OpenAI's API.
Supports GPT-4 and GPT-4 Turbo models.
"""

from collections.abc import AsyncGenerator
from typing import Any

import openai

from .base import (
    BaseLLMProvider,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation.

    Supports GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo models.
    Uses the official OpenAI Python SDK (v1.0+).

    Example:
        >>> provider = OpenAIProvider(
        ...     api_key="sk-...",
        ...     model="gpt-4-turbo"
        ... )
        >>> response = await provider.complete("Translate: Hello")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        timeout: float = 30.0,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo")
            timeout: Request timeout in seconds
        """
        self.client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a single completion from OpenAI.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters (top_p, frequency_penalty, etc.)

        Returns:
            The generated text response

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content
            if content is None:
                raise LLMError("OpenAI returned empty response")

            assert isinstance(content, str)
            return content

        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}") from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI request timed out: {e}") from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from OpenAI.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters

        Yields:
            Text chunks as they are generated

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}") from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI request timed out: {e}") from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e
