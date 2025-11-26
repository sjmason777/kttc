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

"""Yandex GPT (YandexGPT) LLM provider implementation.

Implements the BaseLLMProvider interface for Yandex Cloud Foundation Models API.
Supports YandexGPT Pro and YandexGPT Lite models.

Documentation: https://yandex.cloud/en/docs/foundation-models/
"""

from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from .base import (
    BaseLLMProvider,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class YandexGPTProvider(BaseLLMProvider):
    """Yandex GPT LLM provider implementation.

    Supports YandexGPT Pro (complex tasks, up to 32K tokens) and
    YandexGPT Lite (fast responses, up to 7.4K tokens).

    Example:
        >>> provider = YandexGPTProvider(
        ...     api_key="your-api-key",
        ...     folder_id="your-folder-id",
        ...     model="yandexgpt/latest"  # or "yandexgpt-lite/latest"
        ... )
        >>> response = await provider.complete("Translate: Hello")
    """

    BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        model: str = "yandexgpt/latest",
        timeout: float = 30.0,
    ):
        """Initialize Yandex GPT provider.

        Args:
            api_key: Yandex Cloud API key (set YC_API_KEY env var)
            folder_id: Yandex Cloud folder ID
            model: Model URI (yandexgpt/latest or yandexgpt-lite/latest)
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_key = api_key
        self.folder_id = folder_id
        self.model_uri = f"gpt://{folder_id}/{model}"
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a single completion from Yandex GPT.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (must be > 0 and <= 7400)
            **kwargs: Additional Yandex parameters

        Returns:
            The generated text response

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        url = f"{self.BASE_URL}/completion"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [{"role": "user", "text": prompt}],
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(url, headers=headers, json=payload) as response:
                if response.status == 401:
                    raise LLMAuthenticationError("Yandex API authentication failed")
                if response.status == 429:
                    raise LLMRateLimitError("Yandex API rate limit exceeded")
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(f"Yandex API error (status {response.status}): {error_text}")

                result = await response.json()

                # Extract text from response
                if "result" in result and "alternatives" in result["result"]:
                    alternatives = result["result"]["alternatives"]
                    if alternatives and "message" in alternatives[0]:
                        text: str = alternatives[0]["message"]["text"]
                        return text

                raise LLMError(f"Unexpected Yandex response format: {result}")

        except aiohttp.ClientError as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Yandex request timed out: {e}") from e
            raise LLMError(f"Yandex API error: {e}") from e

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from Yandex GPT.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Yandex parameters

        Yields:
            Text chunks as they are generated

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        url = f"{self.BASE_URL}/completion"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": True,  # Enable streaming
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [{"role": "user", "text": prompt}],
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(url, headers=headers, json=payload) as response:
                if response.status == 401:
                    raise LLMAuthenticationError("Yandex API authentication failed")
                if response.status == 429:
                    raise LLMRateLimitError("Yandex API rate limit exceeded")
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(f"Yandex API error (status {response.status}): {error_text}")

                # Stream response line by line
                async for line in response.content:
                    if line:
                        import json

                        try:
                            chunk = json.loads(line)
                            if "result" in chunk and "alternatives" in chunk["result"]:
                                alternatives = chunk["result"]["alternatives"]
                                if alternatives and "message" in alternatives[0]:
                                    text = alternatives[0]["message"]["text"]
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Yandex request timed out: {e}") from e
            raise LLMError(f"Yandex API error: {e}") from e
