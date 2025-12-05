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

"""Google Gemini LLM provider implementation.

Implements the BaseLLMProvider interface for Google's Gemini API.
Uses REST API via aiohttp for lightweight, async operation.

Supported models:
- gemini-2.0-flash (recommended, fast and capable)
- gemini-2.0-flash-lite (fastest, cost-effective)
- gemini-1.5-pro (most capable)
- gemini-1.5-flash (balanced)
"""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

# Gemini API base URL
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation.

    Uses Google's Generative Language REST API via aiohttp for
    lightweight async operation without heavy SDK dependencies.

    Example:
        >>> provider = GeminiProvider(
        ...     api_key="your-api-key",
        ...     model="gemini-2.0-flash"
        ... )
        >>> response = await provider.complete("Translate: Hello")
    """

    # Pricing per 1M tokens (input, output) - December 2025 rates
    PRICING_PER_1M: dict[str, tuple[float, float]] = {
        "gemini-2.0-flash": (0.10, 0.40),  # $0.10/$0.40 per 1M tokens
        "gemini-2.0-flash-lite": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-flash-8b": (0.0375, 0.15),
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout: float = 60.0,
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Google AI API key (from https://aistudio.google.com/)
            model: Model name (default: gemini-2.0-flash)
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._usage.model_name = model
        self._session: aiohttp.ClientSession | None = None

        # Update pricing in usage tracker
        if model in self.PRICING_PER_1M:
            input_per_1k = self.PRICING_PER_1M[model][0] / 1000
            output_per_1k = self.PRICING_PER_1M[model][1] / 1000
            self._usage.DEFAULT_PRICING[model] = (input_per_1k, output_per_1k)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_request_body(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build request body for Gemini API.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            stream: Whether to stream response

        Returns:
            Request body dictionary
        """
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40,
            },
        }

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a single completion from Gemini.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (ignored for now)

        Returns:
            The generated text response

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
        body = self._build_request_body(prompt, temperature, max_tokens)

        session = await self._get_session()

        try:
            async with session.post(url, json=body) as response:
                if response.status == 401:
                    raise LLMAuthenticationError("Gemini authentication failed: Invalid API key")
                if response.status == 429:
                    raise LLMRateLimitError("Gemini rate limit exceeded")
                if response.status >= 400:
                    error_text = await response.text()
                    raise LLMError(f"Gemini API error ({response.status}): {error_text}")

                data = await response.json()

                # Extract text from response
                try:
                    candidates = data.get("candidates", [])
                    if not candidates:
                        # Check for safety block
                        if "promptFeedback" in data:
                            feedback = data["promptFeedback"]
                            if feedback.get("blockReason"):
                                raise LLMError(
                                    f"Gemini blocked request: {feedback.get('blockReason')}"
                                )
                        raise LLMError("Gemini returned no candidates")

                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if not parts:
                        raise LLMError("Gemini returned empty content")

                    text: str = parts[0].get("text", "")
                    if not text:
                        raise LLMError("Gemini returned empty text")

                except (KeyError, IndexError) as e:
                    raise LLMError(f"Failed to parse Gemini response: {e}") from e

                # Track token usage
                usage_metadata = data.get("usageMetadata", {})
                input_tokens = usage_metadata.get("promptTokenCount", 0)
                output_tokens = usage_metadata.get("candidatesTokenCount", 0)
                if input_tokens or output_tokens:
                    self._usage.add_usage(input_tokens, output_tokens)

                return text

        except TimeoutError as e:
            raise LLMTimeoutError(f"Gemini request timed out: {e}") from e
        except aiohttp.ClientError as e:
            raise LLMError(f"Gemini connection error: {e}") from e

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from Gemini.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        url = f"{GEMINI_API_BASE}/models/{self.model}:streamGenerateContent"
        body = self._build_request_body(prompt, temperature, max_tokens, stream=True)

        session = await self._get_session()

        try:
            async with session.post(url, json=body) as response:
                if response.status == 401:
                    raise LLMAuthenticationError("Gemini authentication failed: Invalid API key")
                if response.status == 429:
                    raise LLMRateLimitError("Gemini rate limit exceeded")
                if response.status >= 400:
                    error_text = await response.text()
                    raise LLMError(f"Gemini API error ({response.status}): {error_text}")

                # Gemini streams JSON array items
                buffer = ""
                async for chunk in response.content.iter_any():
                    if chunk:
                        buffer += chunk.decode("utf-8")

                        # Try to parse complete JSON objects from buffer
                        # Gemini streams as JSON array: [{...}, {...}, ...]
                        while True:
                            # Look for complete JSON object
                            try:
                                # Remove leading [ or , if present
                                clean_buffer = buffer.lstrip("[,\n\r ")

                                # Find end of JSON object
                                if not clean_buffer.startswith("{"):
                                    break

                                # Parse JSON object
                                obj_end = self._find_json_end(clean_buffer)
                                if obj_end == -1:
                                    break

                                json_str = clean_buffer[:obj_end]
                                data = json.loads(json_str)
                                buffer = clean_buffer[obj_end:]

                                # Extract text from chunk
                                candidates = data.get("candidates", [])
                                if candidates:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])
                                    if parts:
                                        text = parts[0].get("text", "")
                                        if text:
                                            yield text

                            except json.JSONDecodeError:
                                break

        except TimeoutError as e:
            raise LLMTimeoutError(f"Gemini request timed out: {e}") from e
        except aiohttp.ClientError as e:
            raise LLMError(f"Gemini connection error: {e}") from e

    def _find_json_end(self, s: str) -> int:
        """Find the end of a JSON object in a string.

        Args:
            s: String starting with '{'

        Returns:
            Index after the closing '}', or -1 if not found
        """
        if not s.startswith("{"):
            return -1

        depth = 0
        in_string = False
        escape_next = False

        for i, c in enumerate(s):
            if escape_next:
                escape_next = False
                continue

            if c == "\\":
                escape_next = True
                continue

            if c == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1

        return -1

    def __del__(self) -> None:
        """Cleanup: close session if still open."""
        if self._session and not self._session.closed:
            # Can't await in __del__, just warn
            logger.warning(
                "GeminiProvider session not properly closed. Use 'await provider.close()'"
            )
