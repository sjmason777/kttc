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

"""Sber GigaChat LLM provider implementation.

Implements the BaseLLMProvider interface for Sber GigaChat API.
Supports various GigaChat models with OAuth 2.0 authentication.

Documentation: https://developers.sber.ru/docs/ru/gigachat/api/
"""

from __future__ import annotations

import base64
import uuid
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


class GigaChatProvider(BaseLLMProvider):
    """Sber GigaChat LLM provider implementation.

    Uses OAuth 2.0 authentication (access token valid for 30 minutes).
    Supports different API access levels (PERS, B2B, CORP).

    Example:
        >>> provider = GigaChatProvider(
        ...     client_id="your-client-id",
        ...     client_secret="your-client-secret",
        ...     scope="GIGACHAT_API_PERS"  # or B2B, CORP
        ... )
        >>> response = await provider.complete("Write a short greeting")
    """

    BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
    AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
        timeout: float = 30.0,
    ):
        """Initialize GigaChat provider.

        Args:
            client_id: Client ID from Sber Developer portal
            client_secret: Client secret from Sber Developer portal
            scope: API scope (GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP)
            model: Model name (GigaChat, GigaChat-Pro, etc.)
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._access_token: str | None = None

    async def _get_access_token(self) -> str:
        """Get OAuth 2.0 access token (valid for 30 minutes).

        Returns:
            Access token string

        Raises:
            LLMAuthenticationError: If authentication fails
        """
        # Return cached token if available
        if self._access_token:
            return self._access_token

        # Create RqUID (unique request ID)
        rq_uid = str(uuid.uuid4())

        # Create authorization header with base64 encoded credentials
        credentials = f"{self.client_id}:{self.client_secret}"
        credentials_base64 = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {credentials_base64}",
            "RqUID": rq_uid,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"scope": self.scope}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(
                self.AUTH_URL, headers=headers, data=data, ssl=False
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAuthenticationError(
                        f"GigaChat authentication failed: {error_text}"
                    )

                result = await response.json()
                if "access_token" not in result:
                    raise LLMAuthenticationError(f"No access token in response: {result}")

                token: str = result["access_token"]
                self._access_token = token  # Cache the token
                return token

        except aiohttp.ClientError as e:
            raise LLMAuthenticationError(f"GigaChat auth error: {e}") from e

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a single completion from GigaChat.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional GigaChat parameters

        Returns:
            The generated text response

        Raises:
            LLMAuthenticationError: If authentication fails
            LLMRateLimitError: If rate limit is exceeded
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        # Get access token (will be cached, token valid for 30 min)
        if not self._access_token:
            self._access_token = await self._get_access_token()

        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(url, headers=headers, json=payload, ssl=False) as response:
                if response.status == 401:
                    # Token expired, get new one and retry
                    self._access_token = await self._get_access_token()
                    headers["Authorization"] = f"Bearer {self._access_token}"

                    async with session.post(
                        url, headers=headers, json=payload, ssl=False
                    ) as retry_response:
                        if retry_response.status == 401:
                            raise LLMAuthenticationError("GigaChat authentication failed")
                        response = retry_response

                if response.status == 429:
                    raise LLMRateLimitError("GigaChat rate limit exceeded")
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(
                        f"GigaChat API error (status {response.status}): {error_text}"
                    )

                result = await response.json()

                # Extract text from response
                if "choices" in result and result["choices"]:
                    message = result["choices"][0].get("message", {})
                    content: str = message.get("content", "")
                    return content

                raise LLMError(f"Unexpected GigaChat response format: {result}")

        except aiohttp.ClientError as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"GigaChat request timed out: {e}") from e
            raise LLMError(f"GigaChat API error: {e}") from e

    async def _handle_stream_response_status(self, response: Any) -> None:
        """Handle streaming response status codes."""
        if response.status == 401:
            self._access_token = await self._get_access_token()
            raise LLMAuthenticationError("GigaChat token expired, retry")
        if response.status == 429:
            raise LLMRateLimitError("GigaChat rate limit exceeded")
        if response.status != 200:
            error_text = await response.text()
            raise LLMError(f"GigaChat API error (status {response.status}): {error_text}")

    def _parse_sse_content(self, line_text: str) -> str | None:
        """Parse SSE line and extract content."""
        import json

        if not line_text.startswith("data: "):
            return None
        data_str = line_text[6:]
        if data_str == "[DONE]":
            return None
        try:
            chunk = json.loads(data_str)
            if "choices" in chunk and chunk["choices"]:
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                return str(content) if content else None
        except json.JSONDecodeError:
            # Silently ignore JSON parsing errors and return None
            pass
        return None

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion from GigaChat."""
        if not self._access_token:
            self._access_token = await self._get_access_token()

        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(url, headers=headers, json=payload, ssl=False) as response:
                await self._handle_stream_response_status(response)
                async for line in response.content:
                    if line:
                        content = self._parse_sse_content(line.decode("utf-8").strip())
                        if content:
                            yield content
        except aiohttp.ClientError as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"GigaChat request timed out: {e}") from e
            raise LLMError(f"GigaChat API error: {e}") from e
