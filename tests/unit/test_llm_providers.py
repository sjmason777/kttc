"""Unit tests for LLM provider implementations.

Tests all provider integrations (OpenAI, Anthropic, Yandex, GigaChat)
using mocks to avoid actual API calls.
"""

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from kttc.llm import (
    AnthropicProvider,
    GigaChatProvider,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    OpenAIProvider,
    YandexGPTProvider,
)

# Configure anyio to only use asyncio backend
pytestmark = pytest.mark.anyio


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    async def test_complete_success(self) -> None:
        """Test successful completion from OpenAI."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")

        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hola"))]

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            result = await provider.complete("Translate: Hello")
            assert result == "Hola"

    async def test_complete_empty_response(self) -> None:
        """Test handling of empty response from OpenAI."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            with pytest.raises(LLMError, match="empty response"):
                await provider.complete("Test prompt")

    async def test_stream_success(self) -> None:
        """Test successful streaming from OpenAI."""
        provider = OpenAIProvider(api_key="test-key")

        # Mock streaming chunks
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Ho"))]),
            Mock(choices=[Mock(delta=Mock(content="la"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),  # End chunk
        ]

        async def mock_stream() -> AsyncIterator[Any]:
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_stream())
        ):
            result = []
            async for chunk in provider.stream("Translate: Hello"):
                result.append(chunk)
            assert result == ["Ho", "la"]

    async def test_authentication_error(self) -> None:
        """Test handling of authentication errors."""
        provider = OpenAIProvider(api_key="invalid-key")

        # Mock the error properly with required parameters
        mock_response = Mock()
        mock_response.status_code = 401
        error = __import__("openai").AuthenticationError(
            "Invalid API key", response=mock_response, body={}
        )

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(LLMAuthenticationError):
                await provider.complete("Test")

    async def test_rate_limit_error(self) -> None:
        """Test handling of rate limit errors."""
        provider = OpenAIProvider(api_key="test-key")

        # Mock the error properly with required parameters
        mock_response = Mock()
        mock_response.status_code = 429
        error = __import__("openai").RateLimitError("Rate limited", response=mock_response, body={})

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(LLMRateLimitError):
                await provider.complete("Test")

    async def test_timeout_error(self) -> None:
        """Test handling of timeout errors."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=__import__("openai").APITimeoutError("Timeout")),
        ):
            with pytest.raises(LLMTimeoutError):
                await provider.complete("Test")


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    async def test_complete_success(self) -> None:
        """Test successful completion from Claude."""
        provider = AnthropicProvider(api_key="test-key", model="claude-3-5-sonnet-20241022")

        # Mock the Anthropic client
        mock_response = Mock()
        mock_response.content = [Mock(text="Hola")]

        with patch.object(
            provider.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            result = await provider.complete("Translate: Hello")
            assert result == "Hola"

    async def test_complete_empty_content(self) -> None:
        """Test handling of empty content from Claude."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = Mock()
        mock_response.content = []

        with patch.object(
            provider.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            with pytest.raises(LLMError, match="empty response"):
                await provider.complete("Test")

    async def test_stream_success(self) -> None:
        """Test successful streaming from Claude."""
        provider = AnthropicProvider(api_key="test-key")

        # Mock streaming
        class MockStream:
            async def __aenter__(self) -> "MockStream":
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def __aiter__(self) -> AsyncIterator[str]:
                for text in ["Ho", "la"]:
                    yield text

            @property
            async def text_stream(self) -> AsyncIterator[str]:
                async for text in self:
                    yield text

        mock_stream = MockStream()

        # Create a proper async context manager mock
        class AsyncContextManagerMock:
            async def __aenter__(self) -> MockStream:
                return mock_stream

            async def __aexit__(self, *args: Any) -> None:
                pass

        with patch.object(
            provider.client.messages, "stream", return_value=AsyncContextManagerMock()
        ):
            result = []
            async for chunk in provider.stream("Translate: Hello"):
                result.append(chunk)
            assert result == ["Ho", "la"]

    async def test_authentication_error(self) -> None:
        """Test handling of authentication errors."""
        provider = AnthropicProvider(api_key="invalid-key")

        # Mock the error properly with required parameters
        mock_response = Mock()
        mock_response.status_code = 401
        error = __import__("anthropic").AuthenticationError(
            "Invalid API key", response=mock_response, body={}
        )

        with patch.object(
            provider.client.messages,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(LLMAuthenticationError):
                await provider.complete("Test")


class TestYandexGPTProvider:
    """Test Yandex GPT provider implementation."""

    async def test_complete_success(self) -> None:
        """Test successful completion from Yandex GPT."""
        provider = YandexGPTProvider(
            api_key="test-key", folder_id="test-folder", model="yandexgpt/latest"
        )

        mock_response_data = {"result": {"alternatives": [{"message": {"text": "Hello"}}]}}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await provider.complete("Translate: Hello")
            assert result == "Hello"

    async def test_authentication_error(self) -> None:
        """Test handling of authentication errors."""
        provider = YandexGPTProvider(api_key="invalid-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 401

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMAuthenticationError):
                await provider.complete("Test")

    async def test_rate_limit_error(self) -> None:
        """Test handling of rate limit errors."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 429

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMRateLimitError):
                await provider.complete("Test")

    async def test_unexpected_response_format(self) -> None:
        """Test handling of unexpected response format."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"invalid": "format"})

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMError, match="Unexpected Yandex response"):
                await provider.complete("Test")

    async def test_stream_success(self) -> None:
        """Test successful streaming from Yandex GPT."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        # Mock streaming response
        mock_chunks = [
            json.dumps({"result": {"alternatives": [{"message": {"text": "Hel"}}]}}),
            json.dumps({"result": {"alternatives": [{"message": {"text": "lo"}}]}}),
        ]

        mock_response = AsyncMock()
        mock_response.status = 200

        class MockContent:
            def __init__(self, chunks: list[str]) -> None:
                self.chunks = chunks
                self.index = 0

            def __aiter__(self) -> "MockContent":
                return self

            async def __anext__(self) -> bytes:
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index].encode()
                self.index += 1
                return chunk

        mock_response.content = MockContent(mock_chunks)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = []
            async for chunk in provider.stream("Translate: Hello"):
                result.append(chunk)
            assert result == ["Hel", "lo"]


class TestGigaChatProvider:
    """Test Sber GigaChat provider implementation."""

    async def test_get_access_token_success(self) -> None:
        """Test successful OAuth token retrieval."""
        provider = GigaChatProvider(
            client_id="test-id", client_secret="test-secret", scope="GIGACHAT_API_PERS"
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"access_token": "test-token-123"})

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            token = await provider._get_access_token()
            assert token == "test-token-123"

    async def test_get_access_token_failure(self) -> None:
        """Test OAuth authentication failure."""
        provider = GigaChatProvider(client_id="invalid-id", client_secret="invalid-secret")

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Invalid credentials")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMAuthenticationError, match="authentication failed"):
                await provider._get_access_token()

    async def test_complete_success(self) -> None:
        """Test successful completion from GigaChat."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "cached-token"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello"}}]}
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await provider.complete("Translate: Hello")
            assert result == "Hello"

    async def test_complete_token_refresh(self) -> None:
        """Test automatic token refresh on 401."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "expired-token"

        # First response: 401 (token expired)
        mock_401_response = AsyncMock()
        mock_401_response.status = 401

        # Token refresh response
        mock_auth_response = AsyncMock()
        mock_auth_response.status = 200
        mock_auth_response.json = AsyncMock(return_value={"access_token": "new-token"})

        # Retry response: success
        mock_success_response = AsyncMock()
        mock_success_response.status = 200
        mock_success_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello"}}]}
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Setup multiple responses
            mock_post.return_value.__aenter__.side_effect = [
                mock_401_response,  # First attempt fails
                mock_auth_response,  # Get new token
                mock_success_response,  # Retry succeeds
            ]

            result = await provider.complete("Translate: Hello")
            assert result == "Hello"
            assert provider._access_token == "new-token"

    async def test_stream_success(self) -> None:
        """Test successful streaming from GigaChat."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        # Mock SSE streaming (use ASCII to avoid byte string issues)
        mock_chunks = [
            b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n',
            b'data: {"choices": [{"delta": {"content": "lo"}}]}\n',
            b"data: [DONE]\n",
        ]

        mock_response = AsyncMock()
        mock_response.status = 200

        class MockContent:
            def __init__(self, chunks: list[bytes]) -> None:
                self.chunks = chunks
                self.index = 0

            def __aiter__(self) -> "MockContent":
                return self

            async def __anext__(self) -> bytes:
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response.content = MockContent(mock_chunks)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            result = []
            async for chunk in provider.stream("Translate: Hello"):
                result.append(chunk)
            assert result == ["Hel", "lo"]

    async def test_timeout_error(self) -> None:
        """Test handling of timeout errors."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Connection timeout occurred")
            with pytest.raises(LLMTimeoutError, match="timed out"):
                await provider.complete("Test")


class TestAdditionalErrorPaths:
    """Additional tests for error handling and edge cases to achieve 100% coverage."""

    async def test_openai_stream_authentication_error(self) -> None:
        """Test authentication error in streaming."""
        provider = OpenAIProvider(api_key="invalid-key")

        mock_response = Mock()
        mock_response.status_code = 401
        error = __import__("openai").AuthenticationError(
            "Invalid API key", response=mock_response, body={}
        )

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(LLMAuthenticationError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_openai_stream_rate_limit_error(self) -> None:
        """Test rate limit error in streaming."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 429
        error = __import__("openai").RateLimitError("Rate limited", response=mock_response, body={})

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(LLMRateLimitError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_openai_stream_timeout_error(self) -> None:
        """Test timeout error in streaming."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=__import__("openai").APITimeoutError("Timeout")),
        ):
            with pytest.raises(LLMTimeoutError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_anthropic_content_type_error(self) -> None:
        """Test unexpected content type in Anthropic response."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = Mock()
        mock_response.content = [Mock(spec=[])]  # No 'text' attribute

        with patch.object(
            provider.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            with pytest.raises(LLMError, match="Unexpected content type"):
                await provider.complete("Test")

    async def test_yandex_api_error(self) -> None:
        """Test general API error in Yandex provider."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMError, match="status 500"):
                await provider.complete("Test")

    async def test_yandex_stream_authentication_error(self) -> None:
        """Test authentication error in Yandex streaming."""
        provider = YandexGPTProvider(api_key="invalid-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 401

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMAuthenticationError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_yandex_stream_rate_limit_error(self) -> None:
        """Test rate limit error in Yandex streaming."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 429

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMRateLimitError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_yandex_stream_api_error(self) -> None:
        """Test general API error in Yandex streaming."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMError, match="status 500"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_gigachat_missing_access_token_in_response(self) -> None:
        """Test GigaChat OAuth when access_token is missing."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"no_token": "here"})

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMAuthenticationError, match="No access token"):
                await provider._get_access_token()

    async def test_gigachat_unexpected_response_format(self) -> None:
        """Test GigaChat with unexpected response format."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"invalid": "format"})

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMError, match="Unexpected GigaChat response"):
                await provider.complete("Test")

    async def test_gigachat_stream_authentication_error(self) -> None:
        """Test authentication error in GigaChat streaming."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "expired-token"

        mock_response = AsyncMock()
        mock_response.status = 401

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMAuthenticationError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_gigachat_stream_rate_limit_error(self) -> None:
        """Test rate limit error in GigaChat streaming."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        mock_response = AsyncMock()
        mock_response.status = 429

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMRateLimitError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_gigachat_stream_api_error(self) -> None:
        """Test general API error in GigaChat streaming."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMError, match="status 500"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_openai_complete_api_error(self) -> None:
        """Test general API error in OpenAI complete."""
        provider = OpenAIProvider(api_key="test-key")

        mock_request = Mock()
        error = __import__("openai").APIError("API error", request=mock_request, body={})

        with patch.object(provider.client.chat.completions, "create", side_effect=error):
            with pytest.raises(LLMError, match="OpenAI API error"):
                await provider.complete("Test")

    async def test_openai_stream_api_error(self) -> None:
        """Test general API error in OpenAI streaming."""
        provider = OpenAIProvider(api_key="test-key")

        mock_request = Mock()
        error = __import__("openai").APIError("API error", request=mock_request, body={})

        with patch.object(provider.client.chat.completions, "create", side_effect=error):
            with pytest.raises(LLMError, match="OpenAI API error"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_anthropic_complete_rate_limit_error(self) -> None:
        """Test rate limit error in Anthropic complete."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 429
        error = __import__("anthropic").RateLimitError(
            "Rate limited", response=mock_response, body={}
        )

        with patch.object(provider.client.messages, "create", side_effect=error):
            with pytest.raises(LLMRateLimitError, match="Anthropic rate limit"):
                await provider.complete("Test")

    async def test_anthropic_complete_timeout_error(self) -> None:
        """Test timeout error in Anthropic complete."""
        provider = AnthropicProvider(api_key="test-key")

        mock_request = Mock()
        error = __import__("anthropic").APITimeoutError(request=mock_request)

        with patch.object(provider.client.messages, "create", side_effect=error):
            with pytest.raises(LLMTimeoutError, match="Anthropic request timed out"):
                await provider.complete("Test")

    async def test_anthropic_complete_api_error(self) -> None:
        """Test general API error in Anthropic complete."""
        provider = AnthropicProvider(api_key="test-key")

        mock_request = Mock()
        error = __import__("anthropic").APIError("API error", request=mock_request, body={})

        with patch.object(provider.client.messages, "create", side_effect=error):
            with pytest.raises(LLMError, match="Anthropic API error"):
                await provider.complete("Test")

    async def test_anthropic_stream_rate_limit_error(self) -> None:
        """Test rate limit error in Anthropic streaming."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 429
        error = __import__("anthropic").RateLimitError(
            "Rate limited", response=mock_response, body={}
        )

        mock_stream = AsyncMock()
        mock_stream.__aenter__.side_effect = error

        with patch.object(provider.client.messages, "stream", return_value=mock_stream):
            with pytest.raises(LLMRateLimitError, match="Anthropic rate limit"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_anthropic_stream_timeout_error(self) -> None:
        """Test timeout error in Anthropic streaming."""
        provider = AnthropicProvider(api_key="test-key")

        mock_request = Mock()
        error = __import__("anthropic").APITimeoutError(request=mock_request)

        mock_stream = AsyncMock()
        mock_stream.__aenter__.side_effect = error

        with patch.object(provider.client.messages, "stream", return_value=mock_stream):
            with pytest.raises(LLMTimeoutError, match="Anthropic request timed out"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_anthropic_stream_api_error(self) -> None:
        """Test general API error in Anthropic streaming."""
        provider = AnthropicProvider(api_key="test-key")

        mock_request = Mock()
        error = __import__("anthropic").APIError("API error", request=mock_request, body={})

        mock_stream = AsyncMock()
        mock_stream.__aenter__.side_effect = error

        with patch.object(provider.client.messages, "stream", return_value=mock_stream):
            with pytest.raises(LLMError, match="Anthropic API error"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_yandex_complete_timeout_error(self) -> None:
        """Test timeout error in Yandex complete."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Connection timeout occurred")
            with pytest.raises(LLMTimeoutError, match="Yandex request timed out"):
                await provider.complete("Test")

    async def test_yandex_complete_client_error(self) -> None:
        """Test general client error in Yandex complete."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Network error")
            with pytest.raises(LLMError, match="Yandex API error"):
                await provider.complete("Test")

    async def test_yandex_stream_timeout_error(self) -> None:
        """Test timeout error in Yandex streaming."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Connection timeout occurred")
            with pytest.raises(LLMTimeoutError, match="Yandex request timed out"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_yandex_stream_client_error(self) -> None:
        """Test general client error in Yandex streaming."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Network error")
            with pytest.raises(LLMError, match="Yandex API error"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_gigachat_auth_client_error(self) -> None:
        """Test client error during GigaChat authentication."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Network error")
            with pytest.raises(LLMAuthenticationError, match="GigaChat auth error"):
                await provider._get_access_token()

    async def test_gigachat_complete_timeout_error(self) -> None:
        """Test timeout error in GigaChat complete."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Connection timeout occurred")
            with pytest.raises(LLMTimeoutError, match="GigaChat request timed out"):
                await provider.complete("Test")

    async def test_gigachat_complete_client_error(self) -> None:
        """Test general client error in GigaChat complete."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Network error")
            with pytest.raises(LLMError, match="GigaChat API error"):
                await provider.complete("Test")

    async def test_gigachat_stream_timeout_error(self) -> None:
        """Test timeout error in GigaChat streaming."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Connection timeout occurred")
            with pytest.raises(LLMTimeoutError, match="GigaChat request timed out"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_gigachat_stream_client_error(self) -> None:
        """Test general client error in GigaChat streaming."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Network error")
            with pytest.raises(LLMError, match="GigaChat API error"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_anthropic_stream_authentication_error(self) -> None:
        """Test authentication error in Anthropic streaming."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 401
        error = __import__("anthropic").AuthenticationError(
            "Invalid key", response=mock_response, body={}
        )

        mock_stream = AsyncMock()
        mock_stream.__aenter__.side_effect = error

        with patch.object(provider.client.messages, "stream", return_value=mock_stream):
            with pytest.raises(LLMAuthenticationError):
                async for _ in provider.stream("Test"):
                    pass

    async def test_yandex_stream_json_decode_error(self) -> None:
        """Test JSON decode error handling in Yandex streaming."""
        provider = YandexGPTProvider(api_key="test-key", folder_id="test-folder")

        mock_response = AsyncMock()
        mock_response.status = 200
        # Return invalid JSON that will cause decode error
        mock_response.content = AsyncMock()
        mock_response.content.__aiter__.return_value = iter([b"invalid json", b""])

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            chunks = []
            async for chunk in provider.stream("Test"):
                chunks.append(chunk)
            # Should handle error gracefully and yield no chunks
            assert len(chunks) == 0

    async def test_gigachat_complete_no_initial_token(self) -> None:
        """Test GigaChat complete when no initial access token exists."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        # Ensure no token is set
        provider._access_token = None

        mock_auth_response = AsyncMock()
        mock_auth_response.status = 200
        mock_auth_response.json = AsyncMock(return_value={"access_token": "new-token"})

        mock_complete_response = AsyncMock()
        mock_complete_response.status = 200
        mock_complete_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "Test response"}}]}
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_auth_response
            # First call gets token, second call makes completion
            mock_post.return_value.__aenter__.side_effect = [
                mock_auth_response,
                mock_complete_response,
            ]
            result = await provider.complete("Test")
            assert result == "Test response"

    async def test_gigachat_complete_auth_fails_after_retry(self) -> None:
        """Test GigaChat complete when auth fails even after retry."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "expired-token"

        mock_401_response = AsyncMock()
        mock_401_response.status = 401

        mock_auth_response = AsyncMock()
        mock_auth_response.status = 200
        mock_auth_response.json = AsyncMock(return_value={"access_token": "new-token"})

        with patch("aiohttp.ClientSession.post") as mock_post:
            # First complete gets 401, get new token succeeds, retry also gets 401
            mock_post.return_value.__aenter__.side_effect = [
                mock_401_response,  # First complete attempt
                mock_auth_response,  # Get new token
                mock_401_response,  # Retry also fails with 401
            ]
            with pytest.raises(LLMAuthenticationError, match="authentication failed"):
                await provider.complete("Test")

    async def test_gigachat_complete_rate_limit(self) -> None:
        """Test GigaChat complete with rate limit error."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        mock_response = AsyncMock()
        mock_response.status = 429

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMRateLimitError, match="rate limit"):
                await provider.complete("Test")

    async def test_gigachat_complete_general_error(self) -> None:
        """Test GigaChat complete with general error status."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            with pytest.raises(LLMError, match="status 500"):
                await provider.complete("Test")

    async def test_gigachat_stream_no_initial_token(self) -> None:
        """Test GigaChat stream when no initial access token exists."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = None

        mock_auth_response = AsyncMock()
        mock_auth_response.status = 200
        mock_auth_response.json = AsyncMock(return_value={"access_token": "new-token"})

        mock_stream_response = AsyncMock()
        mock_stream_response.status = 200
        mock_stream_response.content = AsyncMock()
        chunk_data = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n'
        mock_stream_response.content.__aiter__.return_value = iter([chunk_data, b"data: [DONE]"])

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.side_effect = [
                mock_auth_response,
                mock_stream_response,
            ]
            chunks = []
            async for chunk in provider.stream("Test"):
                chunks.append(chunk)
            assert len(chunks) > 0

    async def test_gigachat_stream_auth_expired(self) -> None:
        """Test GigaChat stream when token is expired (401)."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "expired-token"

        mock_401_response = AsyncMock()
        mock_401_response.status = 401

        mock_auth_response = AsyncMock()
        mock_auth_response.status = 200
        mock_auth_response.json = AsyncMock(return_value={"access_token": "new-token"})

        with patch("aiohttp.ClientSession.post") as mock_post:
            # First call returns 401, second call gets new token, third raises error
            mock_post.return_value.__aenter__.side_effect = [
                mock_401_response,  # Stream returns 401
                mock_auth_response,  # Get new token succeeds
            ]
            with pytest.raises(LLMAuthenticationError, match="token expired"):
                async for _ in provider.stream("Test"):
                    pass

    async def test_gigachat_stream_json_decode_error(self) -> None:
        """Test GigaChat stream with invalid JSON."""
        provider = GigaChatProvider(client_id="test-id", client_secret="test-secret")
        provider._access_token = "test-token"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = AsyncMock()
        # Return invalid JSON
        mock_response.content.__aiter__.return_value = iter([b"data: invalid json\n", b""])

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            chunks = []
            async for chunk in provider.stream("Test"):
                chunks.append(chunk)
            # Should handle error gracefully
            assert len(chunks) == 0
