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

"""Unit tests for Gemini LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kttc.llm import GeminiProvider
from kttc.llm.base import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
)


@pytest.mark.unit
class TestGeminiProviderInit:
    """Test GeminiProvider initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        provider = GeminiProvider(api_key="test-api-key")
        assert provider.api_key == "test-api-key"
        assert provider.model == "gemini-2.0-flash"
        assert provider.timeout == 60.0

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-pro")
        assert provider.model == "gemini-1.5-pro"

    def test_init_with_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        provider = GeminiProvider(api_key="test-key", timeout=120.0)
        assert provider.timeout == 120.0


@pytest.mark.unit
class TestGeminiProviderComplete:
    """Test GeminiProvider complete method."""

    @pytest.mark.asyncio
    async def test_complete_success(self) -> None:
        """Test successful completion."""
        provider = GeminiProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "candidates": [{"content": {"parts": [{"text": "Hello translated"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                },
            }
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch.object(provider, "_get_session", return_value=mock_session):
            result = await provider.complete("Translate: Hello")

        assert result == "Hello translated"
        assert provider.usage.input_tokens == 10
        assert provider.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_auth_error(self) -> None:
        """Test authentication error handling."""
        provider = GeminiProvider(api_key="invalid-key")

        mock_response = MagicMock()
        mock_response.status = 401

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch.object(provider, "_get_session", return_value=mock_session):
            with pytest.raises(LLMAuthenticationError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self) -> None:
        """Test rate limit error handling."""
        provider = GeminiProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 429

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch.object(provider, "_get_session", return_value=mock_session):
            with pytest.raises(LLMRateLimitError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_blocked_response(self) -> None:
        """Test handling of blocked response."""
        provider = GeminiProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"},
            }
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch.object(provider, "_get_session", return_value=mock_session):
            with pytest.raises(LLMError, match="blocked"):
                await provider.complete("Test prompt")


@pytest.mark.unit
class TestGeminiProviderPricing:
    """Test GeminiProvider pricing configuration."""

    def test_pricing_per_1m_exists(self) -> None:
        """Test that pricing configuration exists."""
        assert "gemini-2.0-flash" in GeminiProvider.PRICING_PER_1M
        assert "gemini-1.5-pro" in GeminiProvider.PRICING_PER_1M

    def test_pricing_format(self) -> None:
        """Test pricing format is tuple of (input, output)."""
        for model, pricing in GeminiProvider.PRICING_PER_1M.items():
            assert isinstance(pricing, tuple)
            assert len(pricing) == 2
            assert pricing[0] >= 0  # input price
            assert pricing[1] >= 0  # output price


@pytest.mark.unit
class TestGeminiProviderRequestBody:
    """Test request body building."""

    def test_build_request_body(self) -> None:
        """Test request body structure."""
        provider = GeminiProvider(api_key="test-key")
        body = provider._build_request_body(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100,
        )

        assert "contents" in body
        assert "generationConfig" in body
        assert body["contents"][0]["parts"][0]["text"] == "Test prompt"
        assert body["generationConfig"]["temperature"] == 0.5
        assert body["generationConfig"]["maxOutputTokens"] == 100


@pytest.mark.unit
class TestGeminiProviderJsonParsing:
    """Test JSON parsing utilities."""

    def test_find_json_end_simple(self) -> None:
        """Test finding end of simple JSON object."""
        provider = GeminiProvider(api_key="test-key")
        assert provider._find_json_end('{"key": "value"}more') == 16

    def test_find_json_end_nested(self) -> None:
        """Test finding end of nested JSON object."""
        provider = GeminiProvider(api_key="test-key")
        assert provider._find_json_end('{"a": {"b": 1}}rest') == 15

    def test_find_json_end_with_strings(self) -> None:
        """Test finding end with strings containing braces."""
        provider = GeminiProvider(api_key="test-key")
        result = provider._find_json_end('{"text": "hello {world}"}next')
        assert result == 25

    def test_find_json_end_incomplete(self) -> None:
        """Test incomplete JSON returns -1."""
        provider = GeminiProvider(api_key="test-key")
        assert provider._find_json_end('{"key": "value"') == -1

    def test_find_json_end_not_json(self) -> None:
        """Test non-JSON returns -1."""
        provider = GeminiProvider(api_key="test-key")
        assert provider._find_json_end("not json") == -1
