"""Unit tests for LLM providers.

Tests OpenAI and Anthropic provider implementations with mocking.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kttc.llm.base import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
)


@pytest.mark.unit
class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_openai_client: MagicMock) -> None:
        """Test successful completion."""
        # Arrange
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Translated text"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_openai_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")
            result = await provider.complete("Test prompt")

        assert result == "Translated text"

    @pytest.mark.asyncio
    async def test_complete_authentication_error(self, mock_openai_client: MagicMock) -> None:
        """Test authentication error handling."""
        import openai

        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body={"error": "Invalid API key"},
            )
        )

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_openai_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="invalid-key")

            with pytest.raises(LLMAuthenticationError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_rate_limit_error(self, mock_openai_client: MagicMock) -> None:
        """Test rate limit error handling."""
        import openai

        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": "Rate limit"},
            )
        )

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_openai_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")

            with pytest.raises(LLMRateLimitError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_empty_response(self, mock_openai_client: MagicMock) -> None:
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.usage = None

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_openai_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")

            with pytest.raises(LLMError, match="empty response"):
                await provider.complete("Test prompt")


@pytest.mark.unit
class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    @pytest.fixture
    def mock_anthropic_client(self) -> MagicMock:
        """Create a mock Anthropic client."""
        client = MagicMock()
        client.messages = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_anthropic_client: MagicMock) -> None:
        """Test successful completion."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Translated text"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = mock_anthropic_client

            from kttc.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")
            result = await provider.complete("Test prompt")

        assert result == "Translated text"

    @pytest.mark.asyncio
    async def test_complete_generic_error(self, mock_anthropic_client: MagicMock) -> None:
        """Test generic error handling."""
        # Use a generic exception
        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with patch("kttc.llm.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = mock_anthropic_client

            from kttc.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")

            with pytest.raises((LLMError, Exception)):
                await provider.complete("Test prompt")


@pytest.mark.unit
class TestLLMProviderUsage:
    """Test LLM provider usage tracking."""

    @pytest.mark.asyncio
    async def test_usage_tracking(self) -> None:
        """Test that usage is tracked correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")
            await provider.complete("Test")

            # Use the usage property, not get_usage method
            usage = provider.usage
            assert usage.input_tokens == 100
            assert usage.output_tokens == 50
            assert usage.call_count == 1
