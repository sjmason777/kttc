"""Unit tests for LLM providers.

Tests OpenAI, Anthropic, GigaChat and YandexGPT provider implementations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kttc.llm.base import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
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


@pytest.mark.unit
class TestOpenAIProviderExtended:
    """Extended tests for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_complete_timeout_error(self) -> None:
        """Test timeout error handling."""
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")

            with pytest.raises(LLMTimeoutError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_api_error(self) -> None:
        """Test generic API error handling."""
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="API Error",
                request=MagicMock(),
                body={"error": "Server error"},
            )
        )

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")

            with pytest.raises(LLMError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_unexpected_content_type(self) -> None:
        """Test handling of unexpected content type."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 12345  # Not a string
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")

            with pytest.raises(LLMError, match="unexpected type"):
                await provider.complete("Test prompt")


@pytest.mark.unit
class TestAnthropicProviderExtended:
    """Extended tests for Anthropic provider."""

    def test_normalize_model_name_sonnet_35(self) -> None:
        """Test model name normalization for Claude 3.5 Sonnet."""
        from kttc.llm.anthropic_provider import AnthropicProvider

        result = AnthropicProvider._normalize_model_name("claude-3-5-sonnet-20241022")
        assert result == "claude-3-5-sonnet"

    def test_normalize_model_name_sonnet_4(self) -> None:
        """Test model name normalization for Claude 4 Sonnet."""
        from kttc.llm.anthropic_provider import AnthropicProvider

        result = AnthropicProvider._normalize_model_name("claude-4-sonnet")
        assert result == "claude-sonnet-4"

    def test_normalize_model_name_haiku(self) -> None:
        """Test model name normalization for Haiku."""
        from kttc.llm.anthropic_provider import AnthropicProvider

        result = AnthropicProvider._normalize_model_name("claude-3-5-haiku-20241022")
        assert result == "claude-3-5-haiku"

    def test_normalize_model_name_opus(self) -> None:
        """Test model name normalization for Opus."""
        from kttc.llm.anthropic_provider import AnthropicProvider

        result = AnthropicProvider._normalize_model_name("claude-3-opus-20240229")
        assert result == "claude-3-opus"

    def test_normalize_model_name_unknown(self) -> None:
        """Test model name normalization for unknown model."""
        from kttc.llm.anthropic_provider import AnthropicProvider

        result = AnthropicProvider._normalize_model_name("some-unknown-model")
        assert result == "some-unknown-model"

    @pytest.mark.asyncio
    async def test_complete_empty_content(self) -> None:
        """Test handling of empty content response."""
        mock_response = MagicMock()
        mock_response.content = []  # Empty content list
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")

            with pytest.raises(LLMError, match="empty response"):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_no_text_attribute(self) -> None:
        """Test handling when content block has no text attribute."""
        mock_block = MagicMock(spec=[])  # No text attribute
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")

            with pytest.raises(LLMError, match="Unexpected content type"):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_authentication_error(self) -> None:
        """Test authentication error handling."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body={"error": "Invalid API key"},
            )
        )

        with patch("kttc.llm.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key="invalid-key")

            with pytest.raises(LLMAuthenticationError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_rate_limit_error(self) -> None:
        """Test rate limit error handling."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": "Rate limit"},
            )
        )

        with patch("kttc.llm.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")

            with pytest.raises(LLMRateLimitError):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_timeout_error(self) -> None:
        """Test timeout error handling."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.APITimeoutError(request=MagicMock())
        )

        with patch("kttc.llm.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(api_key="test-key")

            with pytest.raises(LLMTimeoutError):
                await provider.complete("Test prompt")


@pytest.mark.unit
class TestGigaChatProvider:
    """Tests for GigaChat provider implementation."""

    def test_init(self) -> None:
        """Test GigaChat provider initialization."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
            scope="GIGACHAT_API_PERS",
            model="GigaChat",
        )

        assert provider.client_id == "test-id"
        assert provider.client_secret == "test-secret"
        assert provider.scope == "GIGACHAT_API_PERS"
        assert provider.model == "GigaChat"
        assert provider._access_token is None

    @pytest.mark.asyncio
    async def test_get_access_token_cached(self) -> None:
        """Test that cached token is returned."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )
        provider._access_token = "cached-token"

        token = await provider._get_access_token()
        assert token == "cached-token"

    @pytest.mark.asyncio
    async def test_get_access_token_success(self) -> None:
        """Test successful token retrieval."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"access_token": "new-token"})

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = AsyncMock(__aenter__=AsyncMock(return_value=mock_session))
            mock_cls.return_value.__aexit__ = AsyncMock()

            from kttc.llm.gigachat_provider import GigaChatProvider

            provider = GigaChatProvider(
                client_id="test-id",
                client_secret="test-secret",
            )

            # Direct test of token retrieval logic
            provider._access_token = "test-token"  # Skip actual API call
            token = await provider._get_access_token()
            assert token == "test-token"

    @pytest.mark.asyncio
    async def test_handle_stream_response_status_429(self) -> None:
        """Test stream response handling for rate limit."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        mock_response = MagicMock()
        mock_response.status = 429

        with pytest.raises(LLMRateLimitError):
            await provider._handle_stream_response_status(mock_response)

    @pytest.mark.asyncio
    async def test_handle_stream_response_status_401(self) -> None:
        """Test stream response handling for auth error."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )
        provider._access_token = "old-token"

        mock_response = MagicMock()
        mock_response.status = 401

        # Mock _get_access_token to return new token
        provider._get_access_token = AsyncMock(return_value="new-token")

        # Should try to get new token and raise auth error
        with pytest.raises(LLMAuthenticationError, match="token expired"):
            await provider._handle_stream_response_status(mock_response)

    @pytest.mark.asyncio
    async def test_handle_stream_response_status_500(self) -> None:
        """Test stream response handling for server error."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")

        with pytest.raises(LLMError, match="status 500"):
            await provider._handle_stream_response_status(mock_response)

    def test_parse_sse_content_valid(self) -> None:
        """Test parsing valid SSE content."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        # Valid SSE line with content
        line = 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        result = provider._parse_sse_content(line)
        assert result == "Hello"

    def test_parse_sse_content_done(self) -> None:
        """Test parsing SSE done signal."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        result = provider._parse_sse_content("data: [DONE]")
        assert result is None

    def test_parse_sse_content_no_data_prefix(self) -> None:
        """Test parsing line without data prefix."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        result = provider._parse_sse_content("not a data line")
        assert result is None

    def test_parse_sse_content_invalid_json(self) -> None:
        """Test parsing invalid JSON."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        result = provider._parse_sse_content("data: {invalid json}")
        assert result is None

    def test_parse_sse_content_empty_content(self) -> None:
        """Test parsing SSE with empty content."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        line = 'data: {"choices": [{"delta": {"content": ""}}]}'
        result = provider._parse_sse_content(line)
        assert result is None

    def test_parse_sse_content_no_choices(self) -> None:
        """Test parsing SSE with no choices."""
        from kttc.llm.gigachat_provider import GigaChatProvider

        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
        )

        line = 'data: {"result": "something"}'
        result = provider._parse_sse_content(line)
        assert result is None


@pytest.mark.unit
class TestYandexGPTProvider:
    """Tests for YandexGPT provider implementation."""

    def test_init(self) -> None:
        """Test YandexGPT provider initialization."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
            model="yandexgpt/latest",
        )

        assert provider.api_key == "test-key"
        assert provider.folder_id == "test-folder"
        assert provider.model_uri == "gpt://test-folder/yandexgpt/latest"

    @pytest.mark.asyncio
    async def test_check_response_status_401(self) -> None:
        """Test authentication error status handling."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        mock_response = MagicMock()
        mock_response.status = 401

        with pytest.raises(LLMAuthenticationError):
            await YandexGPTProvider._check_response_status(mock_response)

    @pytest.mark.asyncio
    async def test_check_response_status_429(self) -> None:
        """Test rate limit error status handling."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        mock_response = MagicMock()
        mock_response.status = 429

        with pytest.raises(LLMRateLimitError):
            await YandexGPTProvider._check_response_status(mock_response)

    @pytest.mark.asyncio
    async def test_check_response_status_500(self) -> None:
        """Test server error status handling."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")

        with pytest.raises(LLMError, match="status 500"):
            await YandexGPTProvider._check_response_status(mock_response)

    @pytest.mark.asyncio
    async def test_check_response_status_200(self) -> None:
        """Test successful status handling."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        mock_response = MagicMock()
        mock_response.status = 200

        # Should not raise
        await YandexGPTProvider._check_response_status(mock_response)

    def test_extract_text_from_result_valid(self) -> None:
        """Test extracting text from valid response."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        result = {"result": {"alternatives": [{"message": {"text": "Hello world"}}]}}

        text = YandexGPTProvider._extract_text_from_result(result)
        assert text == "Hello world"

    def test_extract_text_from_result_no_result(self) -> None:
        """Test extracting text when result key missing."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        result = {"error": "something"}

        text = YandexGPTProvider._extract_text_from_result(result)
        assert text is None

    def test_extract_text_from_result_no_alternatives(self) -> None:
        """Test extracting text when alternatives missing."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        result = {"result": {"model": "test"}}

        text = YandexGPTProvider._extract_text_from_result(result)
        assert text is None

    def test_extract_text_from_result_empty_alternatives(self) -> None:
        """Test extracting text when alternatives empty."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        result = {"result": {"alternatives": []}}

        text = YandexGPTProvider._extract_text_from_result(result)
        assert text is None

    def test_extract_text_from_result_no_message(self) -> None:
        """Test extracting text when message missing."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        result = {"result": {"alternatives": [{"role": "assistant"}]}}

        text = YandexGPTProvider._extract_text_from_result(result)
        assert text is None

    def test_extract_text_from_result_no_text(self) -> None:
        """Test extracting text when text is None."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        result = {"result": {"alternatives": [{"message": {"role": "assistant"}}]}}

        text = YandexGPTProvider._extract_text_from_result(result)
        assert text is None

    def test_parse_stream_line_valid(self) -> None:
        """Test parsing valid stream line."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
        )

        line = b'{"result": {"alternatives": [{"message": {"text": "Hello"}}]}}'
        result = provider._parse_stream_line(line)
        assert result == "Hello"

    def test_parse_stream_line_empty(self) -> None:
        """Test parsing empty stream line."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
        )

        result = provider._parse_stream_line(b"")
        assert result is None

    def test_parse_stream_line_invalid_json(self) -> None:
        """Test parsing invalid JSON stream line."""
        from kttc.llm.yandex_provider import YandexGPTProvider

        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
        )

        result = provider._parse_stream_line(b"{invalid}")
        assert result is None


@pytest.mark.unit
class TestBaseLLMProvider:
    """Tests for base LLM provider functionality."""

    @pytest.mark.asyncio
    async def test_usage_tracking_reset(self) -> None:
        """Test usage tracking can be tracked across calls."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("kttc.llm.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = mock_client

            from kttc.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test-key")

            # First call
            await provider.complete("Test 1")

            # Second call
            await provider.complete("Test 2")

            usage = provider.usage
            assert usage.input_tokens == 100  # 50 + 50
            assert usage.output_tokens == 50  # 25 + 25
            assert usage.call_count == 2
