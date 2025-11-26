"""Unit tests for Yandex GPT provider module.

Tests Yandex GPT LLM provider with mocking.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest

from kttc.llm.yandex_provider import YandexGPTProvider


@pytest.mark.unit
class TestYandexGPTProvider:
    """Test YandexGPTProvider functionality."""

    def test_provider_initialization(self) -> None:
        """Test provider initializes correctly."""
        provider = YandexGPTProvider(
            api_key="test-api-key",
            folder_id="test-folder-id",
            model="yandexgpt/latest",
        )

        assert provider.api_key == "test-api-key"
        assert provider.folder_id == "test-folder-id"
        assert "yandexgpt/latest" in provider.model_uri
        assert provider.model_uri == "gpt://test-folder-id/yandexgpt/latest"

    def test_provider_with_lite_model(self) -> None:
        """Test provider with YandexGPT Lite model."""
        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
            model="yandexgpt-lite/latest",
        )

        assert "yandexgpt-lite/latest" in provider.model_uri

    def test_provider_with_custom_timeout(self) -> None:
        """Test provider with custom timeout."""
        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
            timeout=60.0,
        )

        assert provider.timeout.total == 60.0

    def test_provider_base_url(self) -> None:
        """Test provider has correct base URL."""
        assert urlparse(YandexGPTProvider.BASE_URL).netloc == "llm.api.cloud.yandex.net"
        assert "foundationModels" in YandexGPTProvider.BASE_URL


@pytest.mark.unit
class TestYandexGPTProviderAsync:
    """Test YandexGPTProvider async methods."""

    @pytest.fixture
    def provider(self) -> YandexGPTProvider:
        """Create a provider instance."""
        return YandexGPTProvider(
            api_key="test-api-key",
            folder_id="test-folder-id",
        )

    @pytest.mark.asyncio
    async def test_complete_success(self, provider: YandexGPTProvider) -> None:
        """Test successful completion."""
        mock_response = {
            "result": {
                "alternatives": [{"message": {"text": "Test response"}}],
                "usage": {"inputTextTokens": "10", "completionTokens": "5"},
            }
        }

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=None)

            mock_session.post = MagicMock(return_value=mock_resp)
            mock_session_cls.return_value = mock_session

            result = await provider.complete("Test prompt")
            assert result == "Test response"


@pytest.mark.unit
class TestYandexGPTProviderUsage:
    """Test YandexGPTProvider usage tracking."""

    def test_usage_property_exists(self) -> None:
        """Test usage property exists."""
        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
        )
        usage = provider.usage
        assert usage is not None

    def test_initial_usage_is_zero(self) -> None:
        """Test initial usage is zero."""
        provider = YandexGPTProvider(
            api_key="test-key",
            folder_id="test-folder",
        )
        usage = provider.usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.call_count == 0


@pytest.mark.unit
class TestYandexGPTProviderModels:
    """Test Yandex GPT model configurations."""

    def test_pro_model_uri(self) -> None:
        """Test YandexGPT Pro model URI."""
        provider = YandexGPTProvider(
            api_key="key",
            folder_id="folder123",
            model="yandexgpt/latest",
        )
        assert provider.model_uri == "gpt://folder123/yandexgpt/latest"

    def test_lite_model_uri(self) -> None:
        """Test YandexGPT Lite model URI."""
        provider = YandexGPTProvider(
            api_key="key",
            folder_id="folder123",
            model="yandexgpt-lite/latest",
        )
        assert provider.model_uri == "gpt://folder123/yandexgpt-lite/latest"

    def test_custom_model_version(self) -> None:
        """Test custom model version."""
        provider = YandexGPTProvider(
            api_key="key",
            folder_id="folder123",
            model="yandexgpt/rc",
        )
        assert provider.model_uri == "gpt://folder123/yandexgpt/rc"
