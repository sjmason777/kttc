"""Unit tests for GigaChat provider module.

Tests Sber GigaChat LLM provider with mocking.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kttc.llm.gigachat_provider import GigaChatProvider


@pytest.mark.unit
class TestGigaChatProvider:
    """Test GigaChatProvider functionality."""

    def test_provider_initialization(self) -> None:
        """Test provider initializes correctly."""
        provider = GigaChatProvider(
            client_id="test-client-id",
            client_secret="test-client-secret",
            scope="GIGACHAT_API_PERS",
        )

        assert provider.client_id == "test-client-id"
        assert provider.client_secret == "test-client-secret"
        assert provider.scope == "GIGACHAT_API_PERS"
        assert provider.model == "GigaChat"

    def test_provider_initialization_with_custom_model(self) -> None:
        """Test provider with custom model."""
        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
            model="GigaChat-Pro",
        )

        assert provider.model == "GigaChat-Pro"

    def test_provider_initialization_with_custom_timeout(self) -> None:
        """Test provider with custom timeout."""
        provider = GigaChatProvider(
            client_id="test-id",
            client_secret="test-secret",
            timeout=60.0,
        )

        assert provider.timeout.total == 60.0

    def test_provider_base_urls(self) -> None:
        """Test provider has correct base URLs."""
        assert "gigachat.devices.sberbank.ru" in GigaChatProvider.BASE_URL
        assert "ngw.devices.sberbank.ru" in GigaChatProvider.AUTH_URL


@pytest.mark.unit
class TestGigaChatProviderScopes:
    """Test GigaChat API scopes."""

    def test_pers_scope(self) -> None:
        """Test personal API scope."""
        provider = GigaChatProvider(
            client_id="id",
            client_secret="secret",
            scope="GIGACHAT_API_PERS",
        )
        assert provider.scope == "GIGACHAT_API_PERS"

    def test_b2b_scope(self) -> None:
        """Test B2B API scope."""
        provider = GigaChatProvider(
            client_id="id",
            client_secret="secret",
            scope="GIGACHAT_API_B2B",
        )
        assert provider.scope == "GIGACHAT_API_B2B"

    def test_corp_scope(self) -> None:
        """Test corporate API scope."""
        provider = GigaChatProvider(
            client_id="id",
            client_secret="secret",
            scope="GIGACHAT_API_CORP",
        )
        assert provider.scope == "GIGACHAT_API_CORP"


@pytest.mark.unit
class TestGigaChatProviderAsync:
    """Test GigaChatProvider async methods."""

    @pytest.fixture
    def provider(self) -> GigaChatProvider:
        """Create a provider instance."""
        return GigaChatProvider(
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

    @pytest.mark.asyncio
    async def test_complete_success(self, provider: GigaChatProvider) -> None:
        """Test successful completion."""
        mock_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch.object(provider, "_get_access_token", new_callable=AsyncMock) as mock_token:
            mock_token.return_value = "test-token"

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

    @pytest.mark.asyncio
    async def test_get_access_token_caching(self, provider: GigaChatProvider) -> None:
        """Test that access token is cached."""
        # Set cached token
        provider._access_token = "cached-token"

        token = await provider._get_access_token()
        assert token == "cached-token"


@pytest.mark.unit
class TestGigaChatProviderUsage:
    """Test GigaChatProvider usage tracking."""

    def test_usage_property_exists(self) -> None:
        """Test usage property exists."""
        provider = GigaChatProvider(
            client_id="id",
            client_secret="secret",
        )
        usage = provider.usage
        assert usage is not None

    def test_initial_usage_is_zero(self) -> None:
        """Test initial usage is zero."""
        provider = GigaChatProvider(
            client_id="id",
            client_secret="secret",
        )
        usage = provider.usage
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.call_count == 0
