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

"""Strict tests for DeepL Machine Translation provider.

All tests use mocks - no actual API calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kttc.mt import DeepLProvider, MTError, MTQuotaExceededError, TranslationResult
from kttc.mt.deepl_provider import DEEPL_API_FREE, DEEPL_API_PRO


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = TranslationResult(
            text="Hallo Welt",
            source_lang="EN",
            target_lang="DE",
        )
        assert result.text == "Hallo Welt"
        assert result.source_lang == "EN"
        assert result.target_lang == "DE"
        assert result.characters == 0

    def test_with_characters(self) -> None:
        """Test result with character count."""
        result = TranslationResult(
            text="Привет мир",
            source_lang="EN",
            target_lang="RU",
            characters=11,
        )
        assert result.characters == 11


class TestDeepLProviderInit:
    """Tests for DeepLProvider initialization."""

    def test_init_free_api(self) -> None:
        """Test initialization with free API."""
        provider = DeepLProvider(api_key="test-key", use_free_api=True)
        assert provider.api_key == "test-key"
        assert provider.base_url == DEEPL_API_FREE

    def test_init_pro_api(self) -> None:
        """Test initialization with pro API."""
        provider = DeepLProvider(api_key="test-key", use_free_api=False)
        assert provider.base_url == DEEPL_API_PRO

    def test_init_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        provider = DeepLProvider(api_key="test-key", timeout=60.0)
        assert provider.timeout == 60.0

    def test_init_total_characters_zero(self) -> None:
        """Test that total_characters starts at zero."""
        provider = DeepLProvider(api_key="test-key")
        assert provider.total_characters == 0


class TestDeepLProviderLanguageNormalization:
    """Tests for language code normalization."""

    def test_normalize_lowercase(self) -> None:
        """Test normalizing lowercase codes."""
        provider = DeepLProvider(api_key="test")
        assert provider._normalize_language("en") == "EN"
        assert provider._normalize_language("de") == "DE"
        assert provider._normalize_language("ru") == "RU"

    def test_normalize_uppercase(self) -> None:
        """Test normalizing uppercase codes."""
        provider = DeepLProvider(api_key="test")
        assert provider._normalize_language("EN") == "EN"
        assert provider._normalize_language("DE") == "DE"

    def test_normalize_with_region(self) -> None:
        """Test normalizing codes with region suffix."""
        provider = DeepLProvider(api_key="test")
        assert provider._normalize_language("en-US") == "EN"
        assert provider._normalize_language("en-GB") == "EN"
        assert provider._normalize_language("pt-BR") == "PT"

    def test_normalize_with_underscore(self) -> None:
        """Test normalizing codes with underscore."""
        provider = DeepLProvider(api_key="test")
        assert provider._normalize_language("zh_CN") == "ZH"
        assert provider._normalize_language("en_US") == "EN"

    def test_normalize_norwegian(self) -> None:
        """Test Norwegian maps to NB (Bokmål)."""
        provider = DeepLProvider(api_key="test")
        assert provider._normalize_language("no") == "NB"
        assert provider._normalize_language("nb") == "NB"

    def test_normalize_unknown_language(self) -> None:
        """Test unknown language codes are uppercased."""
        provider = DeepLProvider(api_key="test")
        assert provider._normalize_language("xx") == "XX"


class TestDeepLProviderTranslate:
    """Tests for translate method with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_translate_success(self) -> None:
        """Test successful translation."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "translations": [{"text": "Hallo Welt", "detected_source_language": "EN"}]
            }
        )

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        result = await provider.translate("Hello World", target_lang="DE")

        assert result.text == "Hallo Welt"
        assert result.source_lang == "EN"
        assert result.target_lang == "DE"

    @pytest.mark.asyncio
    async def test_translate_with_source_lang(self) -> None:
        """Test translation with explicit source language."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"translations": [{"text": "Привет", "detected_source_language": "EN"}]}
        )

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        result = await provider.translate("Hello", target_lang="RU", source_lang="EN")

        assert result.text == "Привет"

    @pytest.mark.asyncio
    async def test_translate_auth_error(self) -> None:
        """Test handling of authentication error."""
        provider = DeepLProvider(api_key="invalid-key")

        mock_response = AsyncMock()
        mock_response.status = 403

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        with pytest.raises(MTError, match="authentication failed"):
            await provider.translate("Hello", target_lang="DE")

    @pytest.mark.asyncio
    async def test_translate_quota_exceeded(self) -> None:
        """Test handling of quota exceeded error."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 456

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        with pytest.raises(MTQuotaExceededError, match="quota exceeded"):
            await provider.translate("Hello", target_lang="DE")

    @pytest.mark.asyncio
    async def test_translate_api_error(self) -> None:
        """Test handling of generic API error."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        with pytest.raises(MTError, match="API error"):
            await provider.translate("Hello", target_lang="DE")


class TestDeepLProviderTranslateBatch:
    """Tests for translate_batch method."""

    @pytest.mark.asyncio
    async def test_translate_batch_empty(self) -> None:
        """Test batch translation with empty list."""
        provider = DeepLProvider(api_key="test-key")
        results = await provider.translate_batch([], target_lang="DE")
        assert results == []

    @pytest.mark.asyncio
    async def test_translate_batch_multiple(self) -> None:
        """Test batch translation with multiple texts."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "translations": [
                    {"text": "Hallo", "detected_source_language": "EN"},
                    {"text": "Welt", "detected_source_language": "EN"},
                    {"text": "Test", "detected_source_language": "EN"},
                ]
            }
        )

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        results = await provider.translate_batch(
            ["Hello", "World", "Test"],
            target_lang="DE",
        )

        assert len(results) == 3
        assert results[0].text == "Hallo"
        assert results[1].text == "Welt"
        assert results[2].text == "Test"

    @pytest.mark.asyncio
    async def test_translate_batch_tracks_characters(self) -> None:
        """Test that batch translation tracks total characters."""
        provider = DeepLProvider(api_key="test-key")
        initial_chars = provider.total_characters

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "translations": [
                    {"text": "Hallo", "detected_source_language": "EN"},
                ]
            }
        )

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        await provider.translate_batch(["Hello"], target_lang="DE")

        assert provider.total_characters > initial_chars


class TestDeepLProviderGetUsage:
    """Tests for get_usage method."""

    @pytest.mark.asyncio
    async def test_get_usage_success(self) -> None:
        """Test successful usage retrieval."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "character_count": 123456,
                "character_limit": 500000,
            }
        )

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        usage = await provider.get_usage()

        assert usage["character_count"] == 123456
        assert usage["character_limit"] == 500000

    @pytest.mark.asyncio
    async def test_get_usage_error(self) -> None:
        """Test handling of usage API error."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 401

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        with pytest.raises(MTError, match="Failed to get usage"):
            await provider.get_usage()


class TestDeepLProviderGetSupportedLanguages:
    """Tests for get_supported_languages method."""

    @pytest.mark.asyncio
    async def test_get_languages_success(self) -> None:
        """Test successful languages retrieval."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[
                {"language": "DE", "name": "German"},
                {"language": "EN", "name": "English"},
                {"language": "RU", "name": "Russian"},
            ]
        )

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        languages = await provider.get_supported_languages()

        assert len(languages) == 3
        assert {"code": "DE", "name": "German"} in languages
        assert {"code": "EN", "name": "English"} in languages

    @pytest.mark.asyncio
    async def test_get_languages_error(self) -> None:
        """Test handling of languages API error."""
        provider = DeepLProvider(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 500

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.closed = False

        provider._session = mock_session

        with pytest.raises(MTError, match="Failed to get languages"):
            await provider.get_supported_languages()


class TestDeepLProviderSession:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_close_session(self) -> None:
        """Test closing the session."""
        provider = DeepLProvider(api_key="test-key")

        # Create a mock session
        mock_session = AsyncMock()
        mock_session.closed = False
        provider._session = mock_session

        await provider.close()

        mock_session.close.assert_called_once()
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_close_already_closed(self) -> None:
        """Test closing when already closed."""
        provider = DeepLProvider(api_key="test-key")
        provider._session = None

        # Should not raise
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self) -> None:
        """Test that _get_session creates a new session if needed."""
        provider = DeepLProvider(api_key="test-key")

        assert provider._session is None

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            session = await provider._get_session()

            mock_session_class.assert_called_once()
            assert session is mock_session

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self) -> None:
        """Test that _get_session reuses existing session."""
        provider = DeepLProvider(api_key="test-key")

        mock_session = MagicMock()
        mock_session.closed = False
        provider._session = mock_session

        session = await provider._get_session()

        assert session is mock_session


class TestMTExceptions:
    """Tests for MT exception classes."""

    def test_mt_error(self) -> None:
        """Test MTError exception."""
        error = MTError("Translation failed")
        assert str(error) == "Translation failed"

    def test_mt_quota_exceeded(self) -> None:
        """Test MTQuotaExceededError exception."""
        error = MTQuotaExceededError("Quota exceeded")
        assert str(error) == "Quota exceeded"
        assert isinstance(error, MTError)  # Should inherit from MTError
