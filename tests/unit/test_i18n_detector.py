"""Unit tests for i18n language detector.

Tests system language detection across platforms.
"""

import os
from unittest.mock import patch

import pytest

from kttc.i18n.detector import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    _normalize_language_code,
    detect_system_language,
    get_supported_languages,
    is_supported_language,
)


@pytest.mark.unit
class TestNormalizeLanguageCode:
    """Test language code normalization."""

    def test_normalize_russian_locale(self) -> None:
        """Test normalizing Russian locale string."""
        assert _normalize_language_code("ru_RU.UTF-8") == "ru"

    def test_normalize_english_us(self) -> None:
        """Test normalizing English US locale."""
        assert _normalize_language_code("en_US") == "en"

    def test_normalize_english_uk(self) -> None:
        """Test normalizing English UK locale."""
        assert _normalize_language_code("en-GB") == "en"

    def test_normalize_chinese_simplified(self) -> None:
        """Test normalizing Chinese simplified."""
        assert _normalize_language_code("zh_CN.UTF-8") == "zh"

    def test_normalize_chinese_traditional(self) -> None:
        """Test normalizing Chinese traditional."""
        assert _normalize_language_code("zh_TW") == "zh"

    def test_normalize_persian(self) -> None:
        """Test normalizing Persian."""
        assert _normalize_language_code("fa_IR") == "fa"

    def test_normalize_hindi(self) -> None:
        """Test normalizing Hindi."""
        assert _normalize_language_code("hi_IN") == "hi"

    def test_normalize_empty_string(self) -> None:
        """Test normalizing empty string returns default."""
        assert _normalize_language_code("") == DEFAULT_LANGUAGE

    def test_normalize_simple_code(self) -> None:
        """Test normalizing simple two-letter code."""
        assert _normalize_language_code("ru") == "ru"
        assert _normalize_language_code("en") == "en"


@pytest.mark.unit
class TestDetectSystemLanguage:
    """Test system language detection."""

    def test_kttc_ui_lang_env_variable(self) -> None:
        """Test KTTC_UI_LANG environment variable takes priority."""
        with patch.dict(os.environ, {"KTTC_UI_LANG": "ru"}):
            assert detect_system_language() == "ru"

    def test_kttc_ui_lang_with_full_locale(self) -> None:
        """Test KTTC_UI_LANG with full locale string."""
        with patch.dict(os.environ, {"KTTC_UI_LANG": "ru_RU.UTF-8"}):
            assert detect_system_language() == "ru"

    def test_lc_all_env_variable(self) -> None:
        """Test LC_ALL environment variable."""
        env = {"LC_ALL": "zh_CN.UTF-8"}
        with patch.dict(os.environ, env, clear=True):
            lang = detect_system_language()
            # Should detect Chinese or fall back to default
            assert lang in SUPPORTED_LANGUAGES

    def test_lang_env_variable(self) -> None:
        """Test LANG environment variable."""
        env = {"LANG": "fa_IR"}
        with patch.dict(os.environ, env, clear=True):
            lang = detect_system_language()
            assert lang in SUPPORTED_LANGUAGES

    def test_unsupported_language_falls_back(self) -> None:
        """Test unsupported language falls back to default."""
        with patch.dict(os.environ, {"KTTC_UI_LANG": "xyz"}):
            # Should not return xyz since it's not supported
            lang = detect_system_language()
            assert lang in SUPPORTED_LANGUAGES

    def test_default_fallback(self) -> None:
        """Test default fallback when no language detected."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("locale.getdefaultlocale", return_value=(None, None)):
                lang = detect_system_language()
                assert lang == DEFAULT_LANGUAGE


@pytest.mark.unit
class TestSupportedLanguages:
    """Test supported languages utilities."""

    def test_get_supported_languages(self) -> None:
        """Test getting list of supported languages."""
        langs = get_supported_languages()

        assert isinstance(langs, list)
        assert len(langs) == len(SUPPORTED_LANGUAGES)
        assert "en" in langs
        assert "ru" in langs
        assert "zh" in langs

    def test_get_supported_languages_sorted(self) -> None:
        """Test that supported languages list is sorted."""
        langs = get_supported_languages()
        assert langs == sorted(langs)

    def test_is_supported_language_true(self) -> None:
        """Test is_supported_language for supported languages."""
        assert is_supported_language("en") is True
        assert is_supported_language("ru") is True
        assert is_supported_language("zh") is True
        assert is_supported_language("fa") is True
        assert is_supported_language("hi") is True

    def test_is_supported_language_false(self) -> None:
        """Test is_supported_language for unsupported languages."""
        assert is_supported_language("de") is False
        assert is_supported_language("xyz") is False

    def test_is_supported_language_with_locale(self) -> None:
        """Test is_supported_language with full locale string."""
        assert is_supported_language("ru_RU.UTF-8") is True
        assert is_supported_language("en-US") is True
        assert is_supported_language("zh_CN") is True


@pytest.mark.unit
class TestSupportedLanguagesConstants:
    """Test supported languages constants."""

    def test_supported_languages_contains_english(self) -> None:
        """Test that English is in supported languages."""
        assert "en" in SUPPORTED_LANGUAGES

    def test_supported_languages_contains_russian(self) -> None:
        """Test that Russian is in supported languages."""
        assert "ru" in SUPPORTED_LANGUAGES

    def test_supported_languages_contains_chinese(self) -> None:
        """Test that Chinese is in supported languages."""
        assert "zh" in SUPPORTED_LANGUAGES

    def test_supported_languages_contains_persian(self) -> None:
        """Test that Persian is in supported languages."""
        assert "fa" in SUPPORTED_LANGUAGES

    def test_supported_languages_contains_hindi(self) -> None:
        """Test that Hindi is in supported languages."""
        assert "hi" in SUPPORTED_LANGUAGES

    def test_default_language_is_english(self) -> None:
        """Test that default language is English."""
        assert DEFAULT_LANGUAGE == "en"
