"""Unit tests for internationalization module.

Tests i18n functionality including language detection and RTL support.
"""

import pytest

from kttc.helpers.detection import detect_language
from kttc.i18n import (
    is_rtl_language,
)


@pytest.mark.unit
class TestLanguageDetection:
    """Test language detection functionality."""

    def test_detect_russian(self) -> None:
        """Test detecting Russian text."""
        text = "Привет мир"
        result = detect_language(text)
        assert result == "ru"

    def test_detect_english(self) -> None:
        """Test detecting English text."""
        text = "Hello world"
        result = detect_language(text)
        assert result == "en"

    def test_detect_chinese(self) -> None:
        """Test detecting Chinese text."""
        text = "你好世界"
        result = detect_language(text)
        assert result == "zh"

    def test_detect_arabic(self) -> None:
        """Test detecting Arabic text."""
        text = "مرحبا بالعالم"
        result = detect_language(text)
        assert result == "ar"

    def test_detect_hindi(self) -> None:
        """Test detecting Hindi text."""
        text = "नमस्ते दुनिया"
        result = detect_language(text)
        assert result == "hi"

    def test_detect_persian(self) -> None:
        """Test detecting Persian text."""
        text = "سلام جهان"
        result = detect_language(text)
        # Persian and Arabic share similar scripts, so detection may return either
        assert result in ["fa", "ar"]

    def test_detect_empty_text(self) -> None:
        """Test detecting empty text returns default."""
        text = ""
        result = detect_language(text)
        assert result == "en"  # Default fallback


@pytest.mark.unit
class TestRTLLanguages:
    """Test RTL language support."""

    def test_is_rtl_arabic(self) -> None:
        """Test Arabic is RTL."""
        assert is_rtl_language("ar") is True

    def test_is_rtl_persian(self) -> None:
        """Test Persian is RTL."""
        assert is_rtl_language("fa") is True

    def test_is_rtl_hebrew(self) -> None:
        """Test Hebrew is RTL."""
        assert is_rtl_language("he") is True

    def test_is_not_rtl_english(self) -> None:
        """Test English is not RTL."""
        assert is_rtl_language("en") is False

    def test_is_not_rtl_russian(self) -> None:
        """Test Russian is not RTL."""
        assert is_rtl_language("ru") is False

    def test_is_not_rtl_chinese(self) -> None:
        """Test Chinese is not RTL."""
        assert is_rtl_language("zh") is False


@pytest.mark.unit
class TestI18nEdgeCases:
    """Test edge cases in i18n module."""

    def test_mixed_language_detection(self) -> None:
        """Test text with mixed languages."""
        text = "Hello мир"
        result = detect_language(text)
        # Should detect some language
        assert result in ["en", "ru"]

    def test_whitespace_only_detection(self) -> None:
        """Test detecting whitespace-only text."""
        text = "   "
        result = detect_language(text)
        assert result == "en"  # Default

    def test_numbers_only_detection(self) -> None:
        """Test detecting numbers-only text."""
        text = "12345"
        result = detect_language(text)
        assert result == "en"  # Default
