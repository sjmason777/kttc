"""Unit tests for language detection module.

Tests automatic language detection and helper selection.
"""

import pytest

from kttc.helpers.detection import detect_language, get_helper_for_language


@pytest.mark.unit
class TestDetectLanguage:
    """Test language detection functionality."""

    def test_detect_russian(self) -> None:
        """Test Russian language detection."""
        assert detect_language("Привет, мир!") == "ru"
        assert detect_language("Как дела?") == "ru"
        assert detect_language("Добрый день") == "ru"

    def test_detect_english(self) -> None:
        """Test English language detection."""
        assert detect_language("Hello, world!") == "en"
        assert detect_language("How are you?") == "en"
        assert detect_language("Good morning") == "en"

    def test_detect_chinese(self) -> None:
        """Test Chinese language detection."""
        assert detect_language("你好世界") == "zh"
        assert detect_language("这是一个测试") == "zh"

    def test_detect_hindi(self) -> None:
        """Test Hindi language detection."""
        assert detect_language("नमस्ते दुनिया") == "hi"
        assert detect_language("यह एक परीक्षण है") == "hi"

    def test_detect_persian(self) -> None:
        """Test Persian language detection."""
        # Persian-specific characters پ چ ژ گ
        assert detect_language("سلام دنیا پیام") == "fa"
        assert detect_language("این یک تست است گربه") == "fa"

    def test_detect_arabic(self) -> None:
        """Test Arabic language detection."""
        # Arabic without Persian-specific characters
        assert detect_language("مرحبا بالعالم") == "ar"

    def test_detect_empty_string(self) -> None:
        """Test empty string returns default."""
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"

    def test_detect_mixed_content(self) -> None:
        """Test mixed language content uses dominant script."""
        # Mostly Russian with some English
        text = "Привет Hello Мир World Тест"
        result = detect_language(text)
        # Should detect Russian since more Cyrillic chars
        assert result == "ru"


@pytest.mark.unit
class TestGetHelperForLanguage:
    """Test helper selection by language code."""

    def test_get_russian_helper(self) -> None:
        """Test getting Russian helper (may be None if deps unavailable)."""
        helper = get_helper_for_language("ru")
        # Helper may or may not be available based on dependencies
        if helper is not None:
            assert helper.language_code == "ru"

    def test_get_english_helper(self) -> None:
        """Test getting English helper (may be None if deps unavailable)."""
        helper = get_helper_for_language("en")
        if helper is not None:
            assert helper.language_code == "en"

    def test_get_chinese_helper(self) -> None:
        """Test getting Chinese helper (may be None if deps unavailable)."""
        helper = get_helper_for_language("zh")
        if helper is not None:
            assert helper.language_code == "zh"

    def test_get_hindi_helper(self) -> None:
        """Test getting Hindi helper (may be None if deps unavailable)."""
        helper = get_helper_for_language("hi")
        if helper is not None:
            assert helper.language_code == "hi"

    def test_get_persian_helper(self) -> None:
        """Test getting Persian helper (may be None if deps unavailable)."""
        helper = get_helper_for_language("fa")
        if helper is not None:
            assert helper.language_code == "fa"

    def test_unsupported_language_returns_none(self) -> None:
        """Test unsupported language returns None."""
        helper = get_helper_for_language("xyz")
        assert helper is None

    def test_case_insensitive(self) -> None:
        """Test language code is case insensitive."""
        helper1 = get_helper_for_language("RU")
        helper2 = get_helper_for_language("ru")
        # Both should behave the same
        assert (helper1 is None) == (helper2 is None)
