"""Unit tests for RTL text processing.

Tests right-to-left text handling for Persian and Arabic.
"""

import pytest

from kttc.i18n.rtl import (
    RTL_LANGUAGES,
    get_ltr_marker,
    get_pop_marker,
    get_rtl_marker,
    is_rtl_language,
    process_rtl_text,
    wrap_ltr,
    wrap_rtl,
)


@pytest.mark.unit
class TestIsRTLLanguage:
    """Test RTL language detection."""

    def test_persian_is_rtl(self) -> None:
        """Test that Persian is detected as RTL."""
        assert is_rtl_language("fa") is True

    def test_arabic_is_rtl(self) -> None:
        """Test that Arabic is detected as RTL."""
        assert is_rtl_language("ar") is True

    def test_hebrew_is_rtl(self) -> None:
        """Test that Hebrew is detected as RTL."""
        assert is_rtl_language("he") is True

    def test_english_is_not_rtl(self) -> None:
        """Test that English is not RTL."""
        assert is_rtl_language("en") is False

    def test_russian_is_not_rtl(self) -> None:
        """Test that Russian is not RTL."""
        assert is_rtl_language("ru") is False

    def test_chinese_is_not_rtl(self) -> None:
        """Test that Chinese is not RTL."""
        assert is_rtl_language("zh") is False

    def test_case_insensitive(self) -> None:
        """Test that language code check is case insensitive."""
        assert is_rtl_language("FA") is True
        assert is_rtl_language("Ar") is True


@pytest.mark.unit
class TestRTLLanguagesConstant:
    """Test RTL_LANGUAGES constant."""

    def test_contains_persian(self) -> None:
        """Test that RTL_LANGUAGES contains Persian."""
        assert "fa" in RTL_LANGUAGES

    def test_contains_arabic(self) -> None:
        """Test that RTL_LANGUAGES contains Arabic."""
        assert "ar" in RTL_LANGUAGES

    def test_contains_hebrew(self) -> None:
        """Test that RTL_LANGUAGES contains Hebrew."""
        assert "he" in RTL_LANGUAGES


@pytest.mark.unit
class TestProcessRTLText:
    """Test RTL text processing."""

    def test_non_rtl_language_unchanged(self) -> None:
        """Test that non-RTL text is returned unchanged."""
        text = "Hello world"
        result = process_rtl_text(text, "en")
        assert result == text

    def test_empty_text_unchanged(self) -> None:
        """Test that empty text is returned unchanged."""
        result = process_rtl_text("", "fa")
        assert result == ""

    def test_rtl_text_processed(self) -> None:
        """Test that RTL text is processed (returns string)."""
        text = "سلام"
        result = process_rtl_text(text, "fa")
        # Even without dependencies, should return a string
        assert isinstance(result, str)

    def test_arabic_text_processed(self) -> None:
        """Test that Arabic text is processed."""
        text = "مرحبا"
        result = process_rtl_text(text, "ar")
        assert isinstance(result, str)


@pytest.mark.unit
class TestUnicodeMarkers:
    """Test Unicode directional markers."""

    def test_rtl_marker(self) -> None:
        """Test RTL marker is correct Unicode character."""
        marker = get_rtl_marker()
        assert marker == "\u2067"

    def test_ltr_marker(self) -> None:
        """Test LTR marker is correct Unicode character."""
        marker = get_ltr_marker()
        assert marker == "\u2066"

    def test_pop_marker(self) -> None:
        """Test pop marker is correct Unicode character."""
        marker = get_pop_marker()
        assert marker == "\u2069"


@pytest.mark.unit
class TestWrapFunctions:
    """Test text wrapping functions."""

    def test_wrap_rtl(self) -> None:
        """Test wrapping text with RTL markers."""
        text = "سلام"
        result = wrap_rtl(text)
        assert result.startswith(get_rtl_marker())
        assert result.endswith(get_pop_marker())
        assert text in result

    def test_wrap_ltr(self) -> None:
        """Test wrapping text with LTR markers."""
        text = "Hello"
        result = wrap_ltr(text)
        assert result.startswith(get_ltr_marker())
        assert result.endswith(get_pop_marker())
        assert text in result

    def test_wrap_empty_string(self) -> None:
        """Test wrapping empty string."""
        result_rtl = wrap_rtl("")
        result_ltr = wrap_ltr("")
        assert get_rtl_marker() in result_rtl
        assert get_ltr_marker() in result_ltr
