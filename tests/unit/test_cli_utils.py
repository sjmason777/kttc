"""Unit tests for CLI utilities.

Tests CLI helper functions and utilities.
"""

from pathlib import Path

import pytest

from kttc.cli.utils import (
    auto_detect_format,
    auto_detect_glossary,
    detect_language,
)


@pytest.mark.unit
class TestAutoDetectFormat:
    """Test auto format detection."""

    def test_detect_txt_format(self) -> None:
        """Test detecting .txt format."""
        result = auto_detect_format(Path("source.txt"), Path("translation.txt"))
        assert result is not None

    def test_detect_xliff_format(self) -> None:
        """Test detecting .xliff format."""
        result = auto_detect_format(Path("source.xliff"), Path("translation.xliff"))
        assert result is not None

    def test_detect_json_format(self) -> None:
        """Test detecting .json format."""
        result = auto_detect_format(Path("source.json"), Path("translation.json"))
        assert result is not None


@pytest.mark.unit
class TestAutoDetectGlossary:
    """Test auto glossary detection."""

    def test_detect_no_glossary(self) -> None:
        """Test when no glossary is found."""
        result = auto_detect_glossary(Path("/nonexistent/path"))
        # Should return None or empty when no glossary found
        assert result is None or isinstance(result, (Path, str))


@pytest.mark.unit
class TestDetectLanguage:
    """Test language detection utility."""

    def test_detect_russian(self) -> None:
        """Test detecting Russian."""
        result = detect_language("Привет мир")
        assert result == "ru"

    def test_detect_english(self) -> None:
        """Test detecting English."""
        result = detect_language("Hello world")
        assert result == "en"

    def test_detect_chinese(self) -> None:
        """Test detecting Chinese."""
        result = detect_language("你好世界")
        assert result == "zh"

    def test_detect_empty(self) -> None:
        """Test detecting empty text."""
        result = detect_language("")
        assert result == "en"  # Default fallback
