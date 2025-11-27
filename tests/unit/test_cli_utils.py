"""Unit tests for CLI utilities.

Tests CLI helper functions and utilities.
"""

from unittest.mock import MagicMock

import pytest

from kttc.cli.utils import (
    _resolve_provider_name,
    auto_detect_format,
    auto_detect_glossary,
    get_available_providers,
)
from kttc.helpers.detection import detect_language


@pytest.mark.unit
class TestAutoDetectFormat:
    """Test auto format detection."""

    def test_explicit_format_override(self) -> None:
        """Test that explicit format overrides auto-detection."""
        result = auto_detect_format("output.txt", "json")
        assert result == "json"

    def test_detect_json_from_extension(self) -> None:
        """Test JSON detection from .json extension."""
        result = auto_detect_format("output.json", None)
        assert result == "json"

    def test_detect_markdown_from_extension(self) -> None:
        """Test Markdown detection from .md extension."""
        result = auto_detect_format("output.md", None)
        assert result == "markdown"

        result = auto_detect_format("output.markdown", None)
        assert result == "markdown"

    def test_detect_html_from_extension(self) -> None:
        """Test HTML detection from .html/.htm extension."""
        result = auto_detect_format("output.html", None)
        assert result == "html"

        result = auto_detect_format("output.htm", None)
        assert result == "html"

    def test_detect_xlsx_from_extension(self) -> None:
        """Test XLSX detection from .xlsx/.xls extension."""
        result = auto_detect_format("output.xlsx", None)
        assert result == "xlsx"

        result = auto_detect_format("output.xls", None)
        assert result == "xlsx"

    def test_default_to_text(self) -> None:
        """Test default to text format."""
        result = auto_detect_format("output.txt", None)
        assert result == "text"

        result = auto_detect_format(None, None)
        assert result == "text"

    def test_case_insensitive(self) -> None:
        """Test case-insensitive extension detection."""
        result = auto_detect_format("output.JSON", None)
        assert result == "json"


@pytest.mark.unit
class TestAutoDetectGlossary:
    """Test auto glossary detection."""

    def test_none_glossary(self) -> None:
        """Test 'none' glossary returns None."""
        result = auto_detect_glossary("none")
        assert result is None

    def test_explicit_glossary(self) -> None:
        """Test explicit glossary name passes through."""
        result = auto_detect_glossary("medical")
        assert result == "medical"

        result = auto_detect_glossary("medical,legal")
        assert result == "medical,legal"

    def test_none_input(self) -> None:
        """Test None input returns None."""
        result = auto_detect_glossary(None)
        assert result is None


@pytest.mark.unit
class TestGetAvailableProviders:
    """Test get_available_providers function."""

    def test_no_providers_configured(self) -> None:
        """Test when no providers are configured."""
        mock_settings = MagicMock()
        mock_settings.get_llm_provider_key.side_effect = ValueError("No key")
        mock_settings.get_llm_provider_credentials.side_effect = ValueError("No creds")

        result = get_available_providers(mock_settings)
        assert result == []

    def test_openai_configured(self) -> None:
        """Test when OpenAI is configured."""
        mock_settings = MagicMock()

        def get_key(provider):
            if provider == "openai":
                return "key"
            raise ValueError()

        mock_settings.get_llm_provider_key.side_effect = get_key
        mock_settings.get_llm_provider_credentials.side_effect = ValueError("No creds")

        result = get_available_providers(mock_settings)
        assert "openai" in result

    def test_anthropic_configured(self) -> None:
        """Test when Anthropic is configured."""
        mock_settings = MagicMock()

        def get_key(provider):
            if provider == "anthropic":
                return "key"
            raise ValueError()

        mock_settings.get_llm_provider_key.side_effect = get_key
        mock_settings.get_llm_provider_credentials.side_effect = ValueError("No creds")

        result = get_available_providers(mock_settings)
        assert "anthropic" in result

    def test_multiple_providers_configured(self) -> None:
        """Test when multiple providers are configured."""
        mock_settings = MagicMock()
        mock_settings.get_llm_provider_key.return_value = "key"
        mock_settings.get_llm_provider_credentials.return_value = {"key": "value"}

        result = get_available_providers(mock_settings)
        assert "openai" in result
        assert "anthropic" in result


@pytest.mark.unit
class TestResolveProviderName:
    """Test _resolve_provider_name function."""

    def test_explicit_provider(self) -> None:
        """Test explicit provider name passed through."""
        mock_settings = MagicMock()
        result = _resolve_provider_name("openai", mock_settings)
        assert result == "openai"

    def test_auto_select_from_available(self) -> None:
        """Test auto-selection from available providers."""
        mock_settings = MagicMock()
        mock_settings.get_llm_provider_key.return_value = "key"
        mock_settings.get_llm_provider_credentials.side_effect = ValueError()

        result = _resolve_provider_name(None, mock_settings)
        assert result in ["openai", "anthropic"]

    def test_no_providers_raises(self) -> None:
        """Test raises when no providers available."""
        mock_settings = MagicMock()
        mock_settings.get_llm_provider_key.side_effect = ValueError("No key")
        mock_settings.get_llm_provider_credentials.side_effect = ValueError("No creds")

        with pytest.raises(RuntimeError, match="No LLM providers configured"):
            _resolve_provider_name(None, mock_settings)


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
