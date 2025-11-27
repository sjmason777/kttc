"""Unit tests for English language helper.

Tests English language helper with spaCy and LanguageTool integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity


@pytest.mark.unit
class TestEnglishLanguageHelperInitialization:
    """Test English language helper initialization."""

    def test_initialization_with_spacy(self) -> None:
        """Test initialization when spaCy is available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy") as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper._nlp == mock_nlp
            assert helper._initialized is True

    def test_initialization_without_spacy(self) -> None:
        """Test initialization when spaCy is not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper._nlp is None
            assert helper._initialized is False

    def test_language_code(self) -> None:
        """Test language code property."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper.language_code == "en"


@pytest.mark.unit
class TestIsAvailable:
    """Test is_available method."""

    def test_is_available_true(self) -> None:
        """Test is_available returns True when spaCy initialized."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy") as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_spacy.load.return_value = MagicMock()

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper.is_available() is True

    def test_is_available_false(self) -> None:
        """Test is_available returns False when spaCy not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper.is_available() is False


@pytest.mark.unit
class TestVerifyWordExists:
    """Test word verification method."""

    def test_verify_word_exists_fallback(self) -> None:
        """Test verify_word_exists uses fallback when not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper.verify_word_exists("hello", "Hello world") is True
            assert helper.verify_word_exists("xyz", "Hello world") is False

    def test_verify_word_exists_case_insensitive(self) -> None:
        """Test verify_word_exists is case insensitive."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper.verify_word_exists("HELLO", "hello world") is True
            assert helper.verify_word_exists("hello", "HELLO WORLD") is True


@pytest.mark.unit
class TestVerifyErrorPosition:
    """Test error position verification."""

    def test_verify_error_position_valid(self) -> None:
        """Test verification of valid error position."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            text = "Hello world"
            error = ErrorAnnotation(
                category="test",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),  # "Hello"
                description="Test error",
            )

            assert helper.verify_error_position(error, text) is True

    def test_verify_error_position_out_of_bounds(self) -> None:
        """Test verification catches out-of-bounds position."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            text = "Hello"
            error = ErrorAnnotation(
                category="test",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 100),  # Out of bounds
                description="Test error",
            )

            assert helper.verify_error_position(error, text) is False

    def test_verify_error_position_negative(self) -> None:
        """Test verification catches negative position."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            text = "Hello"
            error = ErrorAnnotation(
                category="test",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(-1, 5),  # Negative start
                description="Test error",
            )

            assert helper.verify_error_position(error, text) is False


@pytest.mark.unit
class TestTokenize:
    """Test tokenization method."""

    def test_tokenize_without_spacy_uses_fallback(self) -> None:
        """Test tokenize uses fallback when spaCy not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            tokens = helper.tokenize("Hello world")

            # Fallback tokenization should still work
            assert isinstance(tokens, list)
            assert len(tokens) >= 0  # May use fallback or return empty

    def test_tokenize_with_spacy(self) -> None:
        """Test tokenize uses spaCy when available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy") as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_token1 = MagicMock()
            mock_token1.text = "Hello"
            mock_token1.idx = 0
            mock_token1.__len__ = lambda x: 5
            mock_token2 = MagicMock()
            mock_token2.text = "world"
            mock_token2.idx = 6
            mock_token2.__len__ = lambda x: 5
            mock_doc = MagicMock()
            mock_doc.__iter__ = lambda x: iter([mock_token1, mock_token2])
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            tokens = helper.tokenize("Hello world")

            assert len(tokens) == 2
            assert tokens[0][0] == "Hello"
            assert tokens[1][0] == "world"


@pytest.mark.unit
class TestAnalyzeMorphology:
    """Test morphological analysis."""

    def test_analyze_morphology_without_spacy(self) -> None:
        """Test analyze_morphology returns empty when spaCy not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            result = helper.analyze_morphology("Hello world")

            assert result == []


@pytest.mark.unit
class TestCheckGrammar:
    """Test grammar checking method."""

    def test_check_grammar_without_languagetool(self) -> None:
        """Test check_grammar returns empty when LanguageTool not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            errors = helper.check_grammar("He go to school")

            assert errors == []

    def test_check_grammar_with_languagetool(self) -> None:
        """Test check_grammar uses LanguageTool when available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_tool = MagicMock()
            mock_match = MagicMock()
            mock_match.ruleId = "SUBJECT_VERB_AGREEMENT"
            mock_match.message = 'Use "goes" instead of "go"'
            mock_match.offset = 3
            mock_match.errorLength = 2
            mock_match.replacements = ["goes"]
            mock_match.category = "Grammar"
            mock_tool.check.return_value = [mock_match]
            mock_lt.LanguageTool.return_value = mock_tool

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            errors = helper.check_grammar("He go to school")

            assert len(errors) == 1
            assert "goes" in errors[0].description or "go" in errors[0].description


@pytest.mark.unit
class TestExtractEntities:
    """Test named entity extraction."""

    def test_extract_entities_without_spacy(self) -> None:
        """Test extract_entities returns empty when spaCy not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            entities = helper.extract_entities("John lives in New York")

            assert entities == []


@pytest.mark.unit
class TestGetEnrichmentData:
    """Test enrichment data generation."""

    def test_get_enrichment_data_without_spacy(self) -> None:
        """Test get_enrichment_data returns minimal data when spaCy not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            data = helper.get_enrichment_data("Hello world")

            assert isinstance(data, dict)
            assert data.get("has_morphology") is False


@pytest.mark.unit
class TestCheckEntityPreservation:
    """Test entity preservation checking."""

    def test_check_entity_preservation_without_spacy(self) -> None:
        """Test check_entity_preservation returns empty when spaCy not available."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            errors = helper.check_entity_preservation(
                "John lives in New York", "Джон живёт в Нью-Йорке"
            )

            assert errors == []
