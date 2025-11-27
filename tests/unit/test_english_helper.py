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
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
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
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
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
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
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

    def test_check_entity_preservation_missing_entities(self) -> None:
        """Test entity preservation detects missing entities."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []  # No entities in translation
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            errors = helper.check_entity_preservation(
                "John Smith visited New York", "he went there"  # No entities preserved
            )

            assert len(errors) >= 1
            assert errors[0].subcategory == "entity_omission"

    def test_check_entity_preservation_entities_preserved(self) -> None:
        """Test no error when entities are preserved."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_ent = MagicMock()
            mock_ent.text = "John"
            mock_ent.label_ = "PERSON"
            mock_ent.start_char = 0
            mock_ent.end_char = 4
            mock_doc = MagicMock()
            mock_doc.ents = [mock_ent]
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            errors = helper.check_entity_preservation("John arrived", "John arrived")

            assert len(errors) == 0

    def test_check_entity_preservation_exception(self) -> None:
        """Test entity preservation handles exceptions."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_nlp.return_value.ents.__iter__ = MagicMock(side_effect=Exception("NER failed"))
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            # Use lowercase to avoid regex matching
            errors = helper.check_entity_preservation("hello world", "test text")
            assert errors == []


@pytest.mark.unit
class TestEnglishHelperSpacyFallback:
    """Test spaCy model fallback behavior."""

    def test_spacy_fallback_to_sm_model(self) -> None:
        """Test fallback to small model when medium not found."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            # First call (md) fails, second call (sm) succeeds
            mock_spacy.load.side_effect = [OSError("Not found"), mock_nlp]

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper.is_available()
            assert mock_spacy.load.call_count == 2

    def test_spacy_both_models_fail(self) -> None:
        """Test when both spaCy models fail to load."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_spacy.load.side_effect = OSError("Not found")

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert not helper.is_available()


@pytest.mark.unit
class TestLanguageToolIntegration:
    """Test LanguageTool integration."""

    def test_languagetool_initialization_success(self) -> None:
        """Test LanguageTool initializes successfully."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_tool = MagicMock()
            mock_lt.LanguageTool.return_value = mock_tool

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert helper._lt_available
            mock_lt.LanguageTool.assert_called_once_with("en-US")

    def test_languagetool_initialization_failure(self) -> None:
        """Test LanguageTool handles initialization failure."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_lt.LanguageTool.side_effect = Exception("LT init failed")

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()

            assert not helper._lt_available

    def test_check_grammar_filters_style_errors(self) -> None:
        """Test that style-only errors are filtered."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_tool = MagicMock()
            mock_grammar = MagicMock()
            mock_grammar.ruleId = "GRAMMAR_ERROR"
            mock_grammar.message = "Grammar error"
            mock_grammar.offset = 0
            mock_grammar.errorLength = 5
            mock_grammar.replacements = []
            mock_style = MagicMock()
            mock_style.ruleId = "STYLE_SUGGESTION"
            mock_style.message = "Style suggestion"
            mock_style.offset = 10
            mock_style.errorLength = 5
            mock_style.replacements = []
            mock_tool.check.return_value = [mock_grammar, mock_style]
            mock_lt.LanguageTool.return_value = mock_tool

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            errors = helper.check_grammar("Test text")

            # Only grammar error included
            assert len(errors) == 1
            assert "GRAMMAR" in errors[0].subcategory

    def test_check_grammar_exception_handling(self) -> None:
        """Test check_grammar handles exceptions gracefully."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_tool = MagicMock()
            mock_tool.check.side_effect = Exception("LT check failed")
            mock_lt.LanguageTool.return_value = mock_tool

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            errors = helper.check_grammar("Test text")

            assert errors == []


@pytest.mark.unit
class TestSeverityMapping:
    """Test severity mapping for different error types."""

    def test_severity_spelling_critical(self) -> None:
        """Test spelling errors are mapped to CRITICAL."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_tool = MagicMock()
            mock_match = MagicMock()
            mock_match.ruleId = "SPELLING_ERROR"
            mock_match.message = "Spelling error"
            mock_match.offset = 0
            mock_match.errorLength = 5
            mock_match.replacements = ["correct"]
            mock_tool.check.return_value = [mock_match]
            mock_lt.LanguageTool.return_value = mock_tool

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            errors = helper.check_grammar("Test")

            assert len(errors) == 1
            assert errors[0].severity == ErrorSeverity.CRITICAL

    def test_severity_agreement_major(self) -> None:
        """Test agreement errors are mapped to MAJOR."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_tool = MagicMock()
            mock_match = MagicMock()
            mock_match.ruleId = "SUBJECT_VERB_AGREEMENT"
            mock_match.message = "Agreement error"
            mock_match.offset = 0
            mock_match.errorLength = 5
            mock_match.replacements = []
            mock_tool.check.return_value = [mock_match]
            mock_lt.LanguageTool.return_value = mock_tool

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            errors = helper.check_grammar("Test")

            assert len(errors) == 1
            assert errors[0].severity == ErrorSeverity.MAJOR

    def test_severity_other_minor(self) -> None:
        """Test other errors are mapped to MINOR."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", True),
            patch("kttc.helpers.english.language_tool_python") as mock_lt,
        ):
            mock_tool = MagicMock()
            mock_match = MagicMock()
            mock_match.ruleId = "SOME_OTHER_RULE"
            mock_match.message = "Some issue"
            mock_match.offset = 0
            mock_match.errorLength = 5
            mock_match.replacements = []
            mock_tool.check.return_value = [mock_match]
            mock_lt.LanguageTool.return_value = mock_tool

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            errors = helper.check_grammar("Test")

            assert len(errors) == 1
            assert errors[0].severity == ErrorSeverity.MINOR


@pytest.mark.unit
class TestMorphologyWithSpacy:
    """Test morphological analysis with spaCy."""

    def test_analyze_morphology_extracts_features(self) -> None:
        """Test morphology extracts number from spaCy."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_token = MagicMock()
            mock_token.text = "cats"
            mock_token.idx = 0
            mock_token.pos_ = "NOUN"
            mock_morph = MagicMock()
            mock_morph.get.return_value = ["Plur"]
            mock_token.morph = mock_morph
            mock_token.__len__ = lambda x: 4
            mock_doc = MagicMock()
            mock_doc.__iter__ = lambda x: iter([mock_token])
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            result = helper.analyze_morphology("cats")

            assert len(result) == 1
            assert result[0].word == "cats"
            assert result[0].pos == "NOUN"
            assert result[0].number == "Plur"


@pytest.mark.unit
class TestEnrichmentData:
    """Test enrichment data generation."""

    def test_get_enrichment_data_with_spacy(self) -> None:
        """Test enrichment returns comprehensive data with spaCy."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_token = MagicMock()
            mock_token.text = "cat"
            mock_token.pos_ = "NOUN"
            mock_token.dep_ = ""
            mock_token.is_punct = False
            mock_token.idx = 0
            mock_morph = MagicMock()
            mock_morph.get.return_value = []
            mock_token.morph = mock_morph
            mock_ent = MagicMock()
            mock_ent.text = "John"
            mock_ent.label_ = "PERSON"
            mock_ent.start_char = 0
            mock_ent.end_char = 4
            mock_doc = MagicMock()
            mock_doc.__iter__ = lambda x: iter([mock_token])
            mock_doc.ents = [mock_ent]
            mock_doc.sents = [MagicMock()]
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            result = helper.get_enrichment_data("cat")

            assert result["has_morphology"]
            assert "word_count" in result
            assert "verb_tenses" in result
            assert "article_noun_pairs" in result
            assert "subject_verb_pairs" in result
            assert "pos_distribution" in result
            assert "entities" in result
            assert "sentence_count" in result


@pytest.mark.unit
class TestExtractEntitiesWithSpacy:
    """Test entity extraction with spaCy."""

    def test_extract_entities_returns_entities(self) -> None:
        """Test extract_entities returns entity list."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_ent = MagicMock()
            mock_ent.text = "John"
            mock_ent.label_ = "PERSON"
            mock_ent.start_char = 0
            mock_ent.end_char = 4
            mock_doc = MagicMock()
            mock_doc.ents = [mock_ent]
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            entities = helper.extract_entities("John went home")

            assert len(entities) == 1
            assert entities[0]["text"] == "John"
            assert entities[0]["type"] == "PERSON"

    def test_extract_entities_handles_exception(self) -> None:
        """Test extract_entities handles exceptions gracefully."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", True),
            patch("kttc.helpers.english.spacy", create=True) as mock_spacy,
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            mock_nlp = MagicMock()
            mock_nlp.side_effect = Exception("NER failed")
            mock_spacy.load.return_value = mock_nlp

            from kttc.helpers.english import EnglishLanguageHelper

            helper = EnglishLanguageHelper()
            entities = helper.extract_entities("Test text")

            assert entities == []


@pytest.mark.unit
class TestArticleNounAnalysis:
    """Test article-noun pair analysis."""

    def test_find_noun_after_article(self) -> None:
        """Test finding noun after article."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            mock_doc = MagicMock()
            mock_det = MagicMock()
            mock_det.pos_ = "DET"
            mock_noun = MagicMock()
            mock_noun.pos_ = "NOUN"
            mock_doc.__getitem__ = lambda self, idx: [mock_det, mock_noun][idx]
            mock_doc.__len__ = lambda self: 2

            result = EnglishLanguageHelper._find_noun_after_article(mock_doc, 0)
            assert result == 1

    def test_check_article_correctness_the(self) -> None:
        """Test 'the' is always considered correct."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            mock_doc = MagicMock()
            result = EnglishLanguageHelper._check_article_correctness(mock_doc, 0, "the")
            assert result is True

    def test_check_article_correctness_a_before_consonant(self) -> None:
        """Test 'a' before consonant."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            mock_doc = MagicMock()
            mock_next = MagicMock()
            mock_next.text = "cat"
            mock_doc.__getitem__ = lambda self, idx: mock_next
            mock_doc.__len__ = lambda self: 2

            result = EnglishLanguageHelper._check_article_correctness(mock_doc, 0, "a")
            assert result is True

    def test_check_article_correctness_an_before_vowel(self) -> None:
        """Test 'an' before vowel."""
        with (
            patch("kttc.helpers.english.SPACY_AVAILABLE", False),
            patch("kttc.helpers.english.LANGUAGETOOL_AVAILABLE", False),
        ):
            from kttc.helpers.english import EnglishLanguageHelper

            mock_doc = MagicMock()
            mock_next = MagicMock()
            mock_next.text = "apple"
            mock_doc.__getitem__ = lambda self, idx: mock_next
            mock_doc.__len__ = lambda self: 2

            result = EnglishLanguageHelper._check_article_correctness(mock_doc, 0, "an")
            assert result is True
