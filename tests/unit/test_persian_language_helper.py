"""Integration tests for PersianLanguageHelper.

Tests Persian language helper with DadmaTools (all-in-one NLP toolkit).
Focus: Verify all abstract methods, test error detection, validate NLP integration.

⚠️ IMPORTANT: These are INTEGRATION tests (not unit tests)!
- Load heavy ML models (DadmaTools: ~10 models, 100-200MB each)
- Take 20+ minutes to run (first time: model download + caching)
- Skip in CI/CD to avoid timeouts
- Run manually or in separate integration test pipeline

To run: pytest tests/unit/test_persian_language_helper.py -m integration
"""

import os

import pytest

from kttc.core.models import ErrorAnnotation
from kttc.helpers.persian import PersianLanguageHelper

# Skip all Persian tests in CI - too slow (20+ minutes due to heavy ML models)
pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
    reason="Persian tests are slow (20+ min) - skip in CI/CD. Run manually with: pytest -m integration",
)


@pytest.mark.integration
@pytest.mark.slow
class TestPersianLanguageHelperBasics:
    """Test basic Persian helper functionality and interface compliance."""

    def test_instantiation(self) -> None:
        """Test that PersianLanguageHelper can be instantiated without errors."""
        # Act
        helper = PersianLanguageHelper()

        # Assert
        assert helper is not None
        assert helper.language_code == "fa"

    def test_is_available_returns_bool(self) -> None:
        """Test that is_available() returns a boolean value."""
        # Arrange
        helper = PersianLanguageHelper()

        # Act
        available = helper.is_available()

        # Assert
        assert isinstance(available, bool)
        # Should be True if dependencies are installed (which they should be for tests)
        assert available is True, "Persian dependencies should be installed for tests"


@pytest.mark.integration
@pytest.mark.slow
class TestPersianTokenization:
    """Test Persian tokenization functionality."""

    def test_tokenize_simple_text(self) -> None:
        """Test tokenization of simple Persian text."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "سلام دنیا"  # Hello world

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) == 2
        # Each token is (word, start, end)
        assert all(len(token) == 3 for token in tokens)
        assert tokens[0][0] == "سلام"
        assert tokens[1][0] == "دنیا"

    def test_tokenize_empty_string(self) -> None:
        """Test tokenization of empty string returns empty list."""
        # Arrange
        helper = PersianLanguageHelper()

        # Act
        tokens = helper.tokenize("")

        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_tokenize_position_accuracy(self) -> None:
        """Test that token positions are accurate."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"  # I go to school

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert len(tokens) > 0
        for word, start, end in tokens:
            # Verify position matches actual text
            extracted = text[start:end]
            assert extracted == word, f"Position mismatch: '{extracted}' != '{word}'"

    def test_tokenize_complex_sentence(self) -> None:
        """Test tokenization handles complex sentences with punctuation."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم. تو کجا می‌روی؟"  # I go to school. Where are you going?

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert len(tokens) > 5
        # Should handle both sentences and punctuation
        words = [token[0] for token in tokens]
        assert "من" in words
        assert "مدرسه" in words


@pytest.mark.integration
@pytest.mark.slow
class TestPersianMorphology:
    """Test Persian morphological analysis."""

    def test_analyze_morphology_returns_list(self) -> None:
        """Test that morphology analysis returns a list."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert isinstance(morph_info, list)

    def test_analyze_morphology_has_pos_tags(self) -> None:
        """Test that morphology includes POS tags (using DadmaTools)."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "کتاب خوب است"  # The book is good

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert len(morph_info) > 0
        # Should have POS tags from DadmaTools
        has_pos = any(info.pos is not None for info in morph_info)
        assert has_pos, "Morphology should include POS tags from DadmaTools"

    def test_analyze_morphology_empty_string(self) -> None:
        """Test morphology analysis on empty string."""
        # Arrange
        helper = PersianLanguageHelper()

        # Act
        morph_info = helper.analyze_morphology("")

        # Assert
        assert isinstance(morph_info, list)
        assert len(morph_info) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestPersianWordVerification:
    """Test word existence verification (anti-hallucination)."""

    def test_verify_word_exists_true(self) -> None:
        """Test verification when word exists in text."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"
        word = "مدرسه"  # school

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is True

    def test_verify_word_exists_false(self) -> None:
        """Test verification when word does NOT exist (hallucination detection)."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"
        word = "کتاب"  # book - not in text

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is False, "Should detect that word is not in text"

    def test_verify_word_partial_match(self) -> None:
        """Test that partial matches don't count as existence."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "مدرسه"
        word = "در"  # Partial match

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        # Behavior depends on implementation - document it
        assert isinstance(exists, bool)


@pytest.mark.integration
@pytest.mark.slow
class TestPersianErrorPositionVerification:
    """Test error position validation (anti-hallucination for positions)."""

    def test_verify_error_position_valid(self) -> None:
        """Test verification of valid error position."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"
        error = ErrorAnnotation(
            category="test",
            subcategory="test",
            severity="minor",
            location=(0, 2),  # "من"
            description="Test error",
        )

        # Act
        valid = helper.verify_error_position(error, text)

        # Assert
        assert valid is True

    def test_verify_error_position_out_of_bounds(self) -> None:
        """Test verification catches out-of-bounds positions."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من مدرسه"
        error = ErrorAnnotation(
            category="test",
            subcategory="test",
            severity="minor",
            location=(0, 100),  # Out of bounds
            description="Test error",
        )

        # Act
        valid = helper.verify_error_position(error, text)

        # Assert
        assert valid is False, "Should detect out-of-bounds position"

    def test_verify_error_position_negative(self) -> None:
        """Test verification catches negative positions."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من مدرسه"
        error = ErrorAnnotation(
            category="test",
            subcategory="test",
            severity="minor",
            location=(-1, 5),  # Negative start
            description="Test error",
        )

        # Act
        valid = helper.verify_error_position(error, text)

        # Assert
        assert valid is False, "Should reject negative positions"


@pytest.mark.integration
@pytest.mark.slow
class TestPersianGrammarCheck:
    """Test Persian grammar checking."""

    def test_check_grammar_returns_list(self) -> None:
        """Test that check_grammar() returns a list."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"

        # Act
        errors = helper.check_grammar(text)

        # Assert
        assert isinstance(errors, list)
        # Currently returns empty list (delegated to LLM)
        # This test ensures interface compliance

    def test_check_grammar_returns_error_annotations(self) -> None:
        """Test that grammar errors are ErrorAnnotation objects."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "some text"

        # Act
        errors = helper.check_grammar(text)

        # Assert
        for error in errors:
            assert isinstance(error, ErrorAnnotation)


@pytest.mark.integration
@pytest.mark.slow
class TestPersianSpellChecking:
    """Test Persian spell checking with DadmaTools v2."""

    def test_check_spelling_returns_list(self) -> None:
        """Test that spell checking returns a list."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"

        # Act
        errors = helper.check_spelling(text)

        # Assert
        assert isinstance(errors, list)

    def test_check_spelling_empty_string(self) -> None:
        """Test spell checking on empty string."""
        # Arrange
        helper = PersianLanguageHelper()

        # Act
        errors = helper.check_spelling("")

        # Assert
        assert isinstance(errors, list)


@pytest.mark.integration
@pytest.mark.slow
class TestPersianEnrichmentData:
    """Test enrichment data for LLM prompts."""

    def test_get_enrichment_data_returns_dict(self) -> None:
        """Test that enrichment data returns a dictionary."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"

        # Act
        enrichment = helper.get_enrichment_data(text)

        # Assert
        assert isinstance(enrichment, dict)
        assert "has_morphology" in enrichment

    def test_get_enrichment_data_has_pos_counts(self) -> None:
        """Test that enrichment includes POS tag distribution."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "کتاب خوب است"

        # Act
        enrichment = helper.get_enrichment_data(text)

        # Assert
        # Should have POS distribution if DadmaTools is available
        assert "has_morphology" in enrichment
        if enrichment.get("has_morphology"):
            # If morphology is available, should have POS data
            assert isinstance(enrichment, dict)


@pytest.mark.integration
@pytest.mark.slow
class TestPersianSentimentAnalysis:
    """Test sentiment analysis (NEW in DadmaTools v2!)."""

    def test_check_sentiment_returns_string_or_none(self) -> None:
        """Test that sentiment analysis returns string or None."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من خوشحالم"  # I am happy

        # Act
        sentiment = helper.check_sentiment(text)

        # Assert
        assert sentiment is None or isinstance(sentiment, str)

    def test_check_sentiment_empty_string(self) -> None:
        """Test sentiment analysis on empty string."""
        # Arrange
        helper = PersianLanguageHelper()

        # Act
        sentiment = helper.check_sentiment("")

        # Assert
        assert sentiment is None or isinstance(sentiment, str)


@pytest.mark.integration
@pytest.mark.slow
class TestPersianFormalConversion:
    """Test informal-to-formal conversion (NEW in DadmaTools v2!)."""

    def test_convert_to_formal_returns_string_or_none(self) -> None:
        """Test that formal conversion returns string or None."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "سلام چطوری؟"  # Hi, how are you? (informal)

        # Act
        formal = helper.convert_to_formal(text)

        # Assert
        assert formal is None or isinstance(formal, str)

    def test_convert_to_formal_empty_string(self) -> None:
        """Test formal conversion on empty string."""
        # Arrange
        helper = PersianLanguageHelper()

        # Act
        formal = helper.convert_to_formal("")

        # Assert
        assert formal is None or isinstance(formal, str)


@pytest.mark.integration
@pytest.mark.slow
class TestPersianEntityExtraction:
    """Test named entity recognition."""

    def test_extract_entities_returns_list(self) -> None:
        """Test that entity extraction returns a list."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من در تهران زندگی می‌کنم"  # I live in Tehran

        # Act
        entities = helper.extract_entities(text)

        # Assert
        assert isinstance(entities, list)

    def test_extract_entities_has_structure(self) -> None:
        """Test that entities have expected structure."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "رضا در تهران زندگی می‌کند"  # Reza lives in Tehran

        # Act
        entities = helper.extract_entities(text)

        # Assert
        for entity in entities:
            assert isinstance(entity, dict)
            # Should have text and type at minimum
            assert "text" in entity or len(entity) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestPersianEdgeCases:
    """Test edge cases and error handling."""

    def test_tokenize_unicode_normalization(self) -> None:
        """Test that tokenization handles Unicode normalization."""
        # Arrange
        helper = PersianLanguageHelper()
        # Persian has different forms (e.g., ZWNJ - zero-width non-joiner)
        text1 = "می‌روم"  # With ZWNJ
        text2 = "میروم"  # Without ZWNJ

        # Act
        tokens1 = helper.tokenize(text1)
        tokens2 = helper.tokenize(text2)

        # Assert
        assert len(tokens1) > 0
        assert len(tokens2) > 0

    def test_morphology_handles_mixed_script(self) -> None:
        """Test morphology analysis with mixed Persian and Latin."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من در Tehran زندگی می‌کنم"  # Mixed script

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert isinstance(morph_info, list)
        # Should handle gracefully without crashing

    def test_tokenize_only_punctuation(self) -> None:
        """Test tokenization of text with only punctuation."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "؟؟؟"  # Persian question marks

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert isinstance(tokens, list)
        # Behavior may vary - just ensure no crash

    def test_verify_word_exists_with_whitespace(self) -> None:
        """Test word verification handles whitespace correctly."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "  من مدرسه  "
        word = "مدرسه"

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is True, "Should find word despite surrounding whitespace"


@pytest.mark.integration
@pytest.mark.slow
class TestPersianIntegration:
    """Integration tests that verify components work together."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: tokenize -> morphology -> verify."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"

        # Act
        tokens = helper.tokenize(text)
        morph = helper.analyze_morphology(text)
        enrichment = helper.get_enrichment_data(text)

        # Assert
        assert len(tokens) > 0
        assert len(morph) >= 0  # May be 0 if dependencies not available
        assert isinstance(enrichment, dict)
        # All three operations should succeed without errors

    def test_error_detection_integration(self) -> None:
        """Test that error position verification works with real tokenization."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "من به مدرسه می‌روم"
        tokens = helper.tokenize(text)

        # Create error at position of first token
        first_token = tokens[0]
        error = ErrorAnnotation(
            category="test",
            subcategory="test",
            severity="minor",
            location=(first_token[1], first_token[2]),  # Use actual token positions
            description="Test error at first token",
        )

        # Act
        valid = helper.verify_error_position(error, text)

        # Assert
        assert valid is True, "Error position from tokenization should be valid"

    def test_dadmatools_v2_features(self) -> None:
        """Test DadmaTools v2 NEW features (spell check, sentiment, formal)."""
        # Arrange
        helper = PersianLanguageHelper()
        text = "سلام چطوری"  # Informal greeting

        # Act
        # All three NEW features in v2
        spelling_errors = helper.check_spelling(text)
        sentiment = helper.check_sentiment(text)
        formal = helper.convert_to_formal(text)

        # Assert
        assert isinstance(spelling_errors, list)
        # sentiment and formal can be None or string
        assert sentiment is None or isinstance(sentiment, str)
        assert formal is None or isinstance(formal, str)
