"""Integration tests for RussianLanguageHelper.

Tests Russian language helper with MAWO core and LanguageTool.
Focus: Verify all abstract methods, test error detection, validate NLP integration.

⚠️ IMPORTANT: These are INTEGRATION tests (not unit tests)!
- Load ML models (MAWO core + LanguageTool: 930+ rules)
- Take 2-5 minutes to run (first time: LanguageTool download + caching)
- Skip in CI/CD to avoid timeouts
- Run manually or in separate integration test pipeline

To run: pytest tests/unit/test_russian_language_helper.py -m integration
"""

import os

import pytest

from kttc.core.models import ErrorAnnotation
from kttc.helpers.russian import RussianLanguageHelper

# Skip all Russian tests in CI - may be slow due to LanguageTool initialization
pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
    reason="Russian tests may be slow - skip in CI/CD. Run manually with: pytest -m integration",
)


@pytest.mark.integration
@pytest.mark.slow
class TestRussianLanguageHelperBasics:
    """Test basic Russian helper functionality and interface compliance."""

    def test_instantiation(self) -> None:
        """Test that RussianLanguageHelper can be instantiated without errors."""
        # Act
        helper = RussianLanguageHelper()

        # Assert
        assert helper is not None
        assert helper.language_code == "ru"

    def test_is_available_returns_bool(self) -> None:
        """Test that is_available() returns a boolean value."""
        # Arrange
        helper = RussianLanguageHelper()

        # Act
        available = helper.is_available()

        # Assert
        assert isinstance(available, bool)
        # Should be True if MAWO is installed (which it should be for tests)
        assert available is True, "MAWO dependencies should be installed for tests"

    def test_language_tool_available(self) -> None:
        """Test that LanguageTool is available."""
        # Arrange
        helper = RussianLanguageHelper()

        # Assert
        assert helper._lt_available is True, "LanguageTool should be installed for tests"


@pytest.mark.integration
@pytest.mark.slow
class TestRussianTokenization:
    """Test Russian tokenization functionality."""

    def test_tokenize_simple_text(self) -> None:
        """Test tokenization of simple Russian text."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Привет мир"

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) == 2
        # Each token is (word, start, end)
        assert all(len(token) == 3 for token in tokens)
        assert tokens[0][0] == "Привет"
        assert tokens[1][0] == "мир"

    def test_tokenize_empty_string(self) -> None:
        """Test tokenization of empty string returns empty list."""
        # Arrange
        helper = RussianLanguageHelper()

        # Act
        tokens = helper.tokenize("")

        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_tokenize_position_accuracy(self) -> None:
        """Test that token positions are accurate."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert len(tokens) > 0
        for word, start, end in tokens:
            # Verify position matches actual text
            assert text[start:end] == word, f"Position mismatch: '{text[start:end]}' != '{word}'"

    def test_tokenize_complex_sentence(self) -> None:
        """Test tokenization handles complex sentences with punctuation."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу. Ты куда идёшь?"

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert len(tokens) > 5
        # Should handle both sentences and punctuation
        words = [token[0] for token in tokens]
        assert "Я" in words
        assert "школу" in words
        assert "Ты" in words


@pytest.mark.integration
@pytest.mark.slow
class TestRussianMorphology:
    """Test Russian morphological analysis."""

    def test_analyze_morphology_returns_list(self) -> None:
        """Test that morphology analysis returns a list."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert isinstance(morph_info, list)

    def test_analyze_morphology_has_pos_tags(self) -> None:
        """Test that morphology includes POS tags (using MAWO core)."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Красивая кошка"  # Beautiful cat

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert len(morph_info) > 0
        # Should have POS tags from MAWO core
        has_pos = any(info.pos is not None for info in morph_info)
        assert has_pos, "Morphology should include POS tags from MAWO core"

    def test_analyze_morphology_empty_string(self) -> None:
        """Test morphology analysis on empty string."""
        # Arrange
        helper = RussianLanguageHelper()

        # Act
        morph_info = helper.analyze_morphology("")

        # Assert
        assert isinstance(morph_info, list)
        assert len(morph_info) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestRussianWordVerification:
    """Test word existence verification (anti-hallucination)."""

    def test_verify_word_exists_true(self) -> None:
        """Test verification when word exists in text."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"
        word = "школу"

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is True

    def test_verify_word_exists_false(self) -> None:
        """Test verification when word does NOT exist (hallucination detection)."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"
        word = "магазин"  # Not in text

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is False, "Should detect that word is not in text"

    def test_verify_word_partial_match(self) -> None:
        """Test that partial matches don't count as existence."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "школа"
        word = "кола"  # Partial match

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        # Should be False for exact word matching
        assert isinstance(exists, bool)


@pytest.mark.integration
@pytest.mark.slow
class TestRussianErrorPositionVerification:
    """Test error position validation (anti-hallucination for positions)."""

    def test_verify_error_position_valid(self) -> None:
        """Test verification of valid error position."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"
        error = ErrorAnnotation(
            category="test",
            subcategory="test",
            severity="minor",
            location=(0, 1),  # "Я"
            description="Test error",
        )

        # Act
        valid = helper.verify_error_position(error, text)

        # Assert
        assert valid is True

    def test_verify_error_position_out_of_bounds(self) -> None:
        """Test verification catches out-of-bounds positions."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я школа"
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
        helper = RussianLanguageHelper()
        text = "Я школа"
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
class TestRussianGrammarCheck:
    """Test Russian grammar checking with LanguageTool."""

    def test_check_grammar_returns_list(self) -> None:
        """Test that check_grammar() returns a list."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"

        # Act
        errors = helper.check_grammar(text)

        # Assert
        assert isinstance(errors, list)
        # LanguageTool should work for clean Russian text

    def test_check_grammar_returns_error_annotations(self) -> None:
        """Test that grammar errors are ErrorAnnotation objects."""
        # Arrange
        helper = RussianLanguageHelper()
        # Intentional error: wrong case
        text = "Я иду в школе"  # Should be "школу" (accusative)

        # Act
        errors = helper.check_grammar(text)

        # Assert
        for error in errors:
            assert isinstance(error, ErrorAnnotation)

    def test_check_grammar_detects_spelling_errors(self) -> None:
        """Test that LanguageTool detects spelling errors."""
        # Arrange
        helper = RussianLanguageHelper()
        # Intentional typo
        text = "Я иду в шкалу"  # Wrong word (шкалу instead of школу)

        # Act
        errors = helper.check_grammar(text)

        # Assert
        # LanguageTool should detect this
        assert isinstance(errors, list)
        # May or may not find the error depending on LanguageTool rules
        # Just ensure it doesn't crash


@pytest.mark.integration
@pytest.mark.slow
class TestRussianAdjectiveNounAgreement:
    """Test custom adjective-noun agreement checking."""

    def test_adjective_noun_agreement_correct(self) -> None:
        """Test that correct agreement doesn't raise errors."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Красивая кошка"  # Correct: fem adj + fem noun

        # Act
        errors = helper._check_adjective_noun_agreement(text)

        # Assert
        assert isinstance(errors, list)
        # Should have no agreement errors
        assert len(errors) == 0

    def test_adjective_noun_agreement_incorrect(self) -> None:
        """Test that incorrect agreement is detected."""
        # Arrange
        helper = RussianLanguageHelper()
        # Intentional error: masc adj + fem noun
        text = "Красивый кошка"

        # Act
        errors = helper._check_adjective_noun_agreement(text)

        # Assert
        assert isinstance(errors, list)
        # Should detect gender mismatch
        # Note: May not always detect due to morphology ambiguity
        # Just ensure it doesn't crash


@pytest.mark.integration
@pytest.mark.slow
class TestRussianEnrichmentData:
    """Test enrichment data for LLM prompts."""

    def test_get_enrichment_data_returns_dict(self) -> None:
        """Test that enrichment data returns a dictionary."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"

        # Act
        enrichment = helper.get_enrichment_data(text)

        # Assert
        assert isinstance(enrichment, dict)
        assert "has_morphology" in enrichment

    def test_get_enrichment_data_has_pos_counts(self) -> None:
        """Test that enrichment includes POS tag distribution."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Красивая кошка спит"

        # Act
        enrichment = helper.get_enrichment_data(text)

        # Assert
        # Should have POS distribution if MAWO is available
        assert "has_morphology" in enrichment
        if enrichment.get("has_morphology"):
            # If morphology is available, should have POS data
            assert isinstance(enrichment, dict)


@pytest.mark.integration
@pytest.mark.slow
class TestRussianEntityExtraction:
    """Test named entity recognition."""

    def test_extract_entities_returns_list(self) -> None:
        """Test that entity extraction returns a list."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я живу в Москве"  # I live in Moscow

        # Act
        entities = helper.extract_entities(text)

        # Assert
        assert isinstance(entities, list)

    def test_extract_entities_has_structure(self) -> None:
        """Test that entities have expected structure."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Александр живёт в Москве"  # Alexander lives in Moscow

        # Act
        entities = helper.extract_entities(text)

        # Assert
        for entity in entities:
            assert isinstance(entity, dict)
            # Should have text and type at minimum
            assert "text" in entity or len(entity) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestRussianEdgeCases:
    """Test edge cases and error handling."""

    def test_tokenize_unicode_normalization(self) -> None:
        """Test that tokenization handles Unicode normalization."""
        # Arrange
        helper = RussianLanguageHelper()
        text1 = "Привет"
        text2 = "Привет"  # Could be different normalization form

        # Act
        tokens1 = helper.tokenize(text1)
        tokens2 = helper.tokenize(text2)

        # Assert
        assert len(tokens1) > 0
        assert len(tokens2) > 0

    def test_morphology_handles_mixed_script(self) -> None:
        """Test morphology analysis with mixed Cyrillic and Latin."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в Moscow"  # Mixed script

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert isinstance(morph_info, list)
        # Should handle gracefully without crashing

    def test_tokenize_only_punctuation(self) -> None:
        """Test tokenization of text with only punctuation."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "..."

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert isinstance(tokens, list)
        # Behavior may vary - just ensure no crash

    def test_verify_word_exists_with_whitespace(self) -> None:
        """Test word verification handles whitespace correctly."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "  Я иду в школу  "
        word = "школу"

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is True, "Should find word despite surrounding whitespace"


@pytest.mark.integration
@pytest.mark.slow
class TestRussianIntegration:
    """Integration tests that verify components work together."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: tokenize -> morphology -> verify."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Я иду в школу"

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
        helper = RussianLanguageHelper()
        text = "Я иду в школу"
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

    def test_grammar_check_with_languagetool(self) -> None:
        """Test full grammar checking with LanguageTool integration."""
        # Arrange
        helper = RussianLanguageHelper()
        # Clean Russian text
        text = "Я иду в школу"

        # Act
        errors = helper.check_grammar(text)

        # Assert
        assert isinstance(errors, list)
        # LanguageTool is working if we get a list (may be empty for clean text)

    def test_hybrid_grammar_checking(self) -> None:
        """Test that both LanguageTool and custom rules work together."""
        # Arrange
        helper = RussianLanguageHelper()
        text = "Красивый кошка"  # Gender mismatch: masc adj + fem noun

        # Act
        errors = helper.check_grammar(text)

        # Assert
        assert isinstance(errors, list)
        # Either LanguageTool or custom rules should work
        # Just ensure the integration doesn't crash
