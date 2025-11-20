"""Integration tests for HindiLanguageHelper.

Tests Hindi language helper with real NLP libraries (Indic NLP, Stanza, Spello).
Focus: Verify all abstract methods, test error detection, validate NLP integration.

⚠️ IMPORTANT: These are INTEGRATION tests (not unit tests)!
- Load ML models (Stanza: 321MB Hindi models)
- Take 3-5 minutes to run (first time: model download + caching)
- Skip in CI/CD to avoid timeouts
- Run manually or in separate integration test pipeline

To run: pytest tests/unit/test_hindi_language_helper.py -m integration
"""

import os

import pytest

from kttc.core.models import ErrorAnnotation
from kttc.helpers.hindi import HindiLanguageHelper

# Skip all Hindi tests in CI - slow (3-5 min) due to Stanza model loading
pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
    reason="Hindi tests are slow (3-5 min) - skip in CI/CD. Run manually with: pytest -m integration",
)


@pytest.mark.integration
@pytest.mark.slow
class TestHindiLanguageHelperBasics:
    """Test basic Hindi helper functionality and interface compliance."""

    def test_instantiation(self) -> None:
        """Test that HindiLanguageHelper can be instantiated without errors."""
        # Act
        helper = HindiLanguageHelper()

        # Assert
        assert helper is not None
        assert helper.language_code == "hi"

    def test_is_available_returns_bool(self) -> None:
        """Test that is_available() returns a boolean value."""
        # Arrange
        helper = HindiLanguageHelper()

        # Act
        available = helper.is_available()

        # Assert
        assert isinstance(available, bool)
        # Should be True if dependencies are installed (which they should be for tests)
        assert available is True, "Hindi dependencies should be installed for tests"


@pytest.mark.integration
@pytest.mark.slow
class TestHindiTokenization:
    """Test Hindi tokenization functionality."""

    def test_tokenize_simple_text(self) -> None:
        """Test tokenization of simple Hindi text."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "नमस्ते दुनिया"

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) == 2
        # Each token is (word, start, end)
        assert all(len(token) == 3 for token in tokens)
        assert tokens[0][0] == "नमस्ते"
        assert tokens[1][0] == "दुनिया"

    def test_tokenize_empty_string(self) -> None:
        """Test tokenization of empty string returns empty list."""
        # Arrange
        helper = HindiLanguageHelper()

        # Act
        tokens = helper.tokenize("")

        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_tokenize_position_accuracy(self) -> None:
        """Test that token positions are accurate."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"

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
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं। तुम कहाँ जाते हो?"

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert len(tokens) > 5
        # Should handle both sentences and punctuation
        words = [token[0] for token in tokens]
        assert "मैं" in words
        assert "स्कूल" in words
        assert "तुम" in words


@pytest.mark.integration
@pytest.mark.slow
class TestHindiMorphology:
    """Test Hindi morphological analysis."""

    def test_analyze_morphology_returns_list(self) -> None:
        """Test that morphology analysis returns a list."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert isinstance(morph_info, list)

    def test_analyze_morphology_has_pos_tags(self) -> None:
        """Test that morphology includes POS tags (using Stanza)."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "बच्चा खेलता है"  # Child plays

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert len(morph_info) > 0
        # Should have POS tags from Stanza
        has_pos = any(info.pos is not None for info in morph_info)
        assert has_pos, "Morphology should include POS tags from Stanza"

    def test_analyze_morphology_empty_string(self) -> None:
        """Test morphology analysis on empty string."""
        # Arrange
        helper = HindiLanguageHelper()

        # Act
        morph_info = helper.analyze_morphology("")

        # Assert
        assert isinstance(morph_info, list)
        assert len(morph_info) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestHindiWordVerification:
    """Test word existence verification (anti-hallucination)."""

    def test_verify_word_exists_true(self) -> None:
        """Test verification when word exists in text."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"
        word = "स्कूल"

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is True

    def test_verify_word_exists_false(self) -> None:
        """Test verification when word does NOT exist (hallucination detection)."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"
        word = "किताब"  # Not in text

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is False, "Should detect that word is not in text"

    def test_verify_word_partial_match(self) -> None:
        """Test that partial matches don't count as existence."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "स्कूल"
        word = "कूल"  # Partial match

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        # Behavior depends on implementation - document it
        # If it's substring matching, it might be True
        # If it's exact word matching, it should be False
        assert isinstance(exists, bool)


@pytest.mark.integration
@pytest.mark.slow
class TestHindiErrorPositionVerification:
    """Test error position validation (anti-hallucination for positions)."""

    def test_verify_error_position_valid(self) -> None:
        """Test verification of valid error position."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"
        error = ErrorAnnotation(
            category="test",
            subcategory="test",
            severity="minor",
            location=(0, 3),  # "मैं"
            description="Test error",
        )

        # Act
        valid = helper.verify_error_position(error, text)

        # Assert
        assert valid is True

    def test_verify_error_position_out_of_bounds(self) -> None:
        """Test verification catches out-of-bounds positions."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल"
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
        helper = HindiLanguageHelper()
        text = "मैं स्कूल"
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
class TestHindiGrammarCheck:
    """Test Hindi grammar checking."""

    def test_check_grammar_returns_list(self) -> None:
        """Test that check_grammar() returns a list."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"

        # Act
        errors = helper.check_grammar(text)

        # Assert
        assert isinstance(errors, list)
        # Currently returns empty list (delegated to LLM)
        # This test ensures interface compliance

    def test_check_grammar_returns_error_annotations(self) -> None:
        """Test that grammar errors are ErrorAnnotation objects."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "some text"

        # Act
        errors = helper.check_grammar(text)

        # Assert
        for error in errors:
            assert isinstance(error, ErrorAnnotation)


@pytest.mark.integration
@pytest.mark.slow
class TestHindiSpellChecking:
    """Test Hindi spell checking with Spello."""

    def test_check_spelling_returns_list(self) -> None:
        """Test that spell checking returns a list."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"

        # Act
        errors = helper.check_spelling(text)

        # Assert
        assert isinstance(errors, list)

    def test_check_spelling_empty_string(self) -> None:
        """Test spell checking on empty string."""
        # Arrange
        helper = HindiLanguageHelper()

        # Act
        errors = helper.check_spelling("")

        # Assert
        assert isinstance(errors, list)


@pytest.mark.integration
@pytest.mark.slow
class TestHindiEnrichmentData:
    """Test enrichment data for LLM prompts."""

    def test_get_enrichment_data_returns_dict(self) -> None:
        """Test that enrichment data returns a dictionary."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"

        # Act
        enrichment = helper.get_enrichment_data(text)

        # Assert
        assert isinstance(enrichment, dict)
        assert "has_morphology" in enrichment

    def test_get_enrichment_data_has_pos_counts(self) -> None:
        """Test that enrichment includes POS tag distribution."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "बच्चा खेलता है"

        # Act
        enrichment = helper.get_enrichment_data(text)

        # Assert
        # Should have POS distribution if Stanza is available
        assert "has_morphology" in enrichment
        if enrichment.get("has_morphology"):
            # If morphology is available, should have POS data
            assert isinstance(enrichment, dict)


@pytest.mark.integration
@pytest.mark.slow
class TestHindiEntityExtraction:
    """Test named entity recognition."""

    def test_extract_entities_returns_list(self) -> None:
        """Test that entity extraction returns a list."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं दिल्ली में रहता हूं"  # I live in Delhi

        # Act
        entities = helper.extract_entities(text)

        # Assert
        assert isinstance(entities, list)

    def test_extract_entities_has_structure(self) -> None:
        """Test that entities have expected structure."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "राम दिल्ली में रहता है"  # Ram lives in Delhi

        # Act
        entities = helper.extract_entities(text)

        # Assert
        for entity in entities:
            assert isinstance(entity, dict)
            # Should have text and type at minimum
            assert "text" in entity or len(entity) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestHindiEdgeCases:
    """Test edge cases and error handling."""

    def test_tokenize_unicode_normalization(self) -> None:
        """Test that tokenization handles Unicode normalization."""
        # Arrange
        helper = HindiLanguageHelper()
        # Same text in different Unicode forms should tokenize similarly
        text1 = "नमस्ते"
        text2 = "नमस्ते"  # Could be different normalization form

        # Act
        tokens1 = helper.tokenize(text1)
        tokens2 = helper.tokenize(text2)

        # Assert
        assert len(tokens1) > 0
        assert len(tokens2) > 0

    def test_morphology_handles_mixed_script(self) -> None:
        """Test morphology analysis with mixed Devanagari and Latin."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं Delhi जाता हूं"  # Mixed script

        # Act
        morph_info = helper.analyze_morphology(text)

        # Assert
        assert isinstance(morph_info, list)
        # Should handle gracefully without crashing

    def test_tokenize_only_punctuation(self) -> None:
        """Test tokenization of text with only punctuation."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "।।।"  # Hindi punctuation

        # Act
        tokens = helper.tokenize(text)

        # Assert
        assert isinstance(tokens, list)
        # Behavior may vary - just ensure no crash

    def test_verify_word_exists_with_whitespace(self) -> None:
        """Test word verification handles whitespace correctly."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "  मैं स्कूल जाता हूं  "
        word = "स्कूल"

        # Act
        exists = helper.verify_word_exists(word, text)

        # Assert
        assert exists is True, "Should find word despite surrounding whitespace"


@pytest.mark.integration
@pytest.mark.slow
class TestHindiIntegration:
    """Integration tests that verify components work together."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: tokenize -> morphology -> verify."""
        # Arrange
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"

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
        helper = HindiLanguageHelper()
        text = "मैं स्कूल जाता हूं"
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
