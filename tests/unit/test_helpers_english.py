# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive tests for EnglishLanguageHelper."""

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity
from kttc.helpers.english import EnglishLanguageHelper


class TestEnglishLanguageHelper:
    """Test suite for English language helper."""

    @pytest.fixture
    def helper(self):
        """Create English language helper instance."""
        return EnglishLanguageHelper()

    def test_initialization(self, helper):
        """Test that helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "en"

    def test_is_available(self, helper):
        """Test availability check."""
        # Should be available if spaCy is installed
        available = helper.is_available()
        assert isinstance(available, bool)

        # If available, _nlp should be loaded
        if available:
            assert helper._nlp is not None
            assert helper._initialized is True
        else:
            # If not available, log why
            pytest.skip(
                "spaCy not available - install with: python -m spacy download en_core_web_md"
            )

    def test_tokenize_simple(self, helper):
        """Test tokenization of simple English text."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "Hello world"
        tokens = helper.tokenize(text)

        assert len(tokens) == 2
        assert tokens[0] == ("Hello", 0, 5)
        assert tokens[1] == ("world", 6, 11)

    def test_tokenize_with_punctuation(self, helper):
        """Test tokenization with punctuation."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "Hello, world!"
        tokens = helper.tokenize(text)

        # spaCy splits punctuation
        assert len(tokens) == 4  # Hello, , , world, !
        assert tokens[0][0] == "Hello"
        assert tokens[1][0] == ","
        assert tokens[2][0] == "world"
        assert tokens[3][0] == "!"

    def test_tokenize_complex_sentence(self, helper):
        """Test tokenization of complex sentence."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The quick brown fox jumps over the lazy dog."
        tokens = helper.tokenize(text)

        # Should have 10 tokens (9 words + 1 period)
        assert len(tokens) == 10

        # Check positions are correct
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_tokenize_fallback(self):
        """Test tokenization fallback when spaCy unavailable."""
        helper = EnglishLanguageHelper()
        helper._initialized = False
        helper._nlp = None

        text = "Hello world"
        tokens = helper.tokenize(text)

        # Fallback uses simple split
        assert len(tokens) == 2
        assert tokens[0][0] == "Hello"
        assert tokens[1][0] == "world"

    def test_verify_word_exists_found(self, helper):
        """Test word verification when word exists."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The cat sat on the mat"
        assert helper.verify_word_exists("cat", text) is True
        assert helper.verify_word_exists("sat", text) is True
        assert helper.verify_word_exists("mat", text) is True

    def test_verify_word_exists_not_found(self, helper):
        """Test word verification when word doesn't exist."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The cat sat on the mat"
        assert helper.verify_word_exists("dog", text) is False
        assert helper.verify_word_exists("jumped", text) is False

    def test_verify_word_exists_case_insensitive(self, helper):
        """Test word verification is case-insensitive."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The Cat sat on the Mat"
        assert helper.verify_word_exists("cat", text) is True
        assert helper.verify_word_exists("CAT", text) is True
        assert helper.verify_word_exists("Mat", text) is True

    def test_verify_error_position_valid(self, helper):
        """Test error position verification with valid positions."""
        text = "Hello world"

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),  # "Hello"
            description="test",
        )
        assert helper.verify_error_position(error, text) is True

        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(6, 11),  # "world"
            description="test",
        )
        assert helper.verify_error_position(error2, text) is True

    def test_verify_error_position_invalid(self, helper):
        """Test error position verification with invalid positions."""
        text = "Hello world"

        # Out of bounds
        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 100),
            description="test",
        )
        assert helper.verify_error_position(error1, text) is False

        # Negative start
        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(-1, 5),
            description="test",
        )
        assert helper.verify_error_position(error2, text) is False

        # Start >= end
        error3 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(5, 5),
            description="test",
        )
        assert helper.verify_error_position(error3, text) is False

        # Empty substring
        error4 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(5, 6),  # Just the space
            description="test",
        )
        assert helper.verify_error_position(error4, text) is False

    def test_analyze_morphology(self, helper):
        """Test morphological analysis."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The cats are running"
        morphology = helper.analyze_morphology(text)

        assert len(morphology) == 4

        # Check we got MorphologyInfo objects with correct structure
        for morph in morphology:
            assert hasattr(morph, "word")
            assert hasattr(morph, "pos")
            assert hasattr(morph, "start")
            assert hasattr(morph, "stop")
            assert text[morph.start : morph.stop] == morph.word

    def test_get_enrichment_data(self, helper):
        """Test enrichment data extraction."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "Apple CEO Tim Cook announced new products in California."
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["has_morphology"] is True
        assert "word_count" in enrichment
        assert enrichment["word_count"] > 0
        assert "pos_distribution" in enrichment
        assert "entities" in enrichment
        assert "sentence_count" in enrichment
        assert enrichment["sentence_count"] == 1

    def test_get_enrichment_data_multiple_sentences(self, helper):
        """Test enrichment data with multiple sentences."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "This is sentence one. This is sentence two. And sentence three."
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["sentence_count"] == 3

    def test_extract_entities(self, helper):
        """Test named entity extraction."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "Apple CEO Tim Cook announced new products in California."
        entities = helper.extract_entities(text)

        # Should find at least some entities (Apple, Tim Cook, California)
        assert len(entities) > 0

        # Check entity structure
        for entity in entities:
            assert "text" in entity
            assert "type" in entity
            assert "start" in entity
            assert "stop" in entity
            assert text[entity["start"] : entity["stop"]] == entity["text"]

    def test_extract_entities_no_entities(self, helper):
        """Test entity extraction when no entities present."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The cat sat on the mat."
        entities = helper.extract_entities(text)

        # Should return empty list
        assert entities == []

    def test_check_entity_preservation_both_have_entities(self, helper):
        """Test entity preservation when both texts have entities."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        source = "Apple CEO Tim Cook visited California."
        translation = "Apple CEO Tim Cook visited California."

        errors = helper.check_entity_preservation(source, translation)

        # Both have entities, should not raise error
        assert len(errors) == 0

    def test_check_entity_preservation_missing_entities(self, helper):
        """Test entity preservation when translation missing entities."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        source = "Apple CEO Tim Cook visited California."
        translation = "The chief executive visited the state."

        errors = helper.check_entity_preservation(source, translation)

        # Source has capitalized words but translation has no entities
        assert len(errors) > 0
        assert errors[0].category == "accuracy"
        assert errors[0].subcategory == "entity_omission"

    def test_check_entity_preservation_no_source_entities(self, helper):
        """Test entity preservation when source has no entities."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        source = "the cat sat on the mat"
        translation = "the cat sat on the mat"

        errors = helper.check_entity_preservation(source, translation)

        # No entities in source, should not raise error
        assert len(errors) == 0

    def test_check_grammar_no_errors(self, helper):
        """Test grammar checking with correct text."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "The cat sits on the mat."
        errors = helper.check_grammar(text)

        # Correct text should have no errors
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_check_grammar_subject_verb_agreement(self, helper):
        """Test detection of subject-verb agreement error."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "He go to school every day"  # Should be "goes"
        errors = helper.check_grammar(text)

        # Should find at least one error
        assert len(errors) > 0
        # Error should be about grammar/agreement (LanguageTool uses "AGR" for agreement)
        assert any(
            "agr" in err.subcategory.lower() or "verb" in err.description.lower() for err in errors
        )

    def test_check_grammar_article_error(self, helper):
        """Test detection of article error (a vs an)."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "I saw a elephant in the zoo"  # Should be "an elephant"
        errors = helper.check_grammar(text)

        # Should find article error
        assert len(errors) > 0
        assert any(
            "article" in err.subcategory.lower() or "a" in err.description.lower() for err in errors
        )

    def test_check_grammar_with_suggestion(self, helper):
        """Test that errors include suggestions."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "He dont like apples"  # Should be "doesn't"
        errors = helper.check_grammar(text)

        # Should have error with suggestion
        if len(errors) > 0:
            # At least one error should have a suggestion
            assert any(err.suggestion is not None for err in errors)

    def test_check_grammar_severity_mapping(self, helper):
        """Test that severity is mapped correctly."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "He go to scool"  # Grammar + spelling error
        errors = helper.check_grammar(text)

        # Should have errors with severity
        assert len(errors) > 0
        for err in errors:
            assert hasattr(err, "severity")
            assert err.severity in [
                ErrorSeverity.CRITICAL,
                ErrorSeverity.MAJOR,
                ErrorSeverity.MINOR,
            ]

    def test_enrichment_verb_tenses(self, helper):
        """Test enrichment data includes verb tense analysis."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "He walked to school yesterday."
        enrichment = helper.get_enrichment_data(text)

        assert "verb_tenses" in enrichment
        # Should find "walked" verb
        if len(enrichment["verb_tenses"]) > 0:
            assert any("walk" in verb.lower() for verb in enrichment["verb_tenses"])

    def test_enrichment_article_noun_pairs(self, helper):
        """Test enrichment data includes article-noun analysis."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The cat and a dog are friends."
        enrichment = helper.get_enrichment_data(text)

        assert "article_noun_pairs" in enrichment
        # Should find article-noun pairs
        assert len(enrichment["article_noun_pairs"]) > 0

    def test_enrichment_subject_verb_pairs(self, helper):
        """Test enrichment data includes subject-verb analysis."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The cat sits on the mat."
        enrichment = helper.get_enrichment_data(text)

        assert "subject_verb_pairs" in enrichment
        # Should find subject-verb pairs
        if len(enrichment["subject_verb_pairs"]) > 0:
            pair = enrichment["subject_verb_pairs"][0]
            assert "subject" in pair
            assert "verb" in pair

    def test_empty_text_handling(self, helper):
        """Test handling of empty text."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = ""

        tokens = helper.tokenize(text)
        assert tokens == []

        morphology = helper.analyze_morphology(text)
        assert morphology == []

        entities = helper.extract_entities(text)
        assert entities == []

    def test_whitespace_text_handling(self, helper):
        """Test handling of whitespace-only text."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "   \n\t  "

        tokens = helper.tokenize(text)
        # spaCy might tokenize whitespace differently
        assert isinstance(tokens, list)

    def test_special_characters(self, helper):
        """Test handling of special characters."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "Hello @user #hashtag http://example.com"
        tokens = helper.tokenize(text)

        # Should tokenize without errors
        assert len(tokens) > 0

        # Verify positions
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_unicode_text(self, helper):
        """Test handling of Unicode characters."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "Hello café naïve résumé"
        tokens = helper.tokenize(text)

        # Should handle Unicode correctly
        assert len(tokens) == 4

        # Verify positions
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_very_long_text(self, helper):
        """Test handling of very long text."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        # Create a long text
        text = " ".join(["The quick brown fox jumps over the lazy dog."] * 100)

        tokens = helper.tokenize(text)
        morphology = helper.analyze_morphology(text)
        enrichment = helper.get_enrichment_data(text)

        # Should process without errors
        assert len(tokens) > 0
        assert len(morphology) > 0
        assert enrichment["has_morphology"] is True
        assert enrichment["sentence_count"] == 100

    def test_check_grammar_when_unavailable(self, helper):
        """Test grammar checking when LanguageTool is not available."""
        # Temporarily disable LanguageTool
        original_lt_available = helper._lt_available
        helper._lt_available = False

        text = "He go to school every day"
        errors = helper.check_grammar(text)

        # Should return empty list gracefully
        assert errors == []

        # Restore original state
        helper._lt_available = original_lt_available

    def test_tokenize_with_numbers(self, helper):
        """Test tokenization with numbers and mixed content."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "In 2024, we sold 1,000 units for $50.99 each."
        tokens = helper.tokenize(text)

        # Should tokenize numbers correctly
        assert len(tokens) > 0
        # Verify positions are accurate
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_enrichment_data_with_contractions(self, helper):
        """Test enrichment data with contractions."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "I'm going to the store. He's already been there. They've finished."
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["has_morphology"] is True
        assert enrichment["sentence_count"] == 3

    def test_morphology_with_negations(self, helper):
        """Test morphology analysis with negations."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "He doesn't like apples."
        morphology = helper.analyze_morphology(text)

        # Should analyze contractions correctly
        assert len(morphology) > 0
        # Check that contractions are tokenized
        words = [m.word for m in morphology]
        assert "does" in words or "doesn't" in words

    def test_verify_word_exists_fallback(self):
        """Test word verification fallback when spaCy unavailable."""
        helper = EnglishLanguageHelper()
        helper._initialized = False
        helper._nlp = None

        text = "The cat sat on the mat"
        # Should use simple case-insensitive search
        assert helper.verify_word_exists("cat", text) is True
        assert helper.verify_word_exists("CAT", text) is True
        assert helper.verify_word_exists("dog", text) is False

    def test_extract_entities_when_unavailable(self, helper):
        """Test entity extraction when spaCy is not available."""
        # Temporarily disable spaCy
        original_initialized = helper._initialized
        original_nlp = helper._nlp
        helper._initialized = False
        helper._nlp = None

        text = "Apple CEO Tim Cook visited California."
        entities = helper.extract_entities(text)

        # Should return empty list gracefully
        assert entities == []

        # Restore original state
        helper._initialized = original_initialized
        helper._nlp = original_nlp

    def test_check_entity_preservation_when_unavailable(self, helper):
        """Test entity preservation check when spaCy is not available."""
        # Temporarily disable spaCy
        original_initialized = helper._initialized
        original_nlp = helper._nlp
        helper._initialized = False
        helper._nlp = None

        source = "Apple CEO Tim Cook visited California."
        translation = "The CEO visited the state."
        errors = helper.check_entity_preservation(source, translation)

        # Should return empty list gracefully
        assert errors == []

        # Restore original state
        helper._initialized = original_initialized
        helper._nlp = original_nlp

    def test_analyze_morphology_when_unavailable(self):
        """Test morphology analysis when spaCy is not available."""
        helper = EnglishLanguageHelper()
        helper._initialized = False
        helper._nlp = None

        text = "The cat sits on the mat."
        morphology = helper.analyze_morphology(text)

        # Should return empty list gracefully
        assert morphology == []

    def test_get_enrichment_data_when_unavailable(self):
        """Test enrichment data when spaCy is not available."""
        helper = EnglishLanguageHelper()
        helper._initialized = False
        helper._nlp = None

        text = "The cat sits on the mat."
        enrichment = helper.get_enrichment_data(text)

        # Should return minimal data gracefully
        assert enrichment == {"has_morphology": False}

    def test_check_grammar_multiple_errors(self, helper):
        """Test grammar checking with multiple errors in one text."""
        if not helper._lt_available:
            pytest.skip("LanguageTool not available")

        text = "He dont like apples and he go to school every day"
        errors = helper.check_grammar(text)

        # Should find multiple errors (dont -> doesn't, go -> goes)
        assert len(errors) >= 2

    def test_enrichment_data_complex_sentence(self, helper):
        """Test enrichment data with complex sentence structure."""
        if not helper.is_available():
            pytest.skip("spaCy not available")

        text = "The student who was studying hard passed the exam with flying colors."
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["has_morphology"] is True
        assert "verb_tenses" in enrichment
        assert "pos_distribution" in enrichment
        # Should detect relative clause and complex structure
        assert enrichment["word_count"] > 10
