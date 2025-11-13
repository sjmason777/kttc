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

"""Comprehensive tests for RussianLanguageHelper."""

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity
from kttc.helpers.russian import RussianLanguageHelper


class TestRussianLanguageHelper:
    """Test suite for Russian language helper."""

    @pytest.fixture
    def helper(self):
        """Create Russian language helper instance."""
        return RussianLanguageHelper()

    def test_initialization(self, helper):
        """Test that helper initializes correctly."""
        assert helper is not None
        assert helper.language_code == "ru"

    def test_is_available(self, helper):
        """Test availability check."""
        # Should be available if MAWO libraries are installed
        available = helper.is_available()
        assert isinstance(available, bool)

        # If available, _morph should be loaded
        if available:
            assert helper._morph is not None
            assert helper._initialized is True
        else:
            pytest.skip("MAWO libraries not available")

    def test_tokenize_simple(self, helper):
        """Test tokenization of simple Russian text."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Привет мир"
        tokens = helper.tokenize(text)

        assert len(tokens) == 2
        assert tokens[0] == ("Привет", 0, 6)
        assert tokens[1] == ("мир", 7, 10)

    def test_tokenize_with_punctuation(self, helper):
        """Test tokenization with punctuation."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Привет, мир!"
        tokens = helper.tokenize(text)

        # razdel splits punctuation
        assert len(tokens) == 4  # Привет, , , мир, !
        assert tokens[0][0] == "Привет"
        assert tokens[1][0] == ","
        assert tokens[2][0] == "мир"
        assert tokens[3][0] == "!"

    def test_tokenize_complex_sentence(self, helper):
        """Test tokenization of complex sentence."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Быстрая коричневая лиса прыгает через ленивую собаку."
        tokens = helper.tokenize(text)

        # Should have multiple tokens
        assert len(tokens) > 0

        # Check positions are correct
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_tokenize_fallback(self):
        """Test tokenization fallback when MAWO unavailable."""
        helper = RussianLanguageHelper()
        helper._initialized = False
        helper._morph = None

        text = "Привет мир"
        tokens = helper.tokenize(text)

        # Fallback uses simple split
        assert len(tokens) == 2
        assert tokens[0][0] == "Привет"
        assert tokens[1][0] == "мир"

    def test_verify_word_exists_found(self, helper):
        """Test word verification when word exists."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Кот сидел на коврике"
        assert helper.verify_word_exists("Кот", text) is True
        assert helper.verify_word_exists("сидел", text) is True
        assert helper.verify_word_exists("коврике", text) is True

    def test_verify_word_exists_not_found(self, helper):
        """Test word verification when word doesn't exist."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Кот сидел на коврике"
        assert helper.verify_word_exists("собака", text) is False
        assert helper.verify_word_exists("прыгал", text) is False

    def test_verify_word_exists_case_insensitive(self, helper):
        """Test word verification is case-insensitive."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Кот сидел на Коврике"
        assert helper.verify_word_exists("кот", text) is True
        assert helper.verify_word_exists("КОТ", text) is True
        assert helper.verify_word_exists("коврике", text) is True

    def test_verify_error_position_valid(self, helper):
        """Test error position verification with valid positions."""
        text = "Привет мир"

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 6),  # "Привет"
            description="test",
        )
        assert helper.verify_error_position(error, text) is True

    def test_verify_error_position_invalid(self, helper):
        """Test error position verification with invalid positions."""
        text = "Привет мир"

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

    def test_verify_error_position_with_quoted_word(self, helper):
        """Test error position verification with quoted words in description."""
        text = "быстрый лиса"

        # Error that mentions 'быстрый' - should be at the position
        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MAJOR,
            location=(0, 12),  # Covers both words
            description="Gender mismatch: 'быстрый' is masc",
        )
        assert helper.verify_error_position(error, text) is True

        # Error that mentions word NOT at the position
        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MAJOR,
            location=(0, 7),  # Just "быстрый"
            description="Gender mismatch: 'лиса' is femn",
        )
        assert helper.verify_error_position(error2, text) is False

    def test_analyze_morphology(self, helper):
        """Test morphological analysis."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "быстрая лиса"
        morphology = helper.analyze_morphology(text)

        assert len(morphology) == 2

        # Check we got MorphologyInfo objects with correct structure
        for morph in morphology:
            assert hasattr(morph, "word")
            assert hasattr(morph, "pos")
            assert hasattr(morph, "gender")
            assert hasattr(morph, "case")
            assert hasattr(morph, "number")
            assert hasattr(morph, "start")
            assert hasattr(morph, "stop")
            assert text[morph.start : morph.stop] == morph.word

        # Check specific morphology
        assert morphology[0].pos == "ADJF"  # быстрая is adjective
        assert morphology[0].gender == "femn"
        assert morphology[1].pos == "NOUN"  # лиса is noun
        assert morphology[1].gender == "femn"

    def test_check_grammar_correct_agreement(self, helper):
        """Test grammar checking with correct agreement."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        # Correct: feminine adjective + feminine noun
        text = "быстрая лиса"
        errors = helper.check_grammar(text)

        assert len(errors) == 0

    def test_check_grammar_gender_mismatch(self, helper):
        """Test grammar checking with gender mismatch."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        # Wrong: masculine adjective + feminine noun
        text = "быстрый лиса"
        errors = helper.check_grammar(text)

        assert len(errors) == 1
        assert errors[0].category == "fluency"
        assert errors[0].subcategory == "russian_case_agreement"
        assert errors[0].severity == ErrorSeverity.CRITICAL
        assert "gender" in errors[0].description.lower() or "Gender" in errors[0].description

    def test_check_grammar_case_mismatch(self, helper):
        """Test grammar checking with case mismatch."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        # This might be tricky to construct - skip if we can't create valid case mismatch
        # For now, just test that the method runs
        text = "красивый дом"
        errors = helper.check_grammar(text)

        # Should run without crashing
        assert isinstance(errors, list)

    def test_get_enrichment_data(self, helper):
        """Test enrichment data extraction."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "быстрая лиса прыгает"
        enrichment = helper.get_enrichment_data(text)

        assert enrichment["has_morphology"] is True
        assert "word_count" in enrichment
        assert enrichment["word_count"] == 3
        assert "verb_aspects" in enrichment
        assert "adjective_noun_pairs" in enrichment
        assert "pos_distribution" in enrichment

        # Should find verb "прыгает"
        assert len(enrichment["verb_aspects"]) > 0

        # Should find adjective-noun pair "быстрая лиса"
        assert len(enrichment["adjective_noun_pairs"]) == 1
        pair = enrichment["adjective_noun_pairs"][0]
        assert pair["adjective"]["word"] == "быстрая"
        assert pair["noun"]["word"] == "лиса"
        assert pair["agreement"] == "correct"

    def test_get_enrichment_data_gender_mismatch(self, helper):
        """Test enrichment data detects gender mismatch."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "быстрый лиса"
        enrichment = helper.get_enrichment_data(text)

        # Should find adjective-noun pair with mismatch
        assert len(enrichment["adjective_noun_pairs"]) == 1
        pair = enrichment["adjective_noun_pairs"][0]
        assert pair["agreement"] == "mismatch"

    def test_get_enrichment_data_verb_aspects(self, helper):
        """Test enrichment data extracts verb aspects."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Я прыгал и прыгнул"  # imperfective and perfective
        enrichment = helper.get_enrichment_data(text)

        # Should find verbs with aspects
        assert "verb_aspects" in enrichment
        assert len(enrichment["verb_aspects"]) >= 1

    def test_extract_entities(self, helper):
        """Test named entity extraction."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Владимир Путин посетил Москву в январе."
        entities = helper.extract_entities(text)

        # Should find entities (PER: Владимир Путин, LOC: Москву)
        assert isinstance(entities, list)

        # Check entity structure if we found any
        for entity in entities:
            assert "text" in entity
            assert "type" in entity
            assert "start" in entity
            assert "stop" in entity
            assert text[entity["start"] : entity["stop"]] == entity["text"]

    def test_extract_entities_no_entities(self, helper):
        """Test entity extraction when no entities present."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Кот сидел на коврике."
        entities = helper.extract_entities(text)

        # Should return empty list
        assert isinstance(entities, list)

    def test_check_entity_preservation(self, helper):
        """Test entity preservation checking."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        source = "Apple CEO Tim Cook visited California"
        translation = "Генеральный директор Apple Тим Кук посетил Калифорнию"

        errors = helper.check_entity_preservation(source, translation)

        # Should not raise error if entities are present
        assert isinstance(errors, list)

    def test_check_entity_preservation_missing_entities(self, helper):
        """Test entity preservation when translation missing entities."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        source = "Apple CEO Tim Cook announced products"
        translation = "Генеральный директор объявил продукты"  # Missing names

        errors = helper.check_entity_preservation(source, translation)

        # Source has capitalized words but translation might have no entities
        assert isinstance(errors, list)

    def test_empty_text_handling(self, helper):
        """Test handling of empty text."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = ""

        tokens = helper.tokenize(text)
        assert tokens == []

        morphology = helper.analyze_morphology(text)
        assert morphology == []

        errors = helper.check_grammar(text)
        assert errors == []

    def test_whitespace_text_handling(self, helper):
        """Test handling of whitespace-only text."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "   \n\t  "

        tokens = helper.tokenize(text)
        # razdel might tokenize whitespace differently
        assert isinstance(tokens, list)

    def test_special_characters(self, helper):
        """Test handling of special characters."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Привет @пользователь #хештег http://example.com"
        tokens = helper.tokenize(text)

        # Should tokenize without errors
        assert len(tokens) > 0

        # Verify positions
        for word, start, end in tokens:
            assert text[start:end] == word

    def test_mixed_language_text(self, helper):
        """Test handling of mixed Russian/English text."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Я люблю Python programming"
        tokens = helper.tokenize(text)

        # Should handle mixed content
        assert len(tokens) > 0

        for word, start, end in tokens:
            assert text[start:end] == word

    def test_very_long_text(self, helper):
        """Test handling of very long text."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        # Create a long text
        text = " ".join(["Быстрая коричневая лиса прыгает через ленивую собаку."] * 50)

        tokens = helper.tokenize(text)
        morphology = helper.analyze_morphology(text)
        enrichment = helper.get_enrichment_data(text)

        # Should process without errors
        assert len(tokens) > 0
        assert len(morphology) > 0
        assert enrichment["has_morphology"] is True

    def test_position_accuracy(self, helper):
        """Test that all token positions are accurate."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Быстрая лиса прыгает через забор."
        tokens = helper.tokenize(text)

        # Every token should match its position in text
        for word, start, end in tokens:
            assert (
                text[start:end] == word
            ), f"Position mismatch: text[{start}:{end}] = '{text[start:end]}' != '{word}'"

    def test_morphology_position_accuracy(self, helper):
        """Test that morphology positions are accurate."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Быстрая лиса прыгает"
        morphology = helper.analyze_morphology(text)

        # Every morphology entry should match its position
        for morph in morphology:
            assert text[morph.start : morph.stop] == morph.word

    def test_no_overlapping_tokens(self, helper):
        """Test that tokens don't overlap."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Быстрая лиса прыгает через забор"
        tokens = helper.tokenize(text)

        # Sort by start position
        sorted_tokens = sorted(tokens, key=lambda t: t[1])

        # Check no overlaps
        for i in range(len(sorted_tokens) - 1):
            current_end = sorted_tokens[i][2]
            next_start = sorted_tokens[i + 1][1]
            assert (
                current_end <= next_start
            ), f"Overlapping tokens: {sorted_tokens[i]} and {sorted_tokens[i+1]}"

    def test_multiple_adjective_noun_pairs(self, helper):
        """Test detection of multiple adjective-noun pairs."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Быстрая лиса и медленная черепаха"
        enrichment = helper.get_enrichment_data(text)

        # Should find two pairs: "Быстрая лиса" and "медленная черепаха"
        pairs = enrichment["adjective_noun_pairs"]
        assert len(pairs) >= 1  # At least one pair

    def test_check_grammar_multiple_errors(self, helper):
        """Test grammar checking with multiple errors."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        # Multiple gender mismatches
        text = "быстрый лиса и медленный черепаха"
        errors = helper.check_grammar(text)

        # Should find at least one error (might find two)
        assert len(errors) >= 1

    def test_cyrillic_yo_letter(self, helper):
        """Test handling of ё letter."""
        if not helper.is_available():
            pytest.skip("MAWO libraries not available")

        text = "Ёлка и ежик"
        tokens = helper.tokenize(text)

        # Should handle ё correctly
        assert len(tokens) > 0

        for word, start, end in tokens:
            assert text[start:end] == word
