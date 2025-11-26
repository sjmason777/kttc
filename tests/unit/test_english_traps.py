"""Unit tests for English traps validator module.

Tests English-specific linguistic trap detection.
"""

import pytest

from kttc.terminology.english_traps import EnglishTrapsValidator


@pytest.mark.unit
class TestEnglishTrapsValidator:
    """Test EnglishTrapsValidator functionality."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_validator_initialization(self, validator: EnglishTrapsValidator) -> None:
        """Test validator initializes correctly."""
        assert validator is not None
        assert hasattr(validator, "_glossaries")
        assert hasattr(validator, "HOMOPHONE_ERROR_PATTERNS")
        assert hasattr(validator, "ADJECTIVE_CATEGORIES")

    def test_is_available(self, validator: EnglishTrapsValidator) -> None:
        """Test is_available method."""
        # May be True or False depending on glossary files
        result = validator.is_available()
        assert isinstance(result, bool)


@pytest.mark.unit
class TestHomophoneDetection:
    """Test homophone detection functionality."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_find_their_theyre_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'their' used instead of 'they're'."""
        text = "Their going to the store"
        errors = validator.find_homophones_in_text(text)

        assert len(errors) > 0
        assert any(e["wrong_word"] == "their" and e["correct_word"] == "they're" for e in errors)

    def test_find_your_youre_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'your' used instead of 'you're'."""
        text = "Your welcome to join us"
        errors = validator.find_homophones_in_text(text)

        assert len(errors) > 0
        assert any(e["wrong_word"] == "your" and e["correct_word"] == "you're" for e in errors)

    def test_find_its_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'its' vs 'it's' error."""
        text = "Its going to rain today"
        errors = validator.find_homophones_in_text(text)

        assert len(errors) > 0
        assert any(e["wrong_word"] == "its" and e["correct_word"] == "it's" for e in errors)

    def test_find_to_too_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'to' used instead of 'too'."""
        text = "This is to much for me"
        errors = validator.find_homophones_in_text(text)

        assert len(errors) > 0
        assert any(e["wrong_word"] == "to" and e["correct_word"] == "too" for e in errors)

    def test_find_affect_effect_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'affect' used instead of 'effect'."""
        text = "The affect was dramatic"
        errors = validator.find_homophones_in_text(text)

        assert len(errors) > 0
        assert any(e["wrong_word"] == "affect" and e["correct_word"] == "effect" for e in errors)

    def test_find_loose_lose_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'loose' used instead of 'lose'."""
        text = "Don't loose the game"
        errors = validator.find_homophones_in_text(text)

        assert len(errors) > 0
        assert any(e["wrong_word"] == "loose" and e["correct_word"] == "lose" for e in errors)

    def test_no_false_positives_correct_text(self, validator: EnglishTrapsValidator) -> None:
        """Test no errors in correct text."""
        text = "They're going to their house over there"
        errors = validator.find_homophones_in_text(text)

        # This should not detect "their house" as an error
        # because "their" before "house" is correct
        their_theyre_errors = [e for e in errors if e.get("wrong_word") == "their"]
        # The specific pattern we test is "their going/coming/doing" etc.
        assert not any(
            "their going" in e.get("found_text", "").lower() for e in their_theyre_errors
        )

    def test_error_has_required_fields(self, validator: EnglishTrapsValidator) -> None:
        """Test that errors have all required fields."""
        text = "Their going to school"
        errors = validator.find_homophones_in_text(text)

        if errors:
            error = errors[0]
            assert "found_text" in error
            assert "position" in error
            assert "likely_error" in error
            assert "wrong_word" in error
            assert "correct_word" in error
            assert "severity" in error
            assert "suggestion" in error


@pytest.mark.unit
class TestHomophoneContext:
    """Test homophone context checking."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_check_their_context(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'their' requiring context check."""
        text = "I saw their car"
        warnings = validator.check_homophone_context(text)

        assert len(warnings) > 0
        assert any(w["word"] == "their" for w in warnings)

    def test_check_youre_context(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'you're' requiring context check."""
        text = "You're doing great"
        warnings = validator.check_homophone_context(text)

        assert len(warnings) > 0
        assert any(w["word"] == "you're" for w in warnings)

    def test_check_its_context(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting 'its' requiring context check."""
        text = "The cat licked its paw"
        warnings = validator.check_homophone_context(text)

        assert len(warnings) > 0
        assert any(w["word"] == "its" for w in warnings)

    def test_check_to_too_two_context(self, validator: EnglishTrapsValidator) -> None:
        """Test detecting to/too/two requiring context check."""
        text = "I have two books to read, too"
        warnings = validator.check_homophone_context(text)

        assert any(w["word"] == "to" for w in warnings)
        assert any(w["word"] == "too" for w in warnings)
        assert any(w["word"] == "two" for w in warnings)

    def test_warning_has_required_fields(self, validator: EnglishTrapsValidator) -> None:
        """Test warnings have required fields."""
        text = "Their car is there"
        warnings = validator.check_homophone_context(text)

        if warnings:
            warning = warnings[0]
            assert "word" in warning
            assert "type" in warning
            assert "message" in warning
            assert "severity" in warning


@pytest.mark.unit
class TestGetHomophones:
    """Test get_homophones method."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_get_homophones_returns_dict(self, validator: EnglishTrapsValidator) -> None:
        """Test get_homophones returns dictionary."""
        result = validator.get_homophones()
        assert isinstance(result, dict)

    def test_get_homophones_handles_missing_glossary(
        self, validator: EnglishTrapsValidator
    ) -> None:
        """Test get_homophones handles missing glossary gracefully."""
        # Remove the glossary
        validator._glossaries.pop("homophones", None)
        result = validator.get_homophones()
        assert isinstance(result, dict)


@pytest.mark.unit
class TestGetPhrasalVerbs:
    """Test get_phrasal_verbs method."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_get_phrasal_verbs_returns_dict(self, validator: EnglishTrapsValidator) -> None:
        """Test get_phrasal_verbs returns dictionary."""
        result = validator.get_phrasal_verbs()
        assert isinstance(result, dict)


@pytest.mark.unit
class TestAdjectiveCategories:
    """Test adjective category constants."""

    def test_adjective_categories_exist(self) -> None:
        """Test adjective categories are defined."""
        assert "opinion" in EnglishTrapsValidator.ADJECTIVE_CATEGORIES
        assert "size" in EnglishTrapsValidator.ADJECTIVE_CATEGORIES

    def test_opinion_adjectives(self) -> None:
        """Test opinion adjectives are defined."""
        opinion = EnglishTrapsValidator.ADJECTIVE_CATEGORIES["opinion"]
        assert "beautiful" in opinion
        assert "lovely" in opinion
        assert "ugly" in opinion

    def test_size_adjectives(self) -> None:
        """Test size adjectives are defined."""
        size = EnglishTrapsValidator.ADJECTIVE_CATEGORIES["size"]
        assert isinstance(size, list)
        assert len(size) > 0


@pytest.mark.unit
class TestHomophonePatterns:
    """Test homophone error patterns."""

    def test_patterns_are_valid_regex(self) -> None:
        """Test all patterns are valid regex."""
        import re

        for pattern, wrong, correct in EnglishTrapsValidator.HOMOPHONE_ERROR_PATTERNS:
            # Should not raise
            re.compile(pattern, re.IGNORECASE)

    def test_patterns_have_three_elements(self) -> None:
        """Test each pattern tuple has 3 elements."""
        for item in EnglishTrapsValidator.HOMOPHONE_ERROR_PATTERNS:
            assert len(item) == 3
            pattern, wrong, correct = item
            assert isinstance(pattern, str)
            assert isinstance(wrong, str)
            assert isinstance(correct, str)
