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

        for pattern, _, _ in EnglishTrapsValidator.HOMOPHONE_ERROR_PATTERNS:
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


@pytest.mark.unit
class TestGetAdjectiveCategory:
    """Test get_adjective_category method."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_opinion_category(self, validator: EnglishTrapsValidator) -> None:
        """Test opinion adjectives are categorized correctly."""
        assert validator.get_adjective_category("beautiful") == "opinion"
        assert validator.get_adjective_category("lovely") == "opinion"
        assert validator.get_adjective_category("horrible") == "opinion"

    def test_size_category(self, validator: EnglishTrapsValidator) -> None:
        """Test size adjectives are categorized correctly."""
        assert validator.get_adjective_category("big") == "size"
        assert validator.get_adjective_category("small") == "size"
        assert validator.get_adjective_category("huge") == "size"

    def test_age_category(self, validator: EnglishTrapsValidator) -> None:
        """Test age adjectives are categorized correctly."""
        assert validator.get_adjective_category("old") == "age"
        assert validator.get_adjective_category("young") == "age"
        assert validator.get_adjective_category("new") == "age"

    def test_shape_category(self, validator: EnglishTrapsValidator) -> None:
        """Test shape adjectives are categorized correctly."""
        assert validator.get_adjective_category("round") == "shape"
        assert validator.get_adjective_category("square") == "shape"

    def test_color_category(self, validator: EnglishTrapsValidator) -> None:
        """Test color adjectives are categorized correctly."""
        assert validator.get_adjective_category("red") == "color"
        assert validator.get_adjective_category("blue") == "color"

    def test_origin_category(self, validator: EnglishTrapsValidator) -> None:
        """Test origin adjectives are categorized correctly."""
        assert validator.get_adjective_category("american") == "origin"
        assert validator.get_adjective_category("chinese") == "origin"

    def test_material_category(self, validator: EnglishTrapsValidator) -> None:
        """Test material adjectives are categorized correctly."""
        assert validator.get_adjective_category("wooden") == "material"
        assert validator.get_adjective_category("plastic") == "material"

    def test_unknown_adjective(self, validator: EnglishTrapsValidator) -> None:
        """Test unknown adjective returns None."""
        assert validator.get_adjective_category("xyz123") is None
        assert validator.get_adjective_category("") is None

    def test_case_insensitive(self, validator: EnglishTrapsValidator) -> None:
        """Test case insensitive matching."""
        assert validator.get_adjective_category("Beautiful") == "opinion"
        assert validator.get_adjective_category("BIG") == "size"


@pytest.mark.unit
class TestCheckAdjectiveOrder:
    """Test check_adjective_order method."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_detect_order_violation(self, validator: EnglishTrapsValidator) -> None:
        """Test detection of adjective order violation."""
        # The algorithm matches 3 consecutive words and checks if first two
        # are adjectives in wrong order. Color should come after size.
        # Input: "red big ball" - red (color) big (size) - wrong order
        text = "red big ball"
        violations = validator.check_adjective_order(text)

        assert len(violations) >= 1
        assert violations[0]["severity"] == "major"
        assert "correct_order" in violations[0]

    def test_no_violation_correct_order(self, validator: EnglishTrapsValidator) -> None:
        """Test no violation for correct order."""
        text = "a big red ball"
        violations = validator.check_adjective_order(text)

        # Should not find size-color violations
        size_color_violations = [
            v
            for v in violations
            if "big" in v.get("found_text", "") and "red" in v.get("found_text", "")
        ]
        assert len(size_color_violations) == 0

    def test_no_adjectives(self, validator: EnglishTrapsValidator) -> None:
        """Test no errors when no adjectives."""
        text = "hello world"
        violations = validator.check_adjective_order(text)
        # No adjective violations for non-adjective words
        assert isinstance(violations, list)


@pytest.mark.unit
class TestPrepositionErrors:
    """Test preposition error detection."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_get_preposition_errors(self, validator: EnglishTrapsValidator) -> None:
        """Test getting preposition errors."""
        errors = validator.get_preposition_errors()
        assert isinstance(errors, list)

    def test_find_depend_from_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detection of 'depend from' error."""
        text = "It depend from the weather"
        errors = validator.find_preposition_errors(text)

        assert len(errors) >= 1
        assert errors[0]["correction"] == "depend on"
        assert errors[0]["type"] == "preposition_error"

    def test_find_interested_about_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detection of 'interested about' error."""
        text = "I am interested about this topic"
        errors = validator.find_preposition_errors(text)

        assert len(errors) >= 1
        assert errors[0]["correction"] == "interested in"

    def test_find_listen_the_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detection of 'listen the' error."""
        text = "Please listen the music"
        errors = validator.find_preposition_errors(text)

        assert len(errors) >= 1
        assert "listen to" in errors[0]["correction"]

    def test_find_discuss_about_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detection of 'discuss about' error."""
        text = "Let's discuss about this matter"
        errors = validator.find_preposition_errors(text)

        assert len(errors) >= 1
        assert "discuss" in errors[0]["correction"]

    def test_find_married_with_error(self, validator: EnglishTrapsValidator) -> None:
        """Test detection of 'married with' error."""
        text = "She is married with John"
        errors = validator.find_preposition_errors(text)

        assert len(errors) >= 1
        assert errors[0]["correction"] == "married to"

    def test_no_errors_correct_text(self, validator: EnglishTrapsValidator) -> None:
        """Test no errors in correct text."""
        text = "It depends on the weather"
        errors = validator.find_preposition_errors(text)
        # Should not find "depend on" errors since it's correct
        depend_errors = [e for e in errors if "depend" in e.get("found_text", "")]
        assert len(depend_errors) == 0


@pytest.mark.unit
class TestIdioms:
    """Test idiom detection."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_get_idioms(self, validator: EnglishTrapsValidator) -> None:
        """Test getting idioms from glossary."""
        idioms = validator.get_idioms()
        assert isinstance(idioms, dict)

    def test_find_idioms_returns_list(self, validator: EnglishTrapsValidator) -> None:
        """Test find_idioms_in_text returns list."""
        text = "This is easy"
        found = validator.find_idioms_in_text(text)
        assert isinstance(found, list)


@pytest.mark.unit
class TestHeteronyms:
    """Test heteronym detection."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_get_heteronyms(self, validator: EnglishTrapsValidator) -> None:
        """Test getting heteronyms from glossary."""
        het = validator.get_heteronyms()
        assert isinstance(het, dict)

    def test_find_heteronyms_returns_list(self, validator: EnglishTrapsValidator) -> None:
        """Test find_heteronyms_in_text returns list."""
        text = "The lead singer"
        found = validator.find_heteronyms_in_text(text)
        assert isinstance(found, list)


@pytest.mark.unit
class TestFalseFriends:
    """Test false friends detection."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_get_false_friends_all(self, validator: EnglishTrapsValidator) -> None:
        """Test getting all false friends."""
        ff = validator.get_false_friends()
        assert isinstance(ff, dict)

    def test_get_false_friends_russian(self, validator: EnglishTrapsValidator) -> None:
        """Test getting Russian false friends."""
        ff = validator.get_false_friends(source_lang="ru")
        assert isinstance(ff, dict)

    def test_get_false_friends_spanish(self, validator: EnglishTrapsValidator) -> None:
        """Test getting Spanish false friends."""
        ff = validator.get_false_friends(source_lang="es")
        assert isinstance(ff, dict)

    def test_get_false_friends_german(self, validator: EnglishTrapsValidator) -> None:
        """Test getting German false friends."""
        ff = validator.get_false_friends(source_lang="de")
        assert isinstance(ff, dict)

    def test_get_false_friends_french(self, validator: EnglishTrapsValidator) -> None:
        """Test getting French false friends."""
        ff = validator.get_false_friends(source_lang="fr")
        assert isinstance(ff, dict)

    def test_get_false_friends_unknown_lang(self, validator: EnglishTrapsValidator) -> None:
        """Test getting false friends for unknown language."""
        ff = validator.get_false_friends(source_lang="xx")
        assert isinstance(ff, dict)

    def test_find_false_friends_in_context(self, validator: EnglishTrapsValidator) -> None:
        """Test finding false friends in context."""
        source_text = "Магазин"
        target_text = "I visited the magazine"
        found = validator.find_false_friends_in_context(source_text, target_text, "ru")
        assert isinstance(found, list)


@pytest.mark.unit
class TestComprehensiveAnalysis:
    """Test comprehensive text analysis."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_analyze_text_returns_all_categories(self, validator: EnglishTrapsValidator) -> None:
        """Test analyze_text returns all categories."""
        text = "Their going to discuss about it"
        analysis = validator.analyze_text(text)

        assert "homophones" in analysis
        assert "homophone_warnings" in analysis
        assert "phrasal_verbs" in analysis
        assert "heteronyms" in analysis
        assert "adjective_order" in analysis
        assert "preposition_errors" in analysis
        assert "idioms" in analysis

    def test_analyze_text_empty(self, validator: EnglishTrapsValidator) -> None:
        """Test analyze_text with empty string."""
        analysis = validator.analyze_text("")
        assert isinstance(analysis, dict)

    def test_get_translation_warnings(self, validator: EnglishTrapsValidator) -> None:
        """Test getting translation warnings."""
        text = "Their going to discuss about it"
        warnings = validator.get_translation_warnings(text)

        assert isinstance(warnings, list)
        # Should have warnings for homophone and preposition errors
        assert len(warnings) >= 1

    def test_get_translation_warnings_clean(self, validator: EnglishTrapsValidator) -> None:
        """Test warnings for clean text."""
        text = "Hello world"
        warnings = validator.get_translation_warnings(text)
        assert isinstance(warnings, list)

    def test_get_prompt_enrichment(self, validator: EnglishTrapsValidator) -> None:
        """Test getting prompt enrichment."""
        text = "Their going to discuss about plans"
        enrichment = validator.get_prompt_enrichment(text)

        assert isinstance(enrichment, str)

    def test_get_prompt_enrichment_empty(self, validator: EnglishTrapsValidator) -> None:
        """Test prompt enrichment for clean text."""
        text = "Hello"
        enrichment = validator.get_prompt_enrichment(text)
        assert isinstance(enrichment, str)


@pytest.mark.unit
class TestPhrasalVerbsFinding:
    """Test finding phrasal verbs in text."""

    @pytest.fixture
    def validator(self) -> EnglishTrapsValidator:
        """Create a validator instance."""
        return EnglishTrapsValidator()

    def test_find_phrasal_verbs_returns_list(self, validator: EnglishTrapsValidator) -> None:
        """Test find_phrasal_verbs_in_text returns list."""
        text = "Please take off your shoes"
        found = validator.find_phrasal_verbs_in_text(text)
        assert isinstance(found, list)

    def test_find_phrasal_verbs_empty_text(self, validator: EnglishTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_phrasal_verbs_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0
