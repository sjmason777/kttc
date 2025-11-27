# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
# SPDX-License-Identifier: Apache-2.0
"""Tests for terminology traps validators (Russian and English)."""

from __future__ import annotations

import pytest

# ============================================================================
# Russian Traps Validator Tests
# ============================================================================


@pytest.mark.unit
class TestRussianTrapsValidatorInit:
    """Tests for RussianTrapsValidator initialization."""

    def test_init_loads_glossaries(self) -> None:
        """Test that validator loads glossaries on init."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        assert validator.is_available()

    def test_glossary_path(self) -> None:
        """Test glossary path resolution."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        path = validator._get_glossary_path()

        assert path.exists()
        assert path.name == "ru"


@pytest.mark.unit
class TestRussianHomonyms:
    """Tests for Russian homonyms detection."""

    def test_get_homonyms(self) -> None:
        """Test getting homonyms glossary."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        homonyms = validator.get_homonyms()

        assert isinstance(homonyms, dict)

    def test_find_homonyms_in_text(self) -> None:
        """Test finding homonyms in Russian text."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        # "ключ" - homonym (key/spring)
        text = "Я нашёл ключ от замка"
        found = validator.find_homonyms_in_text(text)

        if validator.get_homonyms():
            # If glossary has ключ, it should be found
            assert isinstance(found, list)

    def test_find_homonyms_empty_text(self) -> None:
        """Test finding homonyms in empty text."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        found = validator.find_homonyms_in_text("")

        assert found == []


@pytest.mark.unit
class TestRussianParonyms:
    """Tests for Russian paronyms detection."""

    def test_get_paronyms(self) -> None:
        """Test getting paronyms glossary."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        paronyms = validator.get_paronyms()

        assert isinstance(paronyms, dict)

    def test_find_paronyms_in_text(self) -> None:
        """Test finding paronyms in Russian text."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        # "одеть/надеть" - common paronym pair
        text = "Надо одеть пальто"
        found = validator.find_paronyms_in_text(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestRussianPositionVerbs:
    """Tests for Russian position verbs."""

    def test_get_position_verbs(self) -> None:
        """Test getting position verbs glossary."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        verbs = validator.get_position_verbs()

        assert isinstance(verbs, dict)

    def test_get_position_verb_paradoxes(self) -> None:
        """Test getting position verb paradoxes."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        paradoxes = validator.get_position_verb_paradoxes()

        assert isinstance(paradoxes, dict)

    def test_check_position_verb_usage(self) -> None:
        """Test checking position verb usage."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        # стоять/лежать/сидеть usage check
        text = "Книга стоит на полке"
        issues = validator.check_position_verb_usage(text)

        assert isinstance(issues, list)


@pytest.mark.unit
class TestRussianIdioms:
    """Tests for Russian idioms detection."""

    def test_get_idioms(self) -> None:
        """Test getting idioms glossary."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        idioms = validator.get_idioms()

        assert isinstance(idioms, dict)

    def test_find_idioms_in_text(self) -> None:
        """Test finding idioms in Russian text."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        text = "Это проще пареной репы"
        found = validator.find_idioms_in_text(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestRussianUntranslatable:
    """Tests for Russian untranslatable words."""

    def test_get_untranslatable_words(self) -> None:
        """Test getting untranslatable words glossary."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        words = validator.get_untranslatable_words()

        assert isinstance(words, dict)

    def test_find_untranslatable_in_text(self) -> None:
        """Test finding untranslatable words in text."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        # "тоска" - often cited as untranslatable
        text = "Меня охватила тоска"
        found = validator.find_untranslatable_in_text(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestRussianStressPatterns:
    """Tests for Russian stress patterns."""

    def test_get_stress_homographs(self) -> None:
        """Test getting stress homographs glossary."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        homographs = validator.get_stress_homographs()

        assert isinstance(homographs, dict)

    def test_get_common_stress_errors(self) -> None:
        """Test getting common stress errors."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        errors = validator.get_common_stress_errors()

        assert isinstance(errors, dict)

    def test_find_stress_homographs_in_text(self) -> None:
        """Test finding stress homographs in text."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        # "замок" - castle (зАмок) vs lock (замОк)
        text = "Я видел старый замок"
        found = validator.find_stress_homographs_in_text(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestRussianAnalyzeText:
    """Tests for comprehensive text analysis."""

    def test_analyze_text_returns_dict(self) -> None:
        """Test analyze_text returns proper structure."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        text = "Я нашёл ключ от старого замка"
        result = validator.analyze_text(text)

        assert isinstance(result, dict)
        assert "homonyms" in result
        assert "paronyms" in result
        assert "idioms" in result
        assert "position_verbs" in result
        assert "untranslatable" in result
        assert "stress_homographs" in result

    def test_analyze_text_empty(self) -> None:
        """Test analyze_text with empty string."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        result = validator.analyze_text("")

        assert isinstance(result, dict)

    def test_get_translation_warnings(self) -> None:
        """Test getting translation warnings."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        text = "Это тоска по родине"
        warnings = validator.get_translation_warnings(text)

        assert isinstance(warnings, list)

    def test_get_prompt_enrichment(self) -> None:
        """Test getting prompt enrichment data."""
        from kttc.terminology.russian_traps import RussianTrapsValidator

        validator = RussianTrapsValidator()
        text = "Ключ от замка"
        enrichment = validator.get_prompt_enrichment(text)

        # Returns formatted string for prompt injection
        assert isinstance(enrichment, str)


# ============================================================================
# English Traps Validator Tests
# ============================================================================


@pytest.mark.unit
class TestEnglishTrapsValidatorInit:
    """Tests for EnglishTrapsValidator initialization."""

    def test_init_loads_glossaries(self) -> None:
        """Test that validator loads glossaries on init."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        assert validator.is_available()

    def test_has_constants(self) -> None:
        """Test that validator has expected constants."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        assert hasattr(EnglishTrapsValidator, "HOMOPHONE_ERROR_PATTERNS")
        assert hasattr(EnglishTrapsValidator, "ADJECTIVE_CATEGORIES")
        assert len(EnglishTrapsValidator.HOMOPHONE_ERROR_PATTERNS) > 0
        assert len(EnglishTrapsValidator.ADJECTIVE_CATEGORIES) > 0


@pytest.mark.unit
class TestEnglishHomophones:
    """Tests for English homophones detection."""

    def test_get_homophones(self) -> None:
        """Test getting homophones glossary."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        homophones = validator.get_homophones()

        assert isinstance(homophones, dict)

    def test_find_homophones_in_text(self) -> None:
        """Test finding homophones in English text."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "Their going to the store"
        found = validator.find_homophones_in_text(text)

        assert isinstance(found, list)

    def test_find_homophones_detects_their_error(self) -> None:
        """Test that their/they're error is detected."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "Their going to school tomorrow"
        found = validator.find_homophones_in_text(text)

        # Should detect "their going" as likely error
        assert len(found) >= 1

    def test_check_homophone_context(self) -> None:
        """Test homophone context checking."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "Your welcome to join us"
        issues = validator.check_homophone_context(text)

        assert isinstance(issues, list)
        # "Your welcome" is a common error
        assert len(issues) >= 1


@pytest.mark.unit
class TestEnglishPhrasalVerbs:
    """Tests for English phrasal verbs detection."""

    def test_get_phrasal_verbs(self) -> None:
        """Test getting phrasal verbs glossary."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        verbs = validator.get_phrasal_verbs()

        assert isinstance(verbs, dict)

    def test_find_phrasal_verbs_in_text(self) -> None:
        """Test finding phrasal verbs in text."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "I need to give up smoking"
        found = validator.find_phrasal_verbs_in_text(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestEnglishHeteronyms:
    """Tests for English heteronyms detection."""

    def test_get_heteronyms(self) -> None:
        """Test getting heteronyms glossary."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        heteronyms = validator.get_heteronyms()

        assert isinstance(heteronyms, dict)

    def test_find_heteronyms_in_text(self) -> None:
        """Test finding heteronyms in text."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        # "lead" is a heteronym (verb vs noun)
        text = "He will lead the team"
        found = validator.find_heteronyms_in_text(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestEnglishAdjectiveOrder:
    """Tests for English adjective order checking."""

    def test_adjective_categories_complete(self) -> None:
        """Test that all adjective categories are present."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        categories = EnglishTrapsValidator.ADJECTIVE_CATEGORIES

        expected = ["opinion", "size", "age", "shape", "color"]
        for cat in expected:
            assert cat in categories

    def test_get_adjective_category(self) -> None:
        """Test getting adjective category."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()

        assert validator.get_adjective_category("beautiful") == "opinion"
        assert validator.get_adjective_category("big") == "size"
        assert validator.get_adjective_category("old") == "age"
        assert validator.get_adjective_category("round") == "shape"
        assert validator.get_adjective_category("red") == "color"
        # Unknown adjective
        assert validator.get_adjective_category("unknown_adj") is None

    def test_check_adjective_order(self) -> None:
        """Test adjective order checking."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()

        # Correct order: opinion-size-age-shape-color
        correct = "a beautiful big old round red ball"
        issues_correct = validator.check_adjective_order(correct)

        # Wrong order: color before size
        wrong = "a red big ball"
        issues_wrong = validator.check_adjective_order(wrong)

        assert isinstance(issues_correct, list)
        assert isinstance(issues_wrong, list)


@pytest.mark.unit
class TestEnglishPrepositions:
    """Tests for English preposition error detection."""

    def test_get_preposition_errors(self) -> None:
        """Test getting preposition errors glossary."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        errors = validator.get_preposition_errors()

        assert isinstance(errors, list)

    def test_find_preposition_errors(self) -> None:
        """Test finding preposition errors in text."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        # Common error: "different than" (US) vs "different from" (UK)
        text = "This is different than what I expected"
        found = validator.find_preposition_errors(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestEnglishIdioms:
    """Tests for English idioms detection."""

    def test_get_idioms(self) -> None:
        """Test getting idioms glossary."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        idioms = validator.get_idioms()

        assert isinstance(idioms, dict)

    def test_find_idioms_in_text(self) -> None:
        """Test finding idioms in text."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "This task is a piece of cake"
        found = validator.find_idioms_in_text(text)

        assert isinstance(found, list)


@pytest.mark.unit
class TestEnglishFalseFriends:
    """Tests for English false friends detection."""

    def test_get_false_friends(self) -> None:
        """Test getting false friends glossary."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        friends = validator.get_false_friends()

        assert isinstance(friends, dict)

    def test_find_false_friends_in_context(self) -> None:
        """Test finding false friends with context."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        # "actually" is a false friend (Spanish "actualmente" = currently)
        text = "I actually finished the work"
        source_lang = "en"
        target_lang = "es"
        found = validator.find_false_friends_in_context(text, source_lang, target_lang)

        assert isinstance(found, list)


@pytest.mark.unit
class TestEnglishAnalyzeText:
    """Tests for comprehensive text analysis."""

    def test_analyze_text_returns_dict(self) -> None:
        """Test analyze_text returns proper structure."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "Their going to give up and take a piece of cake"
        result = validator.analyze_text(text)

        assert isinstance(result, dict)
        assert "homophones" in result
        assert "phrasal_verbs" in result
        assert "idioms" in result

    def test_analyze_text_empty(self) -> None:
        """Test analyze_text with empty string."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        result = validator.analyze_text("")

        assert isinstance(result, dict)

    def test_get_translation_warnings(self) -> None:
        """Test getting translation warnings."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "Break a leg in your presentation"
        warnings = validator.get_translation_warnings(text)

        assert isinstance(warnings, list)

    def test_get_prompt_enrichment(self) -> None:
        """Test getting prompt enrichment data."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()
        text = "Their going to lead the team"
        enrichment = validator.get_prompt_enrichment(text)

        # Returns formatted string for prompt injection
        assert isinstance(enrichment, str)


@pytest.mark.unit
class TestHomophoneErrorPatterns:
    """Tests for homophone error pattern matching."""

    def test_their_theyre_pattern(self) -> None:
        """Test their/they're error detection."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()

        # Error cases
        error_texts = [
            "Their going to school",
            "Their is a problem",
            "Their coming soon",
        ]

        for text in error_texts:
            found = validator.find_homophones_in_text(text)
            # Pattern should match
            assert len(found) >= 1, f"Failed to detect error in: {text}"

    def test_your_youre_pattern(self) -> None:
        """Test your/you're error detection."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()

        # Error case
        text = "Your welcome"
        issues = validator.check_homophone_context(text)

        assert len(issues) >= 1

    def test_its_pattern(self) -> None:
        """Test its/it's error detection."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()

        # Error case: "its" should be "it's"
        text = "Its a beautiful day"
        found = validator.find_homophones_in_text(text)

        assert len(found) >= 1

    def test_check_homophone_context_flags_for_review(self) -> None:
        """Test that check_homophone_context flags homophones for manual review.

        Note: This function flags ALL critical homophones for verification,
        regardless of whether usage is correct. It's not an error detector.
        """
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()

        # Correct usages still get flagged for manual review (by design)
        texts_with_critical_homophones = [
            ("They're going to school", "they're"),
            ("It's a beautiful day", "it's"),
            ("You're welcome", "you're"),
            ("Their house is big", "their"),
        ]

        for text, expected_word in texts_with_critical_homophones:
            issues = validator.check_homophone_context(text)
            # Should flag homophone for manual review
            assert len(issues) >= 1, f"Expected review flag for: {text}"
            found_words = [issue["word"] for issue in issues]
            assert expected_word in found_words, f"Expected '{expected_word}' in {found_words}"

    def test_no_critical_homophones_no_warnings(self) -> None:
        """Test text without critical homophones doesn't trigger warnings."""
        from kttc.terminology.english_traps import EnglishTrapsValidator

        validator = EnglishTrapsValidator()

        # Text without any critical homophones
        text = "The quick brown fox jumps over lazy dogs"
        issues = validator.check_homophone_context(text)

        assert len(issues) == 0, f"Unexpected warnings: {issues}"
