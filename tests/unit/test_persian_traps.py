"""Unit tests for Persian language traps validator.

Tests the PersianTrapsValidator class for detecting linguistic traps.
"""

import pytest

from kttc.terminology.persian_traps import PersianTrapsValidator


@pytest.mark.unit
class TestPersianTrapsValidatorInitialization:
    """Test PersianTrapsValidator initialization."""

    def test_init_loads_glossaries(self) -> None:
        """Test that validator initializes and loads glossaries."""
        validator = PersianTrapsValidator()
        assert validator is not None
        assert hasattr(validator, "_glossaries")

    def test_is_available(self) -> None:
        """Test is_available method."""
        validator = PersianTrapsValidator()
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_get_glossary_path(self) -> None:
        """Test getting glossary path."""
        validator = PersianTrapsValidator()
        path = validator._get_glossary_path()
        assert path.name == "fa"
        assert "glossaries" in str(path)


@pytest.mark.unit
class TestFalseFriends:
    """Test Persian-Arabic false friend detection."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_get_false_friends(self, validator: PersianTrapsValidator) -> None:
        """Test getting false friends from glossary."""
        ff = validator.get_false_friends()
        assert isinstance(ff, dict)

    def test_find_false_friends_returns_list(self, validator: PersianTrapsValidator) -> None:
        """Test find_false_friends_in_text returns list."""
        text = "جهاز بزرگ است"
        found = validator.find_false_friends_in_text(text)
        assert isinstance(found, list)

    def test_find_false_friends_empty_text(self, validator: PersianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_false_friends_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestTaarofPhrases:
    """Test Ta'arof politeness phrase detection."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_get_taarof_phrases(self, validator: PersianTrapsValidator) -> None:
        """Test getting Ta'arof phrases from glossary."""
        phrases = validator.get_taarof_phrases()
        assert isinstance(phrases, dict)

    def test_find_taarof_in_text(self, validator: PersianTrapsValidator) -> None:
        """Test find_taarof_in_text returns list."""
        text = "قابل نداره"
        found = validator.find_taarof_in_text(text)
        assert isinstance(found, list)

    def test_get_taarof_scenarios(self, validator: PersianTrapsValidator) -> None:
        """Test getting taarof scenarios."""
        scenarios = validator.get_taarof_scenarios()
        assert isinstance(scenarios, dict)

    def test_find_taarof_empty_text(self, validator: PersianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_taarof_in_text("")
        assert isinstance(found, list)


@pytest.mark.unit
class TestColloquialFormal:
    """Test colloquial/formal register checking."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_get_colloquial_forms(self, validator: PersianTrapsValidator) -> None:
        """Test getting colloquial forms from glossary."""
        forms = validator.get_colloquial_forms()
        assert isinstance(forms, dict)

    def test_check_register_consistency(self, validator: PersianTrapsValidator) -> None:
        """Test check_register_consistency returns list."""
        text = "میخوام برم"
        errors = validator.check_register_consistency(text)
        assert isinstance(errors, list)

    def test_detect_register(self, validator: PersianTrapsValidator) -> None:
        """Test detect_register returns string."""
        text = "سلام حال شما چطور است"
        register = validator.detect_register(text)
        assert register in ["formal", "colloquial", "mixed"]


@pytest.mark.unit
class TestCompoundVerbs:
    """Test compound verb checking."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_get_compound_verbs(self, validator: PersianTrapsValidator) -> None:
        """Test getting compound verbs from glossary."""
        verbs = validator.get_compound_verbs()
        assert isinstance(verbs, dict)

    def test_check_compound_verbs_returns_list(self, validator: PersianTrapsValidator) -> None:
        """Test check_compound_verb_errors returns list."""
        text = "حرف کردن"
        errors = validator.check_compound_verb_errors(text)
        assert isinstance(errors, list)


@pytest.mark.unit
class TestIdiomsDetection:
    """Test idiom detection functionality."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_get_idioms(self, validator: PersianTrapsValidator) -> None:
        """Test getting idioms from glossary."""
        idioms = validator.get_idioms()
        assert isinstance(idioms, dict)

    def test_find_idioms_returns_list(self, validator: PersianTrapsValidator) -> None:
        """Test find_idioms_in_text returns list."""
        text = "دست از سر کسی برداشتن"
        found = validator.find_idioms_in_text(text)
        assert isinstance(found, list)

    def test_find_idioms_empty_text(self, validator: PersianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_idioms_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestUntranslatableWords:
    """Test untranslatable word detection."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_get_untranslatable_words(self, validator: PersianTrapsValidator) -> None:
        """Test getting untranslatable words from glossary."""
        words = validator.get_untranslatable_words()
        assert isinstance(words, dict)

    def test_find_untranslatable_returns_list(self, validator: PersianTrapsValidator) -> None:
        """Test find_untranslatable_in_text returns list."""
        text = "تعارف کردن"
        found = validator.find_untranslatable_in_text(text)
        assert isinstance(found, list)

    def test_find_untranslatable_empty_text(self, validator: PersianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_untranslatable_in_text("")
        assert isinstance(found, list)


@pytest.mark.unit
class TestComprehensiveAnalysis:
    """Test comprehensive text analysis."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_analyze_text_returns_all_categories(self, validator: PersianTrapsValidator) -> None:
        """Test analyze_text returns all categories."""
        text = "سلام، حال شما چطور است؟"
        analysis = validator.analyze_text(text)

        assert "false_friends" in analysis
        assert "taarof" in analysis
        assert "register_issues" in analysis
        assert "detected_register" in analysis
        assert "compound_verb_errors" in analysis
        assert "idioms" in analysis
        assert "untranslatable" in analysis

    def test_analyze_text_empty(self, validator: PersianTrapsValidator) -> None:
        """Test analyze_text with empty string."""
        analysis = validator.analyze_text("")
        assert isinstance(analysis, dict)

    def test_get_translation_warnings(self, validator: PersianTrapsValidator) -> None:
        """Test getting translation warnings."""
        text = "قابل نداره"
        warnings = validator.get_translation_warnings(text)
        assert isinstance(warnings, list)

    def test_get_prompt_enrichment(self, validator: PersianTrapsValidator) -> None:
        """Test getting prompt enrichment."""
        text = "این یک آزمایش است"
        enrichment = validator.get_prompt_enrichment(text)
        assert isinstance(enrichment, str)


@pytest.mark.unit
class TestGlossaryHandling:
    """Test glossary loading and handling."""

    @pytest.fixture
    def validator(self) -> PersianTrapsValidator:
        """Provide validator instance."""
        return PersianTrapsValidator()

    def test_handles_missing_glossary(self, validator: PersianTrapsValidator) -> None:
        """Test graceful handling of missing glossary."""
        validator._glossaries.pop("false_friends", None)
        ff = validator.get_false_friends()
        assert isinstance(ff, dict)

    def test_handles_empty_glossary(self, validator: PersianTrapsValidator) -> None:
        """Test handling of empty glossary."""
        validator._glossaries["idioms"] = {}
        idioms = validator.get_idioms()
        assert isinstance(idioms, dict)
