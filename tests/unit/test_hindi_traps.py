"""Unit tests for Hindi language traps validator.

Tests the HindiTrapsValidator class for detecting linguistic traps.
"""

import pytest

from kttc.terminology.hindi_traps import HindiTrapsValidator


@pytest.mark.unit
class TestHindiTrapsValidatorInitialization:
    """Test HindiTrapsValidator initialization."""

    def test_init_loads_glossaries(self) -> None:
        """Test that validator initializes and loads glossaries."""
        validator = HindiTrapsValidator()
        assert validator is not None
        assert hasattr(validator, "_glossaries")

    def test_is_available(self) -> None:
        """Test is_available method."""
        validator = HindiTrapsValidator()
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_get_glossary_path(self) -> None:
        """Test getting glossary path."""
        validator = HindiTrapsValidator()
        path = validator._get_glossary_path()
        assert path.name == "hi"
        assert "glossaries" in str(path)


@pytest.mark.unit
class TestGenderExceptions:
    """Test gender exception detection."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_get_gender_exceptions(self, validator: HindiTrapsValidator) -> None:
        """Test getting gender exceptions from glossary."""
        exceptions = validator.get_gender_exceptions()
        assert isinstance(exceptions, dict)

    def test_get_gender_exceptions_in_text(self, validator: HindiTrapsValidator) -> None:
        """Test get_gender_exceptions_in_text returns list."""
        text = "पानी ठंडी है"
        found = validator.get_gender_exceptions_in_text(text)
        assert isinstance(found, list)

    def test_get_gender_exceptions_empty_text(self, validator: HindiTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.get_gender_exceptions_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestIdiomsDetection:
    """Test idiom detection functionality."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_get_idioms(self, validator: HindiTrapsValidator) -> None:
        """Test getting idioms from glossary."""
        idioms = validator.get_idioms()
        assert isinstance(idioms, dict)

    def test_get_idioms_in_text(self, validator: HindiTrapsValidator) -> None:
        """Test get_idioms_in_text returns list."""
        text = "आँखों का पानी"
        found = validator.get_idioms_in_text(text)
        assert isinstance(found, list)

    def test_get_idioms_empty_text(self, validator: HindiTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.get_idioms_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestChandrabinduAnusvara:
    """Test Chandrabindu/Anusvara checking."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_get_chandrabindu_words(self, validator: HindiTrapsValidator) -> None:
        """Test getting Chandrabindu words."""
        words = validator.get_chandrabindu_words()
        assert isinstance(words, list)

    def test_get_meaning_change_pairs(self, validator: HindiTrapsValidator) -> None:
        """Test getting meaning change pairs."""
        pairs = validator.get_meaning_change_pairs()
        assert isinstance(pairs, dict)

    def test_check_chandrabindu_errors(self, validator: HindiTrapsValidator) -> None:
        """Test check_chandrabindu_errors returns list."""
        text = "माँ और माम"
        errors = validator.check_chandrabindu_errors(text)
        assert isinstance(errors, list)


@pytest.mark.unit
class TestHomophonesParonyms:
    """Test homophone/paronym detection."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_get_homophones(self, validator: HindiTrapsValidator) -> None:
        """Test getting homophones from glossary."""
        homophones = validator.get_homophones()
        assert isinstance(homophones, dict)

    def test_get_homophones_in_text(self, validator: HindiTrapsValidator) -> None:
        """Test get_homophones_in_text returns list."""
        text = "वह काला है"
        found = validator.get_homophones_in_text(text)
        assert isinstance(found, list)

    def test_get_paronyms(self, validator: HindiTrapsValidator) -> None:
        """Test getting paronyms from glossary."""
        paronyms = validator.get_paronyms()
        assert isinstance(paronyms, dict)


@pytest.mark.unit
class TestAspirationTraps:
    """Test aspiration trap detection."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_get_aspiration_pairs(self, validator: HindiTrapsValidator) -> None:
        """Test getting aspiration pairs."""
        pairs = validator.get_aspiration_pairs()
        assert isinstance(pairs, dict)

    def test_check_aspiration_words_in_text(self, validator: HindiTrapsValidator) -> None:
        """Test check_aspiration_words_in_text returns list."""
        text = "कल और खल"
        found = validator.check_aspiration_words_in_text(text)
        assert isinstance(found, list)


@pytest.mark.unit
class TestVerbErgativity:
    """Test verb ergativity checking."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_get_ergativity_rules(self, validator: HindiTrapsValidator) -> None:
        """Test getting ergativity rules."""
        rules = validator.get_ergativity_rules()
        assert isinstance(rules, dict)

    def test_get_ergativity_examples(self, validator: HindiTrapsValidator) -> None:
        """Test getting ergativity examples."""
        examples = validator.get_ergativity_examples()
        assert isinstance(examples, dict)

    def test_get_common_ergativity_errors(self, validator: HindiTrapsValidator) -> None:
        """Test getting common ergativity errors."""
        errors = validator.get_common_ergativity_errors()
        assert isinstance(errors, dict)

    def test_check_ne_usage(self, validator: HindiTrapsValidator) -> None:
        """Test check_ne_usage returns list."""
        text = "उसने किया"
        found = validator.check_ne_usage(text)
        assert isinstance(found, list)


@pytest.mark.unit
class TestComprehensiveAnalysis:
    """Test comprehensive text analysis."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_analyze_text_returns_all_categories(self, validator: HindiTrapsValidator) -> None:
        """Test analyze_text returns all categories."""
        text = "पानी ठंडी है"
        analysis = validator.analyze_text(text)

        assert "gender_exceptions" in analysis
        assert "idioms" in analysis
        assert "chandrabindu_errors" in analysis
        assert "homophones" in analysis
        assert "aspiration_words" in analysis
        assert "ne_constructions" in analysis

    def test_analyze_text_empty(self, validator: HindiTrapsValidator) -> None:
        """Test analyze_text with empty string."""
        analysis = validator.analyze_text("")
        assert isinstance(analysis, dict)

    def test_get_trap_summary(self, validator: HindiTrapsValidator) -> None:
        """Test getting trap summary."""
        text = "पानी ठंडा है"
        summary = validator.get_trap_summary(text)
        assert isinstance(summary, str)

    def test_get_trap_summary_empty(self, validator: HindiTrapsValidator) -> None:
        """Test trap summary for text without traps."""
        text = "हेलो"
        summary = validator.get_trap_summary(text)
        assert isinstance(summary, str)


@pytest.mark.unit
class TestGlossaryHandling:
    """Test glossary loading and handling."""

    @pytest.fixture
    def validator(self) -> HindiTrapsValidator:
        """Provide validator instance."""
        return HindiTrapsValidator()

    def test_handles_missing_glossary(self, validator: HindiTrapsValidator) -> None:
        """Test graceful handling of missing glossary."""
        validator._glossaries.pop("gender_traps", None)
        exceptions = validator.get_gender_exceptions()
        assert isinstance(exceptions, dict)

    def test_handles_empty_glossary(self, validator: HindiTrapsValidator) -> None:
        """Test handling of empty glossary."""
        validator._glossaries["idioms"] = {}
        idioms = validator.get_idioms()
        assert isinstance(idioms, dict)
