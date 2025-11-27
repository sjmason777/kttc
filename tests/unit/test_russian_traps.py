"""Unit tests for Russian language traps validator.

Tests the RussianTrapsValidator class for detecting linguistic traps.
"""

import pytest

from kttc.terminology.russian_traps import RussianTrapsValidator


@pytest.mark.unit
class TestRussianTrapsValidatorInitialization:
    """Test RussianTrapsValidator initialization."""

    def test_init_loads_glossaries(self) -> None:
        """Test that validator initializes and loads glossaries."""
        validator = RussianTrapsValidator()
        assert validator is not None
        assert hasattr(validator, "_glossaries")

    def test_is_available(self) -> None:
        """Test is_available method."""
        validator = RussianTrapsValidator()
        result = validator.is_available()
        assert isinstance(result, bool)


@pytest.mark.unit
class TestHomonymsDetection:
    """Test homonym detection functionality."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_get_homonyms(self, validator: RussianTrapsValidator) -> None:
        """Test getting homonyms from glossary."""
        homonyms = validator.get_homonyms()
        assert isinstance(homonyms, dict)

    def test_find_homonyms_returns_list(self, validator: RussianTrapsValidator) -> None:
        """Test find_homonyms_in_text returns list."""
        text = "Я нашёл ключ от замка"
        found = validator.find_homonyms_in_text(text)
        assert isinstance(found, list)

    def test_find_homonyms_empty_text(self, validator: RussianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_homonyms_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestParonymsDetection:
    """Test paronym detection functionality."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_get_paronyms(self, validator: RussianTrapsValidator) -> None:
        """Test getting paronyms from glossary."""
        paronyms = validator.get_paronyms()
        assert isinstance(paronyms, dict)

    def test_find_paronyms_returns_list(self, validator: RussianTrapsValidator) -> None:
        """Test find_paronyms_in_text returns list."""
        text = "Это одеть или надеть?"
        found = validator.find_paronyms_in_text(text)
        assert isinstance(found, list)

    def test_find_paronyms_empty_text(self, validator: RussianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_paronyms_in_text("")
        assert isinstance(found, list)


@pytest.mark.unit
class TestPositionVerbs:
    """Test position verb checking."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_get_position_verbs(self, validator: RussianTrapsValidator) -> None:
        """Test getting position verbs semantics."""
        pv = validator.get_position_verbs()
        assert isinstance(pv, dict)

    def test_check_position_verb_usage_returns_list(self, validator: RussianTrapsValidator) -> None:
        """Test check_position_verb_usage returns list."""
        text = "Книга стоит на столе"
        errors = validator.check_position_verb_usage(text)
        assert isinstance(errors, list)

    def test_check_book_stands_error(self, validator: RussianTrapsValidator) -> None:
        """Test detection of 'книга стоит' error."""
        text = "книга стоит на полке"
        errors = validator.check_position_verb_usage(text)

        assert len(errors) >= 1
        assert errors[0]["correct"] == "книга лежит"
        assert errors[0]["severity"] == "major"

    def test_check_fork_stands_error(self, validator: RussianTrapsValidator) -> None:
        """Test detection of 'вилка стоит' error."""
        text = "вилка стоит на столе"
        errors = validator.check_position_verb_usage(text)

        assert len(errors) >= 1
        assert errors[0]["correct"] == "вилка лежит"

    def test_check_picture_stands_on_wall_error(self, validator: RussianTrapsValidator) -> None:
        """Test detection of 'картина стоит на стене' error."""
        text = "картина стоит на стене"
        errors = validator.check_position_verb_usage(text)

        assert len(errors) >= 1
        assert "висит" in errors[0]["correct"]

    def test_check_bird_stands_error(self, validator: RussianTrapsValidator) -> None:
        """Test detection of 'птица стоит' error."""
        text = "птица стоит на ветке"
        errors = validator.check_position_verb_usage(text)

        assert len(errors) >= 1
        assert "сидит" in errors[0]["correct"]

    def test_no_errors_correct_text(self, validator: RussianTrapsValidator) -> None:
        """Test no errors in correct text."""
        text = "книга лежит на столе"
        errors = validator.check_position_verb_usage(text)
        # Should not detect error for correct usage
        book_errors = [e for e in errors if "книга" in e.get("found", "")]
        assert len(book_errors) == 0

    def test_get_position_verb_paradoxes(self, validator: RussianTrapsValidator) -> None:
        """Test getting position verb paradoxes."""
        paradoxes = validator.get_position_verb_paradoxes()
        assert isinstance(paradoxes, dict)


@pytest.mark.unit
class TestIdiomsDetection:
    """Test idiom detection functionality."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_get_idioms(self, validator: RussianTrapsValidator) -> None:
        """Test getting idioms from glossary."""
        idioms = validator.get_idioms()
        assert isinstance(idioms, dict)

    def test_find_idioms_returns_list(self, validator: RussianTrapsValidator) -> None:
        """Test find_idioms_in_text returns list."""
        text = "Он повесил нос"
        found = validator.find_idioms_in_text(text)
        assert isinstance(found, list)

    def test_find_idioms_empty_text(self, validator: RussianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_idioms_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestUntranslatableWords:
    """Test untranslatable word detection."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_get_untranslatable_words(self, validator: RussianTrapsValidator) -> None:
        """Test getting untranslatable words from glossary."""
        words = validator.get_untranslatable_words()
        assert isinstance(words, dict)

    def test_find_untranslatable_returns_list(self, validator: RussianTrapsValidator) -> None:
        """Test find_untranslatable_in_text returns list."""
        text = "Это тоска и хандра"
        found = validator.find_untranslatable_in_text(text)
        assert isinstance(found, list)

    def test_find_untranslatable_empty_text(self, validator: RussianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_untranslatable_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestStressPatterns:
    """Test stress pattern detection."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_get_stress_homographs(self, validator: RussianTrapsValidator) -> None:
        """Test getting stress homographs."""
        homographs = validator.get_stress_homographs()
        assert isinstance(homographs, dict)

    def test_get_common_stress_errors(self, validator: RussianTrapsValidator) -> None:
        """Test getting common stress errors."""
        errors = validator.get_common_stress_errors()
        assert isinstance(errors, dict)

    def test_find_stress_homographs_returns_list(self, validator: RussianTrapsValidator) -> None:
        """Test find_stress_homographs_in_text returns list."""
        text = "замок на двери"
        found = validator.find_stress_homographs_in_text(text)
        assert isinstance(found, list)

    def test_find_stress_homographs_empty_text(self, validator: RussianTrapsValidator) -> None:
        """Test with empty text."""
        found = validator.find_stress_homographs_in_text("")
        assert isinstance(found, list)
        assert len(found) == 0


@pytest.mark.unit
class TestComprehensiveAnalysis:
    """Test comprehensive text analysis."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_analyze_text_returns_all_categories(self, validator: RussianTrapsValidator) -> None:
        """Test analyze_text returns all categories."""
        text = "Книга стоит на столе"
        analysis = validator.analyze_text(text)

        assert "homonyms" in analysis
        assert "paronyms" in analysis
        assert "position_verbs" in analysis
        assert "idioms" in analysis
        assert "untranslatable" in analysis
        assert "stress_homographs" in analysis

    def test_analyze_text_empty(self, validator: RussianTrapsValidator) -> None:
        """Test analyze_text with empty string."""
        analysis = validator.analyze_text("")
        assert isinstance(analysis, dict)

    def test_get_translation_warnings(self, validator: RussianTrapsValidator) -> None:
        """Test getting translation warnings."""
        text = "Книга стоит на столе"
        warnings = validator.get_translation_warnings(text)

        assert isinstance(warnings, list)

    def test_get_translation_warnings_clean(self, validator: RussianTrapsValidator) -> None:
        """Test warnings for text without issues."""
        text = "Привет мир"
        warnings = validator.get_translation_warnings(text)
        assert isinstance(warnings, list)

    def test_get_prompt_enrichment(self, validator: RussianTrapsValidator) -> None:
        """Test getting prompt enrichment."""
        text = "Это сложный текст"
        enrichment = validator.get_prompt_enrichment(text)

        assert isinstance(enrichment, str)

    def test_get_prompt_enrichment_empty(self, validator: RussianTrapsValidator) -> None:
        """Test prompt enrichment for simple text."""
        text = "Привет"
        enrichment = validator.get_prompt_enrichment(text)
        assert isinstance(enrichment, str)


@pytest.mark.unit
class TestFormatMethods:
    """Test private formatting methods."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_format_homonyms_section_empty(self, validator: RussianTrapsValidator) -> None:
        """Test formatting with empty homonyms."""
        result = validator._format_homonyms_section([])
        assert result == []

    def test_format_homonyms_section_with_data(self, validator: RussianTrapsValidator) -> None:
        """Test formatting with homonym data."""
        homonyms = [
            {
                "word": "ключ",
                "meanings": [
                    {"meaning": "от замка", "english": "key"},
                    {"meaning": "родник", "english": "spring"},
                ],
            }
        ]
        result = validator._format_homonyms_section(homonyms)
        assert len(result) > 0
        assert "HOMONYMS" in result[0]

    def test_format_idioms_section_empty(self, validator: RussianTrapsValidator) -> None:
        """Test formatting with empty idioms."""
        result = validator._format_idioms_section([])
        assert result == []

    def test_format_idioms_section_with_data(self, validator: RussianTrapsValidator) -> None:
        """Test formatting with idiom data."""
        idioms = [
            {
                "idiom": "повесить нос",
                "meaning": "расстроиться",
                "literal": "hang one's nose",
                "english_equivalent": "be downhearted",
            }
        ]
        result = validator._format_idioms_section(idioms)
        assert len(result) > 0
        assert "IDIOMS" in result[0]

    def test_format_untranslatable_section_empty(self, validator: RussianTrapsValidator) -> None:
        """Test formatting with empty untranslatable list."""
        result = validator._format_untranslatable_section([])
        assert result == []

    def test_format_untranslatable_section_with_data(
        self, validator: RussianTrapsValidator
    ) -> None:
        """Test formatting with untranslatable word data."""
        words = [
            {
                "word": "тоска",
                "why_untranslatable": "A uniquely Russian concept combining longing, sadness, and spiritual anguish",
            }
        ]
        result = validator._format_untranslatable_section(words)
        assert len(result) > 0
        assert "UNTRANSLATABLE" in result[0]

    def test_format_stress_section_empty(self, validator: RussianTrapsValidator) -> None:
        """Test formatting with empty stress homographs."""
        result = validator._format_stress_section([])
        assert result == []

    def test_format_stress_section_with_data(self, validator: RussianTrapsValidator) -> None:
        """Test formatting with stress homograph data."""
        homographs = [
            {
                "word": "замок",
                "variants": [
                    {"stress": "зАмок", "meaning": "castle"},
                    {"stress": "замОк", "meaning": "lock"},
                ],
            }
        ]
        result = validator._format_stress_section(homographs)
        assert len(result) > 0
        assert "STRESS" in result[0]


@pytest.mark.unit
class TestGlossaryHandling:
    """Test glossary loading and handling."""

    @pytest.fixture
    def validator(self) -> RussianTrapsValidator:
        """Provide validator instance."""
        return RussianTrapsValidator()

    def test_handles_missing_glossary(self, validator: RussianTrapsValidator) -> None:
        """Test graceful handling of missing glossary."""
        # Remove a glossary
        validator._glossaries.pop("homonyms_paronyms", None)
        # Should not raise
        homonyms = validator.get_homonyms()
        assert isinstance(homonyms, dict)

    def test_handles_empty_glossary(self, validator: RussianTrapsValidator) -> None:
        """Test handling of empty glossary."""
        validator._glossaries["idioms"] = {}
        idioms = validator.get_idioms()
        assert isinstance(idioms, dict)

    def test_get_glossary_path(self, validator: RussianTrapsValidator) -> None:
        """Test getting glossary path."""
        path = validator._get_glossary_path()
        assert path.name == "ru"
        assert "glossaries" in str(path)
