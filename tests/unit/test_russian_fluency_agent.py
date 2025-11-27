"""Unit tests for Russian fluency agent.

Tests Russian-specific fluency checking with hybrid NLP + LLM approach.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kttc.agents.fluency_russian import RussianFluencyAgent
from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask


@pytest.mark.unit
class TestRussianFluencyAgentInitialization:
    """Test Russian fluency agent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        mock_provider = MagicMock()
        with patch("kttc.agents.fluency_russian.RussianLanguageHelper"):
            agent = RussianFluencyAgent(mock_provider)

        assert agent.llm_provider == mock_provider
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000
        assert agent.helper is not None

    def test_custom_initialization(self) -> None:
        """Test custom initialization parameters."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(
            mock_provider, temperature=0.3, max_tokens=3000, helper=mock_helper
        )

        assert agent.temperature == 0.3
        assert agent.max_tokens == 3000
        assert agent.helper == mock_helper

    def test_russian_checks_defined(self) -> None:
        """Test that Russian-specific checks are defined."""
        assert "case_agreement" in RussianFluencyAgent.RUSSIAN_CHECKS
        assert "aspect_usage" in RussianFluencyAgent.RUSSIAN_CHECKS
        assert "word_order" in RussianFluencyAgent.RUSSIAN_CHECKS
        assert "particle_usage" in RussianFluencyAgent.RUSSIAN_CHECKS
        assert "register" in RussianFluencyAgent.RUSSIAN_CHECKS


@pytest.mark.unit
class TestGetBasePrompt:
    """Test base prompt generation."""

    def test_get_base_prompt_includes_russian_section(self) -> None:
        """Test that base prompt includes Russian-specific section."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        prompt = agent.get_base_prompt()

        assert "RUSSIAN-SPECIFIC CHECKS" in prompt
        assert "Russian-specific linguistic validation" in prompt


@pytest.mark.unit
class TestErrorsOverlap:
    """Test error overlap detection."""

    def test_errors_overlap_true(self) -> None:
        """Test detecting overlapping errors."""
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 10),
            description="Test",
        )
        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(5, 15),
            description="Test",
        )

        assert RussianFluencyAgent._errors_overlap(error1, error2) is True

    def test_errors_overlap_false_adjacent(self) -> None:
        """Test non-overlapping adjacent errors."""
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 10),
            description="Test",
        )
        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(10, 20),
            description="Test",
        )

        assert RussianFluencyAgent._errors_overlap(error1, error2) is False

    def test_errors_overlap_false_separate(self) -> None:
        """Test non-overlapping separate errors."""
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test",
        )
        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(15, 20),
            description="Test",
        )

        assert RussianFluencyAgent._errors_overlap(error1, error2) is False


@pytest.mark.unit
class TestRemoveDuplicates:
    """Test duplicate error removal."""

    def test_remove_duplicates_with_overlap(self) -> None:
        """Test removing duplicates when errors overlap."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        nlp_error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(5, 10),
            description="NLP error",
        )
        llm_error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 15),
            description="LLM error",
        )

        unique = agent._remove_duplicates([nlp_error], [llm_error])

        assert len(unique) == 0  # NLP error overlaps with LLM error

    def test_remove_duplicates_no_overlap(self) -> None:
        """Test keeping unique errors when no overlap."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        nlp_error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="NLP error",
        )
        llm_error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(20, 30),
            description="LLM error",
        )

        unique = agent._remove_duplicates([nlp_error], [llm_error])

        assert len(unique) == 1
        assert unique[0] == nlp_error


@pytest.mark.unit
class TestVerifyLLMErrors:
    """Test LLM error verification."""

    def test_verify_errors_without_helper(self) -> None:
        """Test that without helper all errors are returned."""
        mock_provider = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=None)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test error",
        )

        verified = agent._verify_llm_errors([error], "Test text")

        assert len(verified) == 1
        assert verified[0] == error

    def test_verify_errors_with_valid_position(self) -> None:
        """Test verifying errors with valid positions."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.verify_error_position.return_value = True
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test error",
        )

        verified = agent._verify_llm_errors([error], "Test text")

        assert len(verified) == 1
        mock_helper.verify_error_position.assert_called_once()

    def test_verify_errors_filters_invalid_position(self) -> None:
        """Test filtering errors with invalid positions."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.verify_error_position.return_value = False
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(100, 200),  # Invalid position
            description="Test error",
        )

        verified = agent._verify_llm_errors([error], "Short text")

        assert len(verified) == 0


@pytest.mark.unit
class TestParseJSONResponse:
    """Test JSON response parsing."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        response = '{"errors": [], "text_type": "technical"}'
        result = agent._parse_json_response(response)

        assert result["errors"] == []
        assert result["text_type"] == "technical"

    def test_parse_json_from_markdown(self) -> None:
        """Test parsing JSON from markdown code block."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        response = """Analysis:
```json
{"errors": [{"subcategory": "test"}]}
```
Done."""

        result = agent._parse_json_response(response)

        assert len(result["errors"]) == 1
        assert result["errors"][0]["subcategory"] == "test"

    def test_parse_json_embedded(self) -> None:
        """Test parsing JSON embedded in text."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        response = 'Here is the result: {"errors": []} End.'
        result = agent._parse_json_response(response)

        assert result["errors"] == []

    def test_parse_invalid_json_returns_empty(self) -> None:
        """Test parsing invalid JSON returns empty errors."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        response = "This is not JSON at all"
        result = agent._parse_json_response(response)

        assert result["errors"] == []


@pytest.mark.unit
class TestFilterFalsePositives:
    """Test false positive filtering."""

    def test_filter_tech_term_gen5(self) -> None:
        """Test filtering Gen 5 technical term."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="russian_numeral_agreement",
            severity=ErrorSeverity.MINOR,
            location=(20, 25),
            description="Numeral agreement error",
        )

        text = "плата поддерживает Gen 5 и USB4"

        filtered = agent._filter_false_positives([error], text)

        assert len(filtered) == 0  # Should be filtered as tech term

    def test_filter_digit_genitive(self) -> None:
        """Test filtering correct digit+genitive form."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="russian_numeral_agreement",
            severity=ErrorSeverity.MINOR,
            location=(0, 8),
            description="Numeral agreement error",
        )

        text = "5 минут назад"  # Correct form

        filtered = agent._filter_false_positives([error], text)

        assert len(filtered) == 0  # Should be filtered as correct form

    def test_filter_particle_phrase_to_zhe(self) -> None:
        """Test filtering standard particle phrase."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="russian_particle_usage",
            severity=ErrorSeverity.MINOR,
            location=(0, 10),
            description="Particle usage error",
        )

        text = "то же самое можно сказать"  # Standard phrase

        filtered = agent._filter_false_positives([error], text)

        assert len(filtered) == 0  # Should be filtered as standard phrase

    def test_keep_real_error(self) -> None:
        """Test keeping real errors."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="russian_case_agreement",
            severity=ErrorSeverity.MAJOR,
            location=(0, 15),
            description="Case agreement error: красивый кошка",
        )

        text = "красивый кошка спит"  # Real error - gender mismatch

        filtered = agent._filter_false_positives([error], text)

        assert len(filtered) == 1  # Should keep real error


@pytest.mark.unit
class TestNLPCheckSync:
    """Test NLP-based grammar checks."""

    def test_nlp_check_without_helper(self) -> None:
        """Test NLP check returns empty when helper unavailable."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = False
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._nlp_check_sync(task)

        assert errors == []

    def test_nlp_check_with_helper(self) -> None:
        """Test NLP check calls helper."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.return_value = []
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._nlp_check_sync(task)

        mock_helper.check_grammar.assert_called_once_with("Привет")
        assert errors == []

    def test_nlp_check_handles_exception(self) -> None:
        """Test NLP check handles exceptions gracefully."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.side_effect = Exception("NLP error")
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._nlp_check_sync(task)

        assert errors == []


@pytest.mark.unit
class TestEntityCheckSync:
    """Test entity preservation checks."""

    def test_entity_check_without_helper(self) -> None:
        """Test entity check returns empty when helper unavailable."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = False
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        task = TranslationTask(
            source_text="John Smith",
            translation="Джон Смит",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._entity_check_sync(task)

        assert errors == []

    def test_entity_check_with_helper(self) -> None:
        """Test entity check calls helper."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_entity_preservation.return_value = []
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        task = TranslationTask(
            source_text="John Smith",
            translation="Джон Смит",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._entity_check_sync(task)

        mock_helper.check_entity_preservation.assert_called_once()
        assert errors == []


@pytest.mark.unit
class TestTrapsCheckSync:
    """Test Russian traps validation."""

    def test_traps_check_without_validator(self) -> None:
        """Test traps check returns empty when validator unavailable."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)
        agent.traps_validator = None

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._traps_check_sync(task)

        assert errors == []

    def test_traps_check_detects_idiom(self) -> None:
        """Test traps check detects idioms."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_validator = MagicMock()
        mock_validator.is_available.return_value = True
        mock_validator.analyze_text.return_value = {
            "idioms": [
                {
                    "idiom": "бить баклуши",
                    "meaning": "to idle, to laze around",
                    "literal": "to hit wooden blocks",
                    "english_equivalent": "to twiddle one's thumbs",
                }
            ],
            "position_verbs": [],
            "homonyms": [],
            "paronyms": [],
        }

        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)
        agent.traps_validator = mock_validator

        task = TranslationTask(
            source_text="He idles",
            translation="Он бить баклуши",
            source_lang="en",
            target_lang="ru",
        )

        errors = agent._traps_check_sync(task)

        # Should detect the idiom
        assert len(errors) >= 1


@pytest.mark.unit
class TestCheckIdioms:
    """Test idiom checking."""

    def test_check_idioms_found(self) -> None:
        """Test detecting idiom in translation."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        translation = "он бить баклуши"
        analysis = {
            "idioms": [
                {
                    "idiom": "бить баклуши",
                    "meaning": "to idle",
                    "literal": "to hit blocks",
                    "english_equivalent": "twiddle thumbs",
                }
            ]
        }

        errors = agent._check_idioms(translation, analysis)

        assert len(errors) == 1
        assert errors[0].subcategory == "russian_idiom"
        assert errors[0].severity == ErrorSeverity.MAJOR

    def test_check_idioms_not_found(self) -> None:
        """Test no error when idiom not in text."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        translation = "он работает"
        analysis = {
            "idioms": [
                {
                    "idiom": "бить баклуши",
                    "meaning": "to idle",
                    "literal": "",
                    "english_equivalent": "",
                }
            ]
        }

        errors = agent._check_idioms(translation, analysis)

        assert len(errors) == 0


@pytest.mark.unit
class TestCheckHomonyms:
    """Test homonym checking."""

    def test_check_critical_homonym(self) -> None:
        """Test detecting critical homonym."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        translation = "замок открыт"
        analysis = {
            "homonyms": [
                {
                    "word": "замок",
                    "severity": "critical",
                    "meanings": [
                        {"meaning": "castle", "english": "castle"},
                        {"meaning": "lock", "english": "lock"},
                    ],
                }
            ]
        }

        errors = agent._check_homonyms(translation, analysis)

        assert len(errors) == 1
        assert errors[0].subcategory == "russian_homonym"
        assert "замок" in errors[0].description

    def test_skip_non_critical_homonym(self) -> None:
        """Test skipping non-critical homonyms."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        translation = "мука лежит"
        analysis = {
            "homonyms": [
                {
                    "word": "мука",
                    "severity": "minor",  # Not critical
                    "meanings": [{"meaning": "flour", "english": "flour"}],
                }
            ]
        }

        errors = agent._check_homonyms(translation, analysis)

        assert len(errors) == 0


@pytest.mark.unit
class TestEvaluate:
    """Test main evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_non_russian_fallback(self) -> None:
        """Test evaluate falls back to base for non-Russian."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",  # Not Russian
        )

        # Should use base fluency agent
        errors = await agent.evaluate(task)

        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_evaluate_russian_runs_all_checks(self) -> None:
        """Test evaluate runs all Russian-specific checks."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            return_value='{"errors": [], "text_type": "technical", "verification_summary": "No errors"}'
        )
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.return_value = []
        mock_helper.check_entity_preservation.return_value = []
        mock_helper.verify_error_position.return_value = True

        mock_case_validator = MagicMock()
        mock_case_validator.get_aspect_usage_rules.return_value = {}
        mock_case_validator.get_case_info.return_value = {}

        mock_traps_validator = MagicMock()
        mock_traps_validator.is_available.return_value = False

        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)
        agent.case_validator = mock_case_validator
        agent.traps_validator = mock_traps_validator

        task = TranslationTask(
            source_text="Hello world",
            translation="Привет мир",
            source_lang="en",
            target_lang="ru",
        )

        errors = await agent.evaluate(task)

        assert isinstance(errors, list)
        # LLM should have been called for Russian-specific checks
        assert mock_provider.complete.called

    @pytest.mark.asyncio
    async def test_evaluate_handles_exceptions(self) -> None:
        """Test evaluate handles exceptions gracefully."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.return_value = []
        mock_helper.check_entity_preservation.return_value = []
        mock_helper.verify_error_position.return_value = True
        mock_helper.get_enrichment_data.return_value = {"has_morphology": False}

        mock_case_validator = MagicMock()
        mock_case_validator.get_aspect_usage_rules.return_value = {}
        mock_case_validator.get_case_info.return_value = {}

        mock_traps_validator = MagicMock()
        mock_traps_validator.is_available.return_value = False

        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)
        agent.case_validator = mock_case_validator
        agent.traps_validator = mock_traps_validator

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        # Should not raise, should return list
        errors = await agent.evaluate(task)

        assert isinstance(errors, list)


@pytest.mark.unit
class TestBuildMorphologySection:
    """Test morphology section building for prompts."""

    def test_build_morphology_without_helper(self) -> None:
        """Test returns empty when helper unavailable."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = False
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        section = agent._build_morphology_section("Привет мир")

        assert section == ""

    def test_build_morphology_without_morphology_data(self) -> None:
        """Test returns empty when no morphology data."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {"has_morphology": False}
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        section = agent._build_morphology_section("Привет мир")

        assert section == ""

    def test_build_morphology_with_data(self) -> None:
        """Test builds section with morphology data."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "verb_aspects": {"идёт": {"aspect_name": "imperfective"}},
            "adjective_noun_pairs": [
                {
                    "adjective": {"word": "красивая", "gender": "fem", "case": "nom"},
                    "noun": {"word": "кошка", "gender": "fem", "case": "nom"},
                    "agreement": "correct",
                }
            ],
        }
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        section = agent._build_morphology_section("красивая кошка идёт")

        assert "MORPHOLOGICAL ANALYSIS" in section
        assert "verb_aspects" in section or "Verbs" in section


@pytest.mark.unit
class TestTechPatterns:
    """Test technical term pattern matching."""

    def test_is_tech_term_gen5(self) -> None:
        """Test detecting Gen 5 as tech term."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        assert agent._is_tech_term_fp("Gen 5", "supports Gen 5") is True

    def test_is_tech_term_wifi7(self) -> None:
        """Test detecting Wi-Fi 7 as tech term."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        assert agent._is_tech_term_fp("Wi-Fi 7", "supports Wi-Fi 7") is True

    def test_is_tech_term_ddr5(self) -> None:
        """Test detecting DDR5 as tech term."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        assert agent._is_tech_term_fp("DDR5", "память DDR5") is True

    def test_not_tech_term(self) -> None:
        """Test regular text is not flagged as tech term."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        assert agent._is_tech_term_fp("пять", "пять минут") is False


@pytest.mark.unit
class TestDigitGenitivePatterns:
    """Test digit+genitive pattern matching."""

    def test_is_digit_genitive_minutes(self) -> None:
        """Test detecting digit+minutes pattern."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        assert agent._is_digit_genitive_location_fp("5 минут", "через 5 минут") is True

    def test_is_digit_genitive_hours(self) -> None:
        """Test detecting digit+hours pattern."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        assert agent._is_digit_genitive_location_fp("10 часов", "прошло 10 часов") is True

    def test_not_digit_genitive(self) -> None:
        """Test regular text is not flagged as digit+genitive."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        agent = RussianFluencyAgent(mock_provider, helper=mock_helper)

        assert agent._is_digit_genitive_location_fp("кошка", "красивая кошка") is False
