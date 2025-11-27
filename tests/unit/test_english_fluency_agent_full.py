"""Unit tests for English fluency agent.

Tests English-specific fluency checking with hybrid LanguageTool + LLM approach.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kttc.agents.fluency_english import EnglishFluencyAgent
from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask


@pytest.mark.unit
class TestEnglishFluencyAgentInitialization:
    """Test English fluency agent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        mock_provider = MagicMock()
        with (
            patch("kttc.agents.fluency_english.EnglishLanguageHelper"),
            patch("kttc.agents.fluency_english.EnglishTrapsValidator"),
        ):
            agent = EnglishFluencyAgent(mock_provider)

        assert agent.llm_provider == mock_provider
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2000
        assert agent.helper is not None
        assert agent.traps_validator is not None

    def test_custom_initialization(self) -> None:
        """Test custom initialization parameters."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(
            mock_provider,
            temperature=0.3,
            max_tokens=3000,
            helper=mock_helper,
            traps_validator=mock_traps,
        )

        assert agent.temperature == 0.3
        assert agent.max_tokens == 3000
        assert agent.helper == mock_helper
        assert agent.traps_validator == mock_traps

    def test_english_checks_defined(self) -> None:
        """Test that English-specific checks are defined."""
        assert "grammar" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "spelling" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "tense" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "homophones" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "phrasal_verbs" in EnglishFluencyAgent.ENGLISH_CHECKS
        assert "idioms" in EnglishFluencyAgent.ENGLISH_CHECKS


@pytest.mark.unit
class TestGetBasePrompt:
    """Test base prompt generation."""

    def test_get_base_prompt_includes_english_section(self) -> None:
        """Test that base prompt includes English-specific section."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        prompt = agent.get_base_prompt()

        assert "ENGLISH-SPECIFIC CHECKS" in prompt
        assert "English-specific linguistic validation" in prompt


@pytest.mark.unit
class TestLanguageToolCheckSync:
    """Test LanguageTool-based grammar checks."""

    def test_languagetool_check_without_helper(self) -> None:
        """Test LanguageTool check returns empty when helper unavailable."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = False
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет",
            translation="Hello",
            source_lang="ru",
            target_lang="en",
        )

        errors = agent._languagetool_check_sync(task)

        assert errors == []

    def test_languagetool_check_with_helper(self) -> None:
        """Test LanguageTool check calls helper."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.return_value = []
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет",
            translation="Hello",
            source_lang="ru",
            target_lang="en",
        )

        errors = agent._languagetool_check_sync(task)

        mock_helper.check_grammar.assert_called_once_with("Hello")
        assert errors == []

    def test_languagetool_check_handles_exception(self) -> None:
        """Test LanguageTool check handles exceptions gracefully."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.side_effect = Exception("LanguageTool error")
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет",
            translation="Hello",
            source_lang="ru",
            target_lang="en",
        )

        errors = agent._languagetool_check_sync(task)

        assert errors == []


@pytest.mark.unit
class TestTrapsCheckSync:
    """Test English traps validation."""

    def test_traps_check_without_validator(self) -> None:
        """Test traps check returns empty when validator unavailable."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        mock_traps.is_available.return_value = False
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет",
            translation="Hello there",
            source_lang="ru",
            target_lang="en",
        )

        errors = agent._traps_check_sync(task)

        assert errors == []

    def test_traps_check_with_validator(self) -> None:
        """Test traps check with available validator."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        mock_traps.is_available.return_value = True
        mock_traps.find_homophones_in_text.return_value = []
        mock_traps.find_phrasal_verbs_in_text.return_value = []
        mock_traps.check_adjective_order.return_value = []
        mock_traps.find_preposition_errors.return_value = []
        mock_traps.find_idioms_in_text.return_value = []

        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет",
            translation="Hello there",
            source_lang="ru",
            target_lang="en",
        )

        errors = agent._traps_check_sync(task)

        # With no traps detected, should return empty list
        assert errors == []
        mock_traps.find_homophones_in_text.assert_called_once_with("Hello there")

    def test_traps_check_handles_exception(self) -> None:
        """Test traps check handles exceptions gracefully."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        mock_traps.is_available.return_value = True
        mock_traps.analyze_text.side_effect = Exception("Traps error")

        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет",
            translation="Hello there",
            source_lang="ru",
            target_lang="en",
        )

        errors = agent._traps_check_sync(task)

        assert errors == []


@pytest.mark.unit
class TestVerifyLLMErrors:
    """Test LLM error verification."""

    def test_verify_errors_without_helper(self) -> None:
        """Test that without helper all errors are returned."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = False
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test error",
        )

        verified = agent._verify_llm_errors([error], "Hello world")

        assert len(verified) == 1
        assert verified[0] == error

    def test_verify_errors_with_valid_position(self) -> None:
        """Test verifying errors with valid positions."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.verify_error_position.return_value = True
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test error",
        )

        verified = agent._verify_llm_errors([error], "Hello world")

        assert len(verified) == 1
        mock_helper.verify_error_position.assert_called_once()


@pytest.mark.unit
class TestRemoveDuplicates:
    """Test duplicate error removal."""

    def test_remove_duplicates_with_overlap(self) -> None:
        """Test removing duplicates when errors overlap."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        lt_error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(5, 10),
            description="LanguageTool error",
        )
        llm_error = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 15),
            description="LLM error",
        )

        unique = agent._remove_duplicates([lt_error], [llm_error])

        assert len(unique) == 0  # LT error overlaps with LLM error


@pytest.mark.unit
class TestParseJSONResponse:
    """Test JSON response parsing."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        response = '{"errors": [], "text_type": "formal"}'
        result = agent._parse_json_response(response)

        assert result["errors"] == []
        assert result["text_type"] == "formal"

    def test_parse_json_from_markdown(self) -> None:
        """Test parsing JSON from markdown code block."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        response = """Analysis:
```json
{"errors": [{"subcategory": "grammar"}]}
```
Done."""

        result = agent._parse_json_response(response)

        assert len(result["errors"]) == 1
        assert result["errors"][0]["subcategory"] == "grammar"

    def test_parse_invalid_json_returns_empty(self) -> None:
        """Test parsing invalid JSON returns empty errors."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        response = "This is not JSON at all"
        result = agent._parse_json_response(response)

        assert result["errors"] == []


@pytest.mark.unit
class TestEvaluate:
    """Test main evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_non_english_fallback(self) -> None:
        """Test evaluate falls back to base for non-English."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": []}')
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",  # Not English
        )

        # Should use base fluency agent
        errors = await agent.evaluate(task)

        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_evaluate_english_runs_all_checks(self) -> None:
        """Test evaluate runs all English-specific checks."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value='{"errors": [], "text_type": "formal"}')
        mock_helper = MagicMock()
        mock_helper.is_available.return_value = True
        mock_helper.check_grammar.return_value = []
        mock_helper.verify_error_position.return_value = True

        mock_traps = MagicMock()
        mock_traps.is_available.return_value = False

        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

        task = TranslationTask(
            source_text="Привет мир",
            translation="Hello world",
            source_lang="ru",
            target_lang="en",
        )

        errors = await agent.evaluate(task)

        assert isinstance(errors, list)
        # LLM should have been called for English-specific checks
        assert mock_provider.complete.called


@pytest.mark.unit
class TestErrorsOverlap:
    """Test error overlap detection."""

    def test_errors_overlap_true(self) -> None:
        """Test detecting overlapping errors."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

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

        assert agent._errors_overlap(error1, error2) is True

    def test_errors_overlap_false(self) -> None:
        """Test non-overlapping errors."""
        mock_provider = MagicMock()
        mock_helper = MagicMock()
        mock_traps = MagicMock()
        agent = EnglishFluencyAgent(mock_provider, helper=mock_helper, traps_validator=mock_traps)

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
            location=(10, 20),
            description="Test",
        )

        assert agent._errors_overlap(error1, error2) is False
