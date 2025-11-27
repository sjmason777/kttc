"""Unit tests for style preservation agent module.

Tests literary style preservation evaluation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.agents.style_preservation import (
    STYLE_PRESERVATION_PROMPT,
    StylePreservationAgent,
)
from kttc.core import ErrorSeverity, TranslationTask


@pytest.mark.unit
class TestStylePreservationAgentInitialization:
    """Test style preservation agent initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        assert agent.llm_provider == mock_provider
        assert agent.temperature == 0.1
        assert agent.max_tokens == 2500
        assert agent.style_profile is None

    def test_custom_initialization(self) -> None:
        """Test custom initialization parameters."""
        mock_provider = MagicMock()
        mock_profile = MagicMock()
        agent = StylePreservationAgent(
            mock_provider, temperature=0.3, max_tokens=3000, style_profile=mock_profile
        )

        assert agent.temperature == 0.3
        assert agent.max_tokens == 3000
        assert agent.style_profile == mock_profile

    def test_category_property(self) -> None:
        """Test category property returns correct value."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        assert agent.category == "style_preservation"

    def test_get_base_prompt(self) -> None:
        """Test get_base_prompt returns the template."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        prompt = agent.get_base_prompt()
        assert prompt == STYLE_PRESERVATION_PROMPT


@pytest.mark.unit
class TestStyleProfile:
    """Test style profile management."""

    def test_set_style_profile(self) -> None:
        """Test setting style profile."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_profile = MagicMock()
        agent.set_style_profile(mock_profile)

        assert agent.style_profile == mock_profile


@pytest.mark.unit
class TestBuildStyleContext:
    """Test style context building."""

    def test_build_context_no_profile(self) -> None:
        """Test building context without profile."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        context = agent._build_style_context()

        assert "No pre-analysis available" in context

    def test_build_context_with_profile(self) -> None:
        """Test building context with style profile."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_profile = MagicMock()
        mock_profile.deviation_score = 0.5
        mock_profile.detected_pattern.value = "literary"
        mock_profile.is_literary = True
        mock_profile.detected_deviations = []

        agent.style_profile = mock_profile

        context = agent._build_style_context()

        assert "0.50" in context
        assert "literary" in context
        assert "Is literary: True" in context

    def test_build_context_with_deviations(self) -> None:
        """Test building context with detected deviations."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_deviation = MagicMock()
        mock_deviation.type.value = "pleonasm"
        mock_deviation.interpretation = "Intentional redundancy"
        mock_deviation.examples = ["live a life", "dream a dream"]

        mock_profile = MagicMock()
        mock_profile.deviation_score = 0.4
        mock_profile.detected_pattern.value = "ornate"
        mock_profile.is_literary = True
        mock_profile.detected_deviations = [mock_deviation]

        agent.style_profile = mock_profile

        context = agent._build_style_context()

        assert "INTENTIONAL DEVIATIONS" in context
        assert "pleonasm" in context
        assert "Intentional redundancy" in context
        assert "live a life" in context


@pytest.mark.unit
class TestBuildAdditionalQuestions:
    """Test additional questions building."""

    def test_build_questions_no_profile(self) -> None:
        """Test building questions without profile."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        questions = agent._build_additional_questions()

        assert questions == ""

    def test_build_questions_no_deviations(self) -> None:
        """Test building questions with no deviations."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_profile = MagicMock()
        mock_profile.detected_deviations = []
        agent.style_profile = mock_profile

        questions = agent._build_additional_questions()

        assert questions == ""

    def test_build_questions_pleonasm(self) -> None:
        """Test building questions for pleonasm deviation."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_deviation = MagicMock()
        mock_deviation.type.value = "pleonasm"

        mock_profile = MagicMock()
        mock_profile.detected_deviations = [mock_deviation]
        agent.style_profile = mock_profile

        questions = agent._build_additional_questions()

        assert "redundancies/repetitions" in questions

    def test_build_questions_skaz(self) -> None:
        """Test building questions for skaz deviation."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_deviation = MagicMock()
        mock_deviation.type.value = "skaz"

        mock_profile = MagicMock()
        mock_profile.detected_deviations = [mock_deviation]
        agent.style_profile = mock_profile

        questions = agent._build_additional_questions()

        assert "folk/oral storytelling" in questions

    def test_build_questions_syntactic_inversion(self) -> None:
        """Test building questions for syntactic inversion."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_deviation = MagicMock()
        mock_deviation.type.value = "syntactic_inversion"

        mock_profile = MagicMock()
        mock_profile.detected_deviations = [mock_deviation]
        agent.style_profile = mock_profile

        questions = agent._build_additional_questions()

        assert "word order patterns" in questions

    def test_build_questions_dedup(self) -> None:
        """Test that duplicate deviation types are deduplicated."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        mock_deviation1 = MagicMock()
        mock_deviation1.type.value = "pleonasm"
        mock_deviation2 = MagicMock()
        mock_deviation2.type.value = "pleonasm"

        mock_profile = MagicMock()
        mock_profile.detected_deviations = [mock_deviation1, mock_deviation2]
        agent.style_profile = mock_profile

        questions = agent._build_additional_questions()

        # Should only have one question for pleonasm
        assert questions.count("redundancies/repetitions") == 1


@pytest.mark.unit
class TestParseStyleErrors:
    """Test style error parsing."""

    def test_parse_no_errors(self) -> None:
        """Test parsing response with no errors."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        response = """
        Analysis complete.
        NO_STYLE_ERRORS_FOUND
        """

        errors = agent._parse_style_errors(response)

        assert len(errors) == 0

    def test_parse_single_error(self) -> None:
        """Test parsing single error from response."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        response = """
        STYLE PRESERVATION SCORE: 75

        ERROR: style_preservation | voice_loss | major | The author's distinctive tone was normalized.
        """

        errors = agent._parse_style_errors(response)

        assert len(errors) == 1
        assert errors[0].category == "style_preservation"
        assert errors[0].subcategory == "voice_loss"
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert "distinctive tone" in errors[0].description

    def test_parse_multiple_errors(self) -> None:
        """Test parsing multiple errors from response."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        response = """
        ERROR: style_preservation | voice_loss | major | Tone was lost.
        ERROR: style_preservation | rhythm_lost | minor | Prose rhythm altered.
        ERROR: style_preservation | device_removed | critical | Literary device removed.
        """

        errors = agent._parse_style_errors(response)

        assert len(errors) == 3
        assert errors[0].severity == ErrorSeverity.MAJOR
        assert errors[1].severity == ErrorSeverity.MINOR
        assert errors[2].severity == ErrorSeverity.CRITICAL

    def test_parse_normalizes_category(self) -> None:
        """Test that non-style_preservation category is normalized."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        response = """
        ERROR: wrong_category | voice_loss | major | Test description.
        """

        errors = agent._parse_style_errors(response)

        assert len(errors) == 1
        assert errors[0].category == "style_preservation"

    def test_parse_invalid_format(self) -> None:
        """Test parsing response with insufficient parts."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        response = """
        ERROR: not enough parts
        Normal text without errors.
        """

        errors = agent._parse_style_errors(response)

        # Should skip invalid lines gracefully (needs at least 4 parts)
        assert len(errors) == 0

    def test_parse_unknown_severity(self) -> None:
        """Test parsing with unknown severity defaults to minor."""
        mock_provider = MagicMock()
        agent = StylePreservationAgent(mock_provider)

        response = """
        ERROR: style_preservation | test | unknown_severity | Description.
        """

        errors = agent._parse_style_errors(response)

        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.MINOR


@pytest.mark.unit
class TestEvaluate:
    """Test the main evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_no_errors(self) -> None:
        """Test evaluate returns no errors for good translation."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            return_value="STYLE PRESERVATION SCORE: 95\n\nNO_STYLE_ERRORS_FOUND"
        )

        agent = StylePreservationAgent(mock_provider)

        task = TranslationTask(
            source_text="The wind whispered through the ancient trees.",
            translation="El viento susurraba entre los árboles antiguos.",
            source_lang="en",
            target_lang="es",
        )

        errors = await agent.evaluate(task)

        assert len(errors) == 0
        mock_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_with_style_profile(self) -> None:
        """Test evaluate uses style profile in prompt."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value="NO_STYLE_ERRORS_FOUND")

        mock_profile = MagicMock()
        mock_profile.deviation_score = 0.5
        mock_profile.detected_pattern.value = "literary"
        mock_profile.is_literary = True
        mock_profile.detected_deviations = []

        agent = StylePreservationAgent(mock_provider, style_profile=mock_profile)

        task = TranslationTask(
            source_text="Test source.",
            translation="Test translation.",
            source_lang="en",
            target_lang="es",
        )

        await agent.evaluate(task)

        # Check that prompt includes style context
        call_args = mock_provider.complete.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[0][0]
        assert "deviation score" in prompt.lower()

    @pytest.mark.asyncio
    async def test_evaluate_detects_errors(self) -> None:
        """Test evaluate detects style errors."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            return_value="""
            STYLE PRESERVATION SCORE: 60

            ERROR: style_preservation | voice_loss | major | The author's unique voice was normalized.
            ERROR: style_preservation | rhythm_lost | minor | Sentence rhythm was altered.
            """
        )

        agent = StylePreservationAgent(mock_provider)

        task = TranslationTask(
            source_text="He lived a life of living.",
            translation="He lived.",  # Lost the pleonasm
            source_lang="en",
            target_lang="es",
        )

        errors = await agent.evaluate(task)

        assert len(errors) == 2

    @pytest.mark.asyncio
    async def test_evaluate_handles_llm_error(self) -> None:
        """Test evaluate raises AgentEvaluationError on LLM failure."""
        from kttc.agents.base import AgentEvaluationError
        from kttc.llm import LLMError

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=LLMError("API error"))

        agent = StylePreservationAgent(mock_provider)

        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        with pytest.raises(AgentEvaluationError, match="LLM evaluation failed"):
            await agent.evaluate(task)

    @pytest.mark.asyncio
    async def test_evaluate_handles_unexpected_error(self) -> None:
        """Test evaluate handles unexpected exceptions."""
        from kttc.agents.base import AgentEvaluationError

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("Unexpected"))

        agent = StylePreservationAgent(mock_provider)

        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        with pytest.raises(AgentEvaluationError, match="Unexpected error"):
            await agent.evaluate(task)


@pytest.mark.unit
class TestPromptTemplate:
    """Test the prompt template."""

    def test_prompt_contains_required_sections(self) -> None:
        """Test prompt template has all required sections."""
        assert "authorial voice" in STYLE_PRESERVATION_PROMPT.lower()
        assert "STYLE PRESERVATION SCORE" in STYLE_PRESERVATION_PROMPT
        assert "ERROR:" in STYLE_PRESERVATION_PROMPT
        assert "NO_STYLE_ERRORS_FOUND" in STYLE_PRESERVATION_PROMPT
        assert "{source_text}" in STYLE_PRESERVATION_PROMPT
        assert "{translation}" in STYLE_PRESERVATION_PROMPT

    def test_prompt_evaluation_questions(self) -> None:
        """Test prompt has evaluation questions."""
        assert "distinctive tone" in STYLE_PRESERVATION_PROMPT.lower()
        assert "literary devices" in STYLE_PRESERVATION_PROMPT.lower()
        assert "rhythm" in STYLE_PRESERVATION_PROMPT.lower()
