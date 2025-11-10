"""Unit tests for TerminologyAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from kttc.agents import AgentEvaluationError, TerminologyAgent
from kttc.core import ErrorSeverity, TranslationTask
from kttc.llm import LLMError, OpenAIProvider

# Configure anyio for async tests
pytestmark = pytest.mark.anyio


class TestTerminologyAgent:
    """Tests for TerminologyAgent."""

    async def test_evaluate_with_errors(self) -> None:
        """Test TerminologyAgent finds terminology errors."""
        # Create mock provider
        provider = OpenAIProvider(api_key="test-key")

        # Mock LLM response with terminology errors
        mock_response = """
ERROR_START
CATEGORY: terminology
SUBCATEGORY: inconsistency
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Term "API" translated inconsistently
SUGGESTION: Use consistent term throughout
ERROR_END

ERROR_START
CATEGORY: terminology
SUBCATEGORY: misuse
SEVERITY: critical
LOCATION: 15-20
DESCRIPTION: Wrong technical term used
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="The API uses REST architecture",
                translation="La interfaz usa arquitectura REST",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 2
            assert all(error.category == "terminology" for error in errors)
            assert errors[0].subcategory == "inconsistency"
            assert errors[0].severity == ErrorSeverity.MAJOR
            assert errors[1].subcategory == "misuse"
            assert errors[1].severity == ErrorSeverity.CRITICAL

    async def test_evaluate_perfect_terminology(self) -> None:
        """Test TerminologyAgent with perfect terminology."""
        provider = OpenAIProvider(api_key="test-key")
        mock_response = "No terminology errors found."

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="Cloud computing is important",
                translation="La computación en la nube es importante",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)
            assert errors == []

    async def test_evaluate_filters_wrong_category(self) -> None:
        """Test TerminologyAgent filters out non-terminology errors."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = """
ERROR_START
CATEGORY: terminology
SUBCATEGORY: misuse
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Wrong technical term
ERROR_END

ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: minor
LOCATION: 10-20
DESCRIPTION: This should be filtered out
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 1
            assert errors[0].category == "terminology"

    async def test_evaluate_llm_error(self) -> None:
        """Test TerminologyAgent handles LLM errors."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(provider, "complete", new=AsyncMock(side_effect=LLMError("API error"))):
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            with pytest.raises(AgentEvaluationError, match="LLM evaluation failed"):
                await agent.evaluate(task)

    async def test_evaluate_unexpected_error(self) -> None:
        """Test TerminologyAgent handles unexpected errors."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider, "complete", new=AsyncMock(side_effect=RuntimeError("Unexpected"))
        ):
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            with pytest.raises(AgentEvaluationError, match="Unexpected error"):
                await agent.evaluate(task)

    async def test_category_property(self) -> None:
        """Test TerminologyAgent category property."""
        provider = OpenAIProvider(api_key="test-key")
        agent = TerminologyAgent(provider)

        assert agent.category == "terminology"

    async def test_custom_temperature_and_tokens(self) -> None:
        """Test TerminologyAgent with custom temperature and max_tokens."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider, "complete", new=AsyncMock(return_value="No errors")
        ) as mock_complete:
            agent = TerminologyAgent(provider, temperature=0.5, max_tokens=1000)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            await agent.evaluate(task)

            # Verify LLM was called with custom parameters
            mock_complete.assert_called_once()
            call_kwargs = mock_complete.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 1000

    async def test_prompt_template_formatting(self) -> None:
        """Test TerminologyAgent formats prompt template correctly."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider, "complete", new=AsyncMock(return_value="No errors")
        ) as mock_complete:
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="API REST",
                translation="REST API",
                source_lang="en",
                target_lang="es",
            )

            await agent.evaluate(task)

            # Verify prompt contains task data
            call_args = mock_complete.call_args
            prompt = call_args.kwargs["prompt"]
            assert "API REST" in prompt
            assert "REST API" in prompt
            assert "en" in prompt
            assert "es" in prompt
            assert "terminology" in prompt.lower()

    async def test_evaluate_with_malformed_errors(self) -> None:
        """Test TerminologyAgent gracefully handles malformed error blocks."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = """
ERROR_START
CATEGORY: terminology
SUBCATEGORY: misuse
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Valid error
ERROR_END

ERROR_START
CATEGORY: terminology
MISSING_FIELDS: true
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            # Should only return valid error
            assert len(errors) == 1
            assert errors[0].description == "Valid error"

    async def test_evaluate_real_translation_scenario(self) -> None:
        """Test TerminologyAgent with realistic translation scenario."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = """
ERROR_START
CATEGORY: terminology
SUBCATEGORY: inconsistency
SEVERITY: major
LOCATION: 10-13
DESCRIPTION: Term "API" translated as "interfaz" here but left as "API" elsewhere
SUGGESTION: Use "API" consistently (established technical term)
ERROR_END

ERROR_START
CATEGORY: terminology
SUBCATEGORY: untranslated
SEVERITY: minor
LOCATION: 20-28
DESCRIPTION: Acronym "RAM" left untranslated when "memoria RAM" could be used
SUGGESTION: Use "memoria RAM" for clarity
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = TerminologyAgent(provider)
            task = TranslationTask(
                source_text="The API allocates RAM efficiently",
                translation="La interfaz asigna RAM eficientemente",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 2
            assert errors[0].subcategory == "inconsistency"
            assert errors[0].suggestion == 'Use "API" consistently (established technical term)'
            assert errors[1].subcategory == "untranslated"
            assert errors[1].severity == ErrorSeverity.MINOR
