"""Unit tests for FluencyAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from kttc.agents import AgentEvaluationError, FluencyAgent
from kttc.core import ErrorSeverity, TranslationTask
from kttc.llm import LLMError, OpenAIProvider

# Configure anyio for async tests
pytestmark = pytest.mark.anyio


class TestFluencyAgent:
    """Tests for FluencyAgent."""

    async def test_evaluate_with_errors(self) -> None:
        """Test FluencyAgent finds fluency errors."""
        # Create mock provider
        provider = OpenAIProvider(api_key="test-key")

        # Mock LLM response with fluency errors
        mock_response = """
ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Incorrect verb tense
SUGGESTION: Use past tense instead
ERROR_END

ERROR_START
CATEGORY: fluency
SUBCATEGORY: spelling
SEVERITY: minor
LOCATION: 15-20
DESCRIPTION: Spelling mistake
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = FluencyAgent(provider)
            task = TranslationTask(
                source_text="The cat sat on the mat",
                translation="El gato sentó en la alfombra",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 2
            assert all(error.category == "fluency" for error in errors)
            assert errors[0].subcategory == "grammar"
            assert errors[0].severity == ErrorSeverity.MAJOR
            assert errors[1].subcategory == "spelling"
            assert errors[1].severity == ErrorSeverity.MINOR

    async def test_evaluate_perfect_translation(self) -> None:
        """Test FluencyAgent with perfect fluency."""
        provider = OpenAIProvider(api_key="test-key")
        mock_response = "No fluency errors found."

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = FluencyAgent(provider)
            task = TranslationTask(
                source_text="Hello, world!",
                translation="¡Hola, mundo!",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)
            assert errors == []

    async def test_evaluate_filters_wrong_category(self) -> None:
        """Test FluencyAgent filters out non-fluency errors."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = """
ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Grammar error
ERROR_END

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: critical
LOCATION: 10-20
DESCRIPTION: This should be filtered out
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = FluencyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 1
            assert errors[0].category == "fluency"

    async def test_evaluate_llm_error(self) -> None:
        """Test FluencyAgent handles LLM errors."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(provider, "complete", new=AsyncMock(side_effect=LLMError("API error"))):
            agent = FluencyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            with pytest.raises(AgentEvaluationError, match="LLM evaluation failed"):
                await agent.evaluate(task)

    async def test_evaluate_unexpected_error(self) -> None:
        """Test FluencyAgent handles unexpected errors."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider, "complete", new=AsyncMock(side_effect=RuntimeError("Unexpected"))
        ):
            agent = FluencyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            with pytest.raises(AgentEvaluationError, match="Unexpected error"):
                await agent.evaluate(task)

    async def test_category_property(self) -> None:
        """Test FluencyAgent category property."""
        provider = OpenAIProvider(api_key="test-key")
        agent = FluencyAgent(provider)

        assert agent.category == "fluency"

    async def test_custom_temperature_and_tokens(self) -> None:
        """Test FluencyAgent with custom temperature and max_tokens."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider, "complete", new=AsyncMock(return_value="No errors")
        ) as mock_complete:
            agent = FluencyAgent(provider, temperature=0.5, max_tokens=1000)
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
        """Test FluencyAgent formats prompt template correctly."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider, "complete", new=AsyncMock(return_value="No errors")
        ) as mock_complete:
            agent = FluencyAgent(provider)
            task = TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            )

            await agent.evaluate(task)

            # Verify prompt contains task data
            call_args = mock_complete.call_args
            prompt = call_args.kwargs["prompt"]
            assert "Hello" in prompt
            assert "Hola" in prompt
            assert "en" in prompt
            assert "es" in prompt
            assert "fluency" in prompt.lower()

    async def test_evaluate_with_malformed_errors(self) -> None:
        """Test FluencyAgent gracefully handles malformed error blocks."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = """
ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Valid error
ERROR_END

ERROR_START
CATEGORY: fluency
MISSING_FIELDS: true
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = FluencyAgent(provider)
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
        """Test FluencyAgent with realistic translation scenario."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = """
ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: major
LOCATION: 11-16
DESCRIPTION: Incorrect article usage - should be 'the' not 'a'
SUGGESTION: Change 'a cat' to 'the cat'
ERROR_END

ERROR_START
CATEGORY: fluency
SUBCATEGORY: readability
SEVERITY: minor
LOCATION: 20-35
DESCRIPTION: Awkward phrasing - more natural would be 'is sitting'
SUGGESTION: Replace 'sits' with 'is sitting'
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=mock_response)):
            agent = FluencyAgent(provider)
            task = TranslationTask(
                source_text="The cat is on the mat",
                translation="A cat sits on the mat",
                source_lang="en",
                target_lang="en",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 2
            assert errors[0].subcategory == "grammar"
            assert errors[0].suggestion == "Change 'a cat' to 'the cat'"
            assert errors[1].subcategory == "readability"
            assert errors[1].severity == ErrorSeverity.MINOR
