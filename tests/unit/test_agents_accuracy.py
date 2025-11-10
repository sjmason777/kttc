"""Unit tests for AccuracyAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from kttc.agents import AccuracyAgent, AgentEvaluationError
from kttc.core import ErrorSeverity, TranslationTask
from kttc.llm import LLMError, OpenAIProvider

# Configure anyio for async tests
pytestmark = pytest.mark.anyio


class TestAccuracyAgent:
    """Test AccuracyAgent for translation accuracy evaluation."""

    async def test_evaluate_with_errors(self) -> None:
        """Test evaluation that finds accuracy errors."""
        # Create mock provider
        provider = OpenAIProvider(api_key="test-key")

        # Mock LLM response with errors
        llm_response = """
The translation has the following issues:

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-3
DESCRIPTION: 'Cat' mistranslated as 'dog'
SUGGESTION: Use correct animal name
ERROR_END

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: omission
SEVERITY: critical
LOCATION: 10-15
DESCRIPTION: Missing word 'on'
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=llm_response)):
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="The cat is on the mat",
                translation="El perro está la alfombra",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 2
            assert errors[0].category == "accuracy"
            assert errors[0].subcategory == "mistranslation"
            assert errors[0].severity == ErrorSeverity.MAJOR
            assert errors[1].subcategory == "omission"
            assert errors[1].severity == ErrorSeverity.CRITICAL

    async def test_evaluate_perfect_translation(self) -> None:
        """Test evaluation of perfect translation with no errors."""
        provider = OpenAIProvider(api_key="test-key")

        llm_response = """
The translation is accurate and faithful to the source.
No accuracy errors were found.
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=llm_response)):
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="Hello, world!",
                translation="Hola, mundo!",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)
            assert len(errors) == 0

    async def test_evaluate_filters_wrong_category(self) -> None:
        """Test that agent filters out errors from wrong categories."""
        provider = OpenAIProvider(api_key="test-key")

        # LLM returns mixed categories (accuracy + fluency)
        llm_response = """
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Accuracy error
ERROR_END

ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: minor
LOCATION: 10-15
DESCRIPTION: Grammar error (should be filtered out)
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=llm_response)):
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            # Should only return accuracy errors
            assert len(errors) == 1
            assert errors[0].category == "accuracy"

    async def test_evaluate_llm_error(self) -> None:
        """Test handling of LLM errors during evaluation."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider,
            "complete",
            new=AsyncMock(side_effect=LLMError("API error")),
        ):
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            with pytest.raises(AgentEvaluationError, match="LLM evaluation failed"):
                await agent.evaluate(task)

    async def test_evaluate_unexpected_error(self) -> None:
        """Test handling of unexpected errors during evaluation."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(
            provider,
            "complete",
            new=AsyncMock(side_effect=RuntimeError("Unexpected error")),
        ):
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            with pytest.raises(AgentEvaluationError, match="Unexpected error"):
                await agent.evaluate(task)

    async def test_category_property(self) -> None:
        """Test agent category property."""
        provider = OpenAIProvider(api_key="test-key")
        agent = AccuracyAgent(provider)
        assert agent.category == "accuracy"

    async def test_custom_temperature_and_tokens(self) -> None:
        """Test agent with custom temperature and max_tokens settings."""
        provider = OpenAIProvider(api_key="test-key")

        llm_response = "No errors found."

        with patch.object(
            provider, "complete", new=AsyncMock(return_value=llm_response)
        ) as mock_complete:
            agent = AccuracyAgent(provider, temperature=0.5, max_tokens=1000)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            await agent.evaluate(task)

            # Verify custom settings were used
            mock_complete.assert_called_once()
            call_kwargs = mock_complete.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 1000

    async def test_prompt_template_formatting(self) -> None:
        """Test that prompt template is correctly formatted with task data."""
        provider = OpenAIProvider(api_key="test-key")

        llm_response = "No errors."

        with patch.object(
            provider, "complete", new=AsyncMock(return_value=llm_response)
        ) as mock_complete:
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            )

            await agent.evaluate(task)

            # Verify prompt contains task data
            prompt = mock_complete.call_args.kwargs["prompt"]
            assert "Hello" in prompt
            assert "Hola" in prompt
            assert "en" in prompt or "English" in prompt
            assert "es" in prompt or "Spanish" in prompt

    async def test_evaluate_with_malformed_errors(self) -> None:
        """Test that malformed errors are skipped gracefully."""
        provider = OpenAIProvider(api_key="test-key")

        # Mix of valid and malformed errors
        llm_response = """
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Valid error
ERROR_END

ERROR_START
CATEGORY: accuracy
SEVERITY: major
LOCATION: 10-15
DESCRIPTION: Missing subcategory (malformed)
ERROR_END

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: omission
SEVERITY: invalid_severity
LOCATION: 20-25
DESCRIPTION: Invalid severity (malformed)
ERROR_END
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=llm_response)):
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="Test",
                translation="Prueba",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            # Should only return valid errors
            assert len(errors) == 1
            assert errors[0].description == "Valid error"

    async def test_evaluate_real_translation_scenario(self) -> None:
        """Test realistic translation evaluation scenario."""
        provider = OpenAIProvider(api_key="test-key")

        # Realistic LLM response
        llm_response = """
Analysis of the translation:

The source says "patient" but the translation uses "cliente" (client).
This is medically incorrect.

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: critical
LOCATION: 4-12
DESCRIPTION: Medical term 'patient' mistranslated as 'cliente' (client)
SUGGESTION: Use 'paciente' for medical context
ERROR_END

The word "needs" is completely missing from the translation.

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: omission
SEVERITY: major
LOCATION: 13-18
DESCRIPTION: The verb 'needs' is omitted in translation
SUGGESTION: Add 'necesita' to complete the meaning
ERROR_END

Overall, the translation has serious accuracy issues that could cause confusion
in a medical setting.
"""

        with patch.object(provider, "complete", new=AsyncMock(return_value=llm_response)):
            agent = AccuracyAgent(provider)
            task = TranslationTask(
                source_text="The patient needs treatment",
                translation="El cliente tratamiento",
                source_lang="en",
                target_lang="es",
            )

            errors = await agent.evaluate(task)

            assert len(errors) == 2
            assert errors[0].severity == ErrorSeverity.CRITICAL
            assert "patient" in errors[0].description.lower()
            assert errors[1].severity == ErrorSeverity.MAJOR
            assert "omission" in errors[1].subcategory.lower()
