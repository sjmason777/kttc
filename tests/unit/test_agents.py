"""Unit tests for QA Agents.

Tests agent logic with mocked LLM responses.
Focus: Fast, isolated tests without real API calls.
"""

from typing import Any

import pytest

from kttc.agents import AccuracyAgent, FluencyAgent, TerminologyAgent
from kttc.core.models import TranslationTask


@pytest.mark.unit
class TestAccuracyAgent:
    """Test AccuracyAgent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_evaluate_no_errors(
        self, mock_llm: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test evaluation returns no errors for perfect translation."""
        # Arrange
        agent = AccuracyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert isinstance(errors, list)
        assert len(errors) == 0
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_errors(
        self, mock_llm_with_errors: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test evaluation returns errors from LLM."""
        # Arrange
        agent = AccuracyAgent(mock_llm_with_errors)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert errors[0].category == "accuracy"
        assert errors[0].subcategory == "mistranslation"
        assert mock_llm_with_errors.call_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_filters_wrong_category(
        self, mock_llm_class: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test agent filters out errors from wrong category."""
        # Arrange
        response_with_fluency_error = """ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: minor
LOCATION: 0-5
DESCRIPTION: Grammar issue
ERROR_END"""
        mock_llm = mock_llm_class(response=response_with_fluency_error)
        agent = AccuracyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert len(errors) == 0  # Fluency error filtered out by Accuracy agent

    def test_category_property(self, mock_llm: Any) -> None:
        """Test agent reports correct category."""
        # Arrange
        agent = AccuracyAgent(mock_llm)

        # Act & Assert
        assert agent.category == "accuracy"


@pytest.mark.unit
class TestFluencyAgent:
    """Test FluencyAgent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_evaluate_no_errors(
        self, mock_llm: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test fluency evaluation with no errors."""
        # Arrange
        agent = FluencyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert isinstance(errors, list)
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_evaluate_fluency_errors(
        self, mock_llm_class: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test fluency agent detects fluency errors."""
        # Arrange
        response = """ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: minor
LOCATION: 5-10
DESCRIPTION: Grammar mistake
ERROR_END"""
        mock_llm = mock_llm_class(response=response)
        agent = FluencyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert len(errors) == 1
        assert errors[0].category == "fluency"
        assert errors[0].subcategory == "grammar"

    def test_category_property(self, mock_llm: Any) -> None:
        """Test agent reports correct category."""
        # Arrange
        agent = FluencyAgent(mock_llm)

        # Act & Assert
        assert agent.category == "fluency"


@pytest.mark.unit
class TestTerminologyAgent:
    """Test TerminologyAgent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_evaluate_no_errors(
        self, mock_llm: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test terminology evaluation with no errors."""
        # Arrange
        agent = TerminologyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert isinstance(errors, list)
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_evaluate_terminology_errors(
        self, mock_llm_class: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test terminology agent detects terminology errors."""
        # Arrange
        response = """ERROR_START
CATEGORY: terminology
SUBCATEGORY: inconsistency
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Inconsistent term translation
SUGGESTION: Use standard term
ERROR_END"""
        mock_llm = mock_llm_class(response=response)
        agent = TerminologyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert len(errors) == 1
        assert errors[0].category == "terminology"
        assert errors[0].subcategory == "inconsistency"
        assert errors[0].suggestion == "Use standard term"

    def test_category_property(self, mock_llm: Any) -> None:
        """Test agent reports correct category."""
        # Arrange
        agent = TerminologyAgent(mock_llm)

        # Act & Assert
        assert agent.category == "terminology"


@pytest.mark.unit
class TestAgentErrorHandling:
    """Test agent error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_response(
        self, mock_llm_class: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test agent handles invalid JSON gracefully."""
        # Arrange
        mock_llm = mock_llm_class(response="Not valid JSON")
        agent = AccuracyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert isinstance(errors, list)
        # Should return empty list or handle gracefully
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_missing_errors_key(
        self, mock_llm_class: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test agent handles missing 'errors' key."""
        # Arrange
        mock_llm = mock_llm_class(response='{"result": "no errors key"}')
        agent = AccuracyAgent(mock_llm)

        # Act
        errors = await agent.evaluate(sample_translation_task)

        # Assert
        assert isinstance(errors, list)
        assert len(errors) == 0
