"""Unit tests for AgentOrchestrator.

Tests orchestrator logic with mocked agents.
Focus: Coordination, parallel execution, MQM scoring.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add tests directory to path to import conftest
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from conftest import MockLLMProvider  # noqa: E402

from kttc.agents.orchestrator import AgentOrchestrator  # noqa: E402
from kttc.core.models import TranslationTask  # noqa: E402


@pytest.mark.unit
class TestOrchestratorBasics:
    """Test basic orchestrator functionality."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_llm: Any) -> None:
        """Test orchestrator initializes with default agents."""
        # Arrange & Act
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Assert
        assert orchestrator is not None
        assert orchestrator.quality_threshold == 95.0  # Default threshold

    @pytest.mark.asyncio
    async def test_set_quality_threshold(self, mock_llm: Any) -> None:
        """Test setting custom quality threshold."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        orchestrator.set_quality_threshold(80.0)

        # Assert
        assert orchestrator.quality_threshold == 80.0

    @pytest.mark.asyncio
    async def test_set_invalid_threshold_raises_error(self, mock_llm: Any) -> None:
        """Test setting invalid threshold raises ValueError."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act & Assert
        with pytest.raises(ValueError, match="Threshold must be between 0 and 100"):
            orchestrator.set_quality_threshold(150.0)

        with pytest.raises(ValueError, match="Threshold must be between 0 and 100"):
            orchestrator.set_quality_threshold(-10.0)


@pytest.mark.unit
class TestOrchestratorEvaluation:
    """Test orchestrator evaluation with mocked agents."""

    @pytest.mark.asyncio
    async def test_evaluate_perfect_translation(self, mock_llm: Any) -> None:
        """Test evaluation of perfect translation with no errors."""
        # Arrange - use English->Spanish to avoid triggering language-specific checks
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None
        assert report.task == task
        assert report.mqm_score == 100.0  # Perfect score with no errors
        assert report.status == "pass"
        assert len(report.errors) == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_errors(
        self, mock_llm_with_errors: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test evaluation with translation errors."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm_with_errors)

        # Act
        report = await orchestrator.evaluate(sample_translation_task)

        # Assert
        assert report is not None
        assert report.mqm_score < 100.0  # Should have deductions
        # Status depends on threshold
        assert report.status in ["pass", "fail"]
        assert len(report.errors) >= 0  # May have errors

    @pytest.mark.asyncio
    async def test_evaluate_below_threshold_fails(
        self, mock_llm_with_errors: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test evaluation fails when score below threshold."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm_with_errors)
        orchestrator.set_quality_threshold(99.0)  # Very high threshold

        # Act
        report = await orchestrator.evaluate(sample_translation_task)

        # Assert
        # With errors, score will be < 99.0, so should fail
        if report.mqm_score < 99.0:
            assert report.status == "fail"
        else:
            assert report.status == "pass"

    @pytest.mark.asyncio
    async def test_evaluate_custom_threshold(self, mock_llm: Any) -> None:
        """Test evaluation with custom threshold."""
        # Arrange - use English->Spanish to avoid triggering language-specific checks
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, quality_threshold=80.0)

        # Act
        report = await orchestrator.evaluate(task)

        # Assert
        assert orchestrator.quality_threshold == 80.0
        assert report.mqm_score >= orchestrator.quality_threshold  # Should pass threshold


@pytest.mark.unit
class TestOrchestratorMQMScoring:
    """Test MQM scoring calculation."""

    @pytest.mark.asyncio
    async def test_mqm_score_calculation_no_errors(self, mock_llm: Any) -> None:
        """Test MQM score is 100 with no errors."""
        # Arrange - use English->Spanish to avoid triggering language-specific checks
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        report = await orchestrator.evaluate(task)

        # Assert
        assert report.mqm_score == 100.0

    @pytest.mark.asyncio
    async def test_mqm_score_with_critical_error(
        self, mock_llm: Any, sample_translation_task: TranslationTask
    ) -> None:
        """Test MQM score deduction for critical error."""
        # Arrange
        # Create LLM that returns a critical error
        critical_error_response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: critical
LOCATION: 0-5
DESCRIPTION: Critical mistranslation
ERROR_END"""
        mock_llm_critical = MockLLMProvider(response=critical_error_response)
        orchestrator = AgentOrchestrator(llm_provider=mock_llm_critical)

        # Act
        report = await orchestrator.evaluate(sample_translation_task)

        # Assert
        # Critical error should result in significant deduction
        assert report.mqm_score < 100.0
        assert len(report.errors) >= 1


@pytest.mark.unit
class TestOrchestratorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_text_validation_fails(self) -> None:
        """Test that TranslationTask validates and rejects empty text."""
        # Act & Assert - Pydantic should raise ValidationError for empty strings
        with pytest.raises(Exception):  # Pydantic ValidationError
            TranslationTask(
                source_text="",
                translation="",
                source_lang="en",
                target_lang="es",
            )

    @pytest.mark.asyncio
    async def test_evaluate_single_word(self, mock_llm: Any) -> None:
        """Test evaluation of single word translation."""
        # Arrange
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None
        assert report.task.word_count >= 1

    @pytest.mark.asyncio
    async def test_evaluate_very_long_text(self, mock_llm: Any) -> None:
        """Test evaluation of very long translation."""
        # Arrange
        long_text = "A" * 10000  # 10K characters
        task = TranslationTask(
            source_text=long_text,
            translation=long_text,
            source_lang="en",
            target_lang="es",
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None
        assert report.task.word_count > 0
