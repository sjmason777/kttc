"""Unit tests for refinement module (TEaR loop).

Tests iterative translation refinement with mocked agents.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.agents.refinement import (
    ConvergenceCheck,
    IterativeRefinement,
    RefinementResult,
)
from kttc.core.models import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask


@pytest.mark.unit
class TestRefinementResult:
    """Test RefinementResult model."""

    def test_result_creation(self) -> None:
        """Test creating refinement result."""
        # Act
        result = RefinementResult(
            final_translation="¡Hola!",
            iterations=3,
            initial_score=90.0,
            final_score=97.5,
            improvement=7.5,
        )

        # Assert
        assert result.final_translation == "¡Hola!"
        assert result.iterations == 3
        assert result.initial_score == 90.0
        assert result.final_score == 97.5
        assert result.improvement == 7.5

    def test_result_with_qa_reports(self) -> None:
        """Test creating refinement result with QA reports."""
        # Arrange
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        report = QAReport(
            task=task,
            mqm_score=95.0,
            errors=[],
            status="pass",
        )

        # Act
        result = RefinementResult(
            final_translation="¡Hola!",
            iterations=1,
            initial_score=90.0,
            final_score=95.0,
            improvement=5.0,
            qa_reports=[report],
            converged=True,
            convergence_reason="Threshold met",
        )

        # Assert
        assert len(result.qa_reports) == 1
        assert result.qa_reports[0].mqm_score == 95.0
        assert result.converged is True
        assert result.convergence_reason == "Threshold met"


@pytest.mark.unit
class TestConvergenceCheck:
    """Test ConvergenceCheck type."""

    def test_convergence_check_structure(self) -> None:
        """Test that ConvergenceCheck has the expected structure."""
        # Act
        check: ConvergenceCheck = {"converged": True, "reason": "Test reason"}

        # Assert
        assert check["converged"] is True
        assert check["reason"] == "Test reason"


@pytest.mark.unit
class TestIterativeRefinement:
    """Test IterativeRefinement with mocked components."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        """Test refinement system initialization."""
        # Arrange
        mock_llm = MagicMock()

        # Act
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
            min_improvement=1.0,
        )

        # Assert
        assert refinement.llm == mock_llm
        assert refinement.max_iterations == 3
        assert refinement.convergence_threshold == 95.0
        assert refinement.min_improvement == 1.0

    @pytest.mark.asyncio
    async def test_refine_stops_at_max_iterations(self) -> None:
        """Test that refinement stops at max iterations."""
        # Arrange
        mock_llm = MagicMock()
        mock_orchestrator = MagicMock()

        # Create mock task for QAReport
        task = TranslationTask(
            source_text="Test",
            translation="Prueba inicial",
            source_lang="en",
            target_lang="es",
        )

        # Mock evaluate to return low quality scores
        mock_orchestrator.evaluate = AsyncMock(
            return_value=QAReport(
                task=task,
                mqm_score=80.0,  # Below threshold
                errors=[],
                status="fail",
            )
        )

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=2,
            convergence_threshold=95.0,
        )

        # Act
        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        # Assert
        # Should stop after max_iterations even if quality not reached
        assert result.iterations <= 2

    @pytest.mark.asyncio
    async def test_refine_stops_when_quality_reached(self) -> None:
        """Test that refinement stops when quality threshold is reached."""
        # Arrange
        mock_llm = MagicMock()
        mock_orchestrator = MagicMock()

        # Create mock task for QAReport
        task = TranslationTask(
            source_text="Test",
            translation="Excellent translation",
            source_lang="en",
            target_lang="es",
        )

        # Mock evaluate to return high quality score
        mock_orchestrator.evaluate = AsyncMock(
            return_value=QAReport(
                task=task,
                mqm_score=98.0,  # Above threshold
                errors=[],
                status="pass",
            )
        )

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=5,
            convergence_threshold=95.0,
        )

        # Act
        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        # Assert
        # Should stop early when quality is good
        assert result.final_score >= 95.0
        assert result.iterations <= 5

    @pytest.mark.asyncio
    async def test_refine_stops_when_no_errors(self) -> None:
        """Test that refinement stops when no errors are found."""
        # Arrange
        mock_llm = MagicMock()
        mock_orchestrator = MagicMock()

        # Create mock task for QAReport
        task = TranslationTask(
            source_text="Test",
            translation="Perfect translation",
            source_lang="en",
            target_lang="es",
        )

        # Mock evaluate to return no errors
        mock_orchestrator.evaluate = AsyncMock(
            return_value=QAReport(
                task=task,
                mqm_score=90.0,
                errors=[],  # No errors
                status="pass",
            )
        )

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=5,
            convergence_threshold=95.0,
        )

        # Act
        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        # Assert
        assert result.converged is True
        assert result.convergence_reason == "No errors detected"
        assert result.iterations == 1  # Should stop after first iteration

    @pytest.mark.asyncio
    async def test_refine_applies_corrections(self) -> None:
        """Test that refinement applies corrections when errors are found."""
        # Arrange
        mock_llm = MagicMock()
        mock_orchestrator = MagicMock()

        # Create mock task for QAReport
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        # Create error annotation
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 4),
            description="Incorrect greeting",
            suggestion="Use 'Hola' instead",
        )

        # Mock evaluate to return errors on first call, then no errors
        mock_orchestrator.evaluate = AsyncMock()
        mock_orchestrator.evaluate.side_effect = [
            QAReport(
                task=task,
                mqm_score=80.0,
                errors=[error],
                status="fail",
            ),
            QAReport(
                task=task,
                mqm_score=95.0,
                errors=[],  # No errors after correction
                status="pass",
            ),
        ]

        # Mock LLM to return improved translation
        mock_llm.complete = AsyncMock(return_value="¡Hola mundo!")

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=5,
            convergence_threshold=90.0,
        )

        # Act
        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        # Assert
        assert result.iterations == 2
        assert result.final_score >= 90.0
        assert mock_llm.complete.call_count == 1  # Should call LLM to get correction

    def test_check_convergence_no_reports(self) -> None:
        """Test convergence check with no reports."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        # Act
        result = refinement._check_convergence([], 0)

        # Assert
        assert result["converged"] is False
        assert result["reason"] == ""

    def test_check_convergence_threshold_met(self) -> None:
        """Test convergence when threshold is met."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        task = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="es",
        )

        reports = [
            QAReport(
                task=task,
                mqm_score=98.0,  # Above threshold
                errors=[],
                status="pass",
            )
        ]

        # Act
        result = refinement._check_convergence(reports, 0)

        # Assert
        assert result["converged"] is True
        assert "98.00 >= 95.0" in result["reason"]

    def test_check_convergence_improvement_stagnation(self) -> None:
        """Test convergence when improvement stagnates."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
            min_improvement=2.0,
        )

        task = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="es",
        )

        reports = [
            QAReport(
                task=task,
                mqm_score=90.0,
                errors=[],
                status="pass",
            ),
            QAReport(
                task=task,
                mqm_score=91.0,  # Only 1 point improvement
                errors=[],
                status="pass",
            ),
        ]

        # Act
        result = refinement._check_convergence(reports, 1)

        # Assert
        assert result["converged"] is True
        assert "Improvement stagnated" in result["reason"]
        assert "+1.00 < 2.0" in result["reason"]

    def test_build_refinement_prompt(self) -> None:
        """Test building refinement prompt."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Incorrect translation",
            suggestion="Use correct translation",
        )

        # Act
        prompt = refinement._build_refinement_prompt(
            source="Hello world",
            translation="Hola mundo",
            errors=[error],
            iteration=0,
        )

        # Assert
        assert "Hello world" in prompt
        assert "Hola mundo" in prompt
        assert "accuracy/mistranslation" in prompt
        assert "Incorrect translation" in prompt
        assert "Use correct translation" in prompt

    def test_extract_translation_plain_text(self) -> None:
        """Test extracting translation from plain text response."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        response = "¡Hola mundo!"

        # Act
        extracted = refinement._extract_translation(response)

        # Assert
        assert extracted == "¡Hola mundo!"

    def test_extract_translation_with_markdown(self) -> None:
        """Test extracting translation from markdown response."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        response = "```text\n¡Hola mundo!\n```"

        # Act
        extracted = refinement._extract_translation(response)

        # Assert
        assert extracted == "¡Hola mundo!"

    def test_extract_translation_with_label(self) -> None:
        """Test extracting translation from response with label."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        response = "Improved translation: ¡Hola mundo!"

        # Act
        extracted = refinement._extract_translation(response)

        # Assert
        assert extracted == "¡Hola mundo!"

    def test_create_result_empty_reports(self) -> None:
        """Test creating result with empty reports."""
        # Arrange
        mock_llm = MagicMock()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        # Act
        result = refinement._create_result(
            final_translation="¡Hola!",
            qa_reports=[],
            converged=True,
            convergence_reason="Test reason",
        )

        # Assert
        assert result.final_translation == "¡Hola!"
        assert result.initial_score == 0.0
        assert result.final_score == 0.0
        assert result.improvement == 0.0
        assert result.converged is True

    def test_apply_refinement_with_errors(self) -> None:
        """Test applying refinement with errors."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="¡Hola mundo!")

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=95.0,
        )

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Incorrect translation",
            suggestion="Use correct translation",
        )

        # Act
        result = refinement._build_refinement_prompt(
            source="Hello world",
            translation="Hola mundo",
            errors=[error],
            iteration=0,
        )

        # Assert
        assert "Hello world" in result
        assert "Hola mundo" in result
        assert "accuracy/mistranslation" in result
        assert mock_llm.complete.call_count == 0  # This is just building the prompt
