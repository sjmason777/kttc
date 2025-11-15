"""Unit tests for refinement module (TEaR loop).

Tests iterative translation refinement with mocked agents.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.agents.refinement import IterativeRefinement, RefinementResult
from kttc.core.models import QAReport, TranslationTask


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
