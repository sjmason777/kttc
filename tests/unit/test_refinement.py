"""Tests for Iterative Refinement system."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask
from kttc.core.refinement import IterativeRefinement, RefinementResult


class TestRefinementResult:
    """Test RefinementResult model."""

    def test_create_result(self) -> None:
        """Test creating refinement result."""
        task1 = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="es",
        )
        task2 = TranslationTask(
            source_text="Test",
            translation="Test2",
            source_lang="en",
            target_lang="es",
        )

        reports = [
            QAReport(
                task=task1,
                mqm_score=85.0,
                errors=[],
                status="fail",
            ),
            QAReport(
                task=task2,
                mqm_score=95.0,
                errors=[],
                status="pass",
            ),
        ]

        result = RefinementResult(
            final_translation="Test2",
            iterations=2,
            initial_score=85.0,
            final_score=95.0,
            improvement=10.0,
            qa_reports=reports,
            converged=True,
            convergence_reason="Threshold met",
        )

        assert result.final_translation == "Test2"
        assert result.iterations == 2
        assert result.initial_score == 85.0
        assert result.final_score == 95.0
        assert result.improvement == 10.0
        assert result.converged is True
        assert result.convergence_reason == "Threshold met"


class TestIterativeRefinement:
    """Test IterativeRefinement functionality."""

    @pytest.fixture
    def mock_llm(self) -> Mock:
        """Create mock LLM provider."""
        llm = Mock()
        llm.complete = AsyncMock(return_value="Improved translation")
        return llm

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator."""
        orchestrator = Mock()
        return orchestrator

    @pytest.fixture
    def task(self) -> TranslationTask:
        """Create test task."""
        return TranslationTask(
            source_text="Hello, world!",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

    def test_initialization(self, mock_llm: Mock) -> None:
        """Test refinement initialization."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=3, convergence_threshold=95.0, min_improvement=1.0
        )

        assert refinement.llm == mock_llm
        assert refinement.max_iterations == 3
        assert refinement.convergence_threshold == 95.0
        assert refinement.min_improvement == 1.0

    async def test_refine_converges_on_threshold(
        self, mock_llm: Mock, mock_orchestrator: Mock, task: TranslationTask
    ) -> None:
        """Test refinement converges when threshold is met."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=3, convergence_threshold=95.0
        )

        # Mock QA report with high score
        qa_report = QAReport(
            task=task,
            mqm_score=96.0,
            errors=[],
            status="pass",
        )

        mock_orchestrator.evaluate = AsyncMock(return_value=qa_report)

        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        assert result.converged is True
        assert "Threshold met" in result.convergence_reason
        assert result.final_score == 96.0
        assert result.iterations == 1

    async def test_refine_stops_on_stagnation(
        self, mock_llm: Mock, mock_orchestrator: Mock, task: TranslationTask
    ) -> None:
        """Test refinement stops when improvement stagnates."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=5, convergence_threshold=95.0, min_improvement=1.0
        )

        # Mock QA reports with minimal improvement
        reports = [
            QAReport(
                task=task,
                mqm_score=90.0,
                errors=[
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="grammar",
                        severity=ErrorSeverity.MINOR,
                        description="Minor issue",
                        location=(0, 5),
                    )
                ],
                status="fail",
            ),
            QAReport(
                task=task,
                mqm_score=90.2,  # Only 0.2 improvement
                errors=[],
                status="fail",
            ),
        ]

        mock_orchestrator.evaluate = AsyncMock(side_effect=reports)

        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        assert result.converged is True
        assert "stagnated" in result.convergence_reason
        assert result.iterations == 2

    async def test_refine_reaches_max_iterations(
        self, mock_llm: Mock, mock_orchestrator: Mock, task: TranslationTask
    ) -> None:
        """Test refinement stops at max iterations."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=2, convergence_threshold=95.0, min_improvement=0.5
        )

        # Mock QA reports with different scores but not reaching threshold
        reports = [
            QAReport(
                task=task,
                mqm_score=80.0,
                errors=[
                    ErrorAnnotation(
                        category="accuracy",
                        subcategory="mistranslation",
                        severity=ErrorSeverity.MAJOR,
                        description="Wrong meaning",
                        location=(0, 5),
                    )
                ],
                status="fail",
            ),
            QAReport(
                task=task,
                mqm_score=85.0,  # Improved by 5.0 (above min_improvement)
                errors=[
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="grammar",
                        severity=ErrorSeverity.MINOR,
                        description="Minor issue",
                        location=(0, 5),
                    )
                ],
                status="fail",
            ),
        ]

        mock_orchestrator.evaluate = AsyncMock(side_effect=reports)

        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        assert result.converged is False
        assert "Max iterations" in result.convergence_reason
        assert result.iterations == 2

    async def test_apply_refinement_prioritizes_critical_errors(self, mock_llm: Mock) -> None:
        """Test refinement prioritizes critical and major errors."""
        refinement = IterativeRefinement(llm_provider=mock_llm, max_iterations=3)

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                description="Critical error",
                location=(0, 5),
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                description="Minor error",
                location=(6, 10),
            ),
        ]

        result = await refinement.apply_refinement(
            source="Hello",
            translation="Hola",
            errors=errors,
            iteration=0,
        )

        assert result == "Improved translation"
        mock_llm.complete.assert_called_once()

        # Check that prompt includes critical error
        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]
        assert "CRITICAL" in prompt
        assert "Critical error" in prompt

    async def test_extract_translation_handles_markdown(self, mock_llm: Mock) -> None:
        """Test extraction handles markdown code blocks."""
        refinement = IterativeRefinement(llm_provider=mock_llm, max_iterations=3)

        # Test markdown format
        assert refinement._extract_translation("```\nTest\n```") == "Test"

        # Test with label
        assert refinement._extract_translation("Translation: Test") == "Test"
        assert refinement._extract_translation("Improved translation: Test") == "Test"

        # Test plain text
        assert refinement._extract_translation("  Test  ") == "Test"

    def test_check_convergence_threshold_met(self, mock_llm: Mock) -> None:
        """Test convergence detection when threshold is met."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=3, convergence_threshold=95.0
        )

        test_task = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="es",
        )

        reports = [
            QAReport(
                task=test_task,
                mqm_score=96.0,
                errors=[],
                status="pass",
            )
        ]

        result = refinement._check_convergence(reports, 0)

        assert result["converged"] is True
        assert "Threshold met" in result["reason"]

    def test_check_convergence_empty_reports(self, mock_llm: Mock) -> None:
        """Test convergence check with empty reports list (edge case)."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=3, convergence_threshold=95.0
        )

        # Edge case: empty reports list
        result = refinement._check_convergence([], 0)

        assert result["converged"] is False
        assert result["reason"] == ""

    def test_check_convergence_stagnation(self, mock_llm: Mock) -> None:
        """Test convergence detection on improvement stagnation."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=3, convergence_threshold=95.0, min_improvement=1.0
        )

        test_task1 = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="es",
        )
        test_task2 = TranslationTask(
            source_text="Test",
            translation="Test2",
            source_lang="en",
            target_lang="es",
        )

        reports = [
            QAReport(
                task=test_task1,
                mqm_score=90.0,
                errors=[],
                status="fail",
            ),
            QAReport(
                task=test_task2,
                mqm_score=90.1,
                errors=[],
                status="fail",
            ),
        ]

        result = refinement._check_convergence(reports, 1)

        assert result["converged"] is True
        assert "stagnated" in result["reason"]

    def test_build_refinement_prompt(self, mock_llm: Mock) -> None:
        """Test refinement prompt building."""
        refinement = IterativeRefinement(llm_provider=mock_llm, max_iterations=3)

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                description="Wrong translation",
                suggestion="Correct translation",
                location=(0, 5),
            )
        ]

        prompt = refinement._build_refinement_prompt(
            source="Hello",
            translation="Hola",
            errors=errors,
            iteration=0,
        )

        assert "Hello" in prompt
        assert "Hola" in prompt
        assert "MAJOR" in prompt
        assert "Wrong translation" in prompt
        assert "Correct translation" in prompt
        assert "ITERATION: 1" in prompt

    def test_create_result(self, mock_llm: Mock) -> None:
        """Test result creation."""
        refinement = IterativeRefinement(llm_provider=mock_llm, max_iterations=3)

        test_task1 = TranslationTask(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="es",
        )
        test_task2 = TranslationTask(
            source_text="Test",
            translation="Test2",
            source_lang="en",
            target_lang="es",
        )

        reports = [
            QAReport(
                task=test_task1,
                mqm_score=85.0,
                errors=[],
                status="fail",
            ),
            QAReport(
                task=test_task2,
                mqm_score=95.0,
                errors=[],
                status="pass",
            ),
        ]

        result = refinement._create_result(
            final_translation="Test2",
            qa_reports=reports,
            converged=True,
            convergence_reason="Test reason",
        )

        assert result.final_translation == "Test2"
        assert result.iterations == 2
        assert result.initial_score == 85.0
        assert result.final_score == 95.0
        assert result.improvement == 10.0
        assert result.converged is True
        assert result.convergence_reason == "Test reason"

    async def test_refine_stops_on_no_errors(
        self, mock_llm: Mock, mock_orchestrator: Mock, task: TranslationTask
    ) -> None:
        """Test refinement stops when no errors are found."""
        refinement = IterativeRefinement(
            llm_provider=mock_llm, max_iterations=3, convergence_threshold=95.0
        )

        # Mock QA report with no errors
        qa_report = QAReport(
            task=task,
            mqm_score=85.0,  # Below threshold but no errors
            errors=[],
            status="fail",
        )

        mock_orchestrator.evaluate = AsyncMock(return_value=qa_report)

        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        assert result.converged is True
        assert "No errors detected" in result.convergence_reason
        assert result.iterations == 1
