# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
# SPDX-License-Identifier: Apache-2.0
"""Tests for iterative refinement agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask

if TYPE_CHECKING:
    pass


# ============================================================================
# Mock Classes
# ============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Improved translation text"]
        self.call_count = 0

    async def complete(self, prompt: str, **kwargs) -> str:
        """Return mock response."""
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


class MockOrchestrator:
    """Mock agent orchestrator for testing."""

    def __init__(self, reports: list[QAReport] | None = None):
        self.reports = reports or []
        self.call_count = 0

    async def evaluate(self, task: TranslationTask) -> QAReport:
        """Return mock QA report."""
        if self.call_count < len(self.reports):
            report = self.reports[self.call_count]
        else:
            # Default report with high score (converged)
            report = QAReport(
                mqm_score=98.0,
                errors=[],
                task=task,
                status="pass",
            )
        self.call_count += 1
        return report


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.unit
class TestIterativeRefinementInit:
    """Tests for IterativeRefinement initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        from kttc.agents.refinement import IterativeRefinement

        mock_llm = MockLLMProvider()
        refinement = IterativeRefinement(llm_provider=mock_llm)

        assert refinement.llm == mock_llm
        assert refinement.max_iterations == 3
        assert refinement.convergence_threshold == 95.0
        assert refinement.min_improvement == 1.0

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        from kttc.agents.refinement import IterativeRefinement

        mock_llm = MockLLMProvider()
        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=5,
            convergence_threshold=90.0,
            min_improvement=0.5,
        )

        assert refinement.max_iterations == 5
        assert refinement.convergence_threshold == 90.0
        assert refinement.min_improvement == 0.5


@pytest.mark.unit
class TestConvergenceCheck:
    """Tests for convergence checking logic."""

    def test_convergence_empty_reports(self) -> None:
        """Test convergence check with no reports."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())
        result = refinement._check_convergence([], 0)

        assert result["converged"] is False
        assert result["reason"] == ""

    def test_convergence_threshold_met(self) -> None:
        """Test convergence when threshold is met."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(
            llm_provider=MockLLMProvider(),
            convergence_threshold=95.0,
        )

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        reports = [QAReport(mqm_score=96.0, errors=[], task=task, status="pass")]
        result = refinement._check_convergence(reports, 0)

        assert result["converged"] is True
        assert "Threshold met" in result["reason"]
        assert "96.00" in result["reason"]

    def test_convergence_improvement_stagnation(self) -> None:
        """Test convergence when improvement stagnates."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(
            llm_provider=MockLLMProvider(),
            convergence_threshold=99.0,
            min_improvement=1.0,
        )

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        reports = [
            QAReport(mqm_score=85.0, errors=[], task=task, status="pass"),
            QAReport(mqm_score=85.3, errors=[], task=task, status="pass"),
        ]

        result = refinement._check_convergence(reports, 1)

        assert result["converged"] is True
        assert "stagnated" in result["reason"]

    def test_no_convergence_good_improvement(self) -> None:
        """Test no convergence when there's good improvement."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(
            llm_provider=MockLLMProvider(),
            convergence_threshold=99.0,
            min_improvement=1.0,
        )

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        reports = [
            QAReport(mqm_score=80.0, errors=[], task=task, status="pass"),
            QAReport(mqm_score=85.0, errors=[], task=task, status="pass"),
        ]

        result = refinement._check_convergence(reports, 1)

        assert result["converged"] is False


@pytest.mark.unit
class TestBuildRefinementPrompt:
    """Tests for refinement prompt building."""

    def test_build_prompt_basic(self) -> None:
        """Test basic prompt building."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                location=(0, 5),
                description="Word mistranslated",
                suggestion="correct word",
            )
        ]

        prompt = refinement._build_refinement_prompt(
            source="Hello world",
            translation="Привет мир",
            errors=errors,
            iteration=0,
        )

        assert "Hello world" in prompt
        assert "Привет мир" in prompt
        assert "MAJOR" in prompt
        assert "mistranslation" in prompt
        assert "Word mistranslated" in prompt
        assert "correct word" in prompt
        assert "ITERATION: 1" in prompt

    def test_build_prompt_with_location(self) -> None:
        """Test prompt with error location extraction."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())

        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 6),
                description="Grammar error",
            )
        ]

        prompt = refinement._build_refinement_prompt(
            source="Test source",
            translation="Привет мир",
            errors=errors,
            iteration=1,
        )

        assert "Привет" in prompt
        assert "CRITICAL" in prompt


@pytest.mark.unit
class TestExtractTranslation:
    """Tests for translation extraction from LLM response."""

    def test_extract_plain_text(self) -> None:
        """Test extraction from plain text."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())
        result = refinement._extract_translation("Это улучшенный перевод")

        assert result == "Это улучшенный перевод"

    def test_extract_from_markdown(self) -> None:
        """Test extraction from markdown code block."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())
        result = refinement._extract_translation("```\nИзвлечённый текст\n```")

        assert result == "Извлечённый текст"

    def test_extract_with_label(self) -> None:
        """Test extraction with 'Improved translation:' label."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())
        result = refinement._extract_translation("Improved translation: Текст перевода")

        assert result == "Текст перевода"

    def test_extract_with_translation_label(self) -> None:
        """Test extraction with 'Translation:' label."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())
        result = refinement._extract_translation("Translation: Просто перевод")

        assert result == "Просто перевод"


@pytest.mark.unit
class TestCreateResult:
    """Tests for result creation."""

    def test_create_result_with_reports(self) -> None:
        """Test result creation with QA reports."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        reports = [
            QAReport(mqm_score=80.0, errors=[], task=task, status="pass"),
            QAReport(mqm_score=90.0, errors=[], task=task, status="pass"),
        ]

        result = refinement._create_result(
            final_translation="Финальный перевод",
            qa_reports=reports,
            converged=True,
            convergence_reason="Threshold met",
        )

        assert result.final_translation == "Финальный перевод"
        assert result.iterations == 2
        assert result.initial_score == 80.0
        assert result.final_score == 90.0
        assert result.improvement == 10.0
        assert result.converged is True
        assert result.convergence_reason == "Threshold met"

    def test_create_result_empty_reports(self) -> None:
        """Test result creation with no reports."""
        from kttc.agents.refinement import IterativeRefinement

        refinement = IterativeRefinement(llm_provider=MockLLMProvider())

        result = refinement._create_result(
            final_translation="Текст",
            qa_reports=[],
            converged=False,
            convergence_reason="No reports",
        )

        assert result.initial_score == 0.0
        assert result.final_score == 0.0
        assert result.improvement == 0.0


@pytest.mark.unit
class TestApplyRefinement:
    """Tests for refinement application."""

    @pytest.mark.asyncio
    async def test_apply_refinement_critical_errors(self) -> None:
        """Test refinement prioritizes critical errors."""
        from kttc.agents.refinement import IterativeRefinement

        mock_llm = MockLLMProvider(["Исправленный перевод"])
        refinement = IterativeRefinement(llm_provider=mock_llm)

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="test",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 5),
                description="Critical error",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(6, 10),
                description="Minor error",
            ),
        ]

        result = await refinement.apply_refinement(
            source="Test source",
            translation="Тестовый перевод",
            errors=errors,
            iteration=0,
        )

        assert result == "Исправленный перевод"
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_apply_refinement_only_minor(self) -> None:
        """Test refinement with only minor errors."""
        from kttc.agents.refinement import IterativeRefinement

        mock_llm = MockLLMProvider(["Улучшенный текст"])
        refinement = IterativeRefinement(llm_provider=mock_llm)

        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="style",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Minor style issue",
            ),
        ]

        result = await refinement.apply_refinement(
            source="Source",
            translation="Перевод",
            errors=errors,
            iteration=0,
        )

        assert result == "Улучшенный текст"


@pytest.mark.unit
class TestRefineUntilConvergence:
    """Tests for full refinement loop."""

    @pytest.mark.asyncio
    async def test_refine_converges_immediately(self) -> None:
        """Test refinement when first evaluation meets threshold."""
        from kttc.agents.refinement import IterativeRefinement

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        reports = [QAReport(mqm_score=98.0, errors=[], task=task, status="pass")]

        mock_llm = MockLLMProvider()
        mock_orchestrator = MockOrchestrator(reports=reports)

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            convergence_threshold=95.0,
        )

        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        assert result.converged is True
        assert result.iterations == 1
        assert result.final_score == 98.0

    @pytest.mark.asyncio
    async def test_refine_no_errors(self) -> None:
        """Test refinement stops when no errors found."""
        from kttc.agents.refinement import IterativeRefinement

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        # Score below threshold but no errors
        reports = [QAReport(mqm_score=85.0, errors=[], task=task, status="pass")]

        mock_llm = MockLLMProvider()
        mock_orchestrator = MockOrchestrator(reports=reports)

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            convergence_threshold=99.0,
        )

        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        assert result.converged is True
        assert "No errors" in result.convergence_reason

    @pytest.mark.asyncio
    async def test_refine_max_iterations(self) -> None:
        """Test refinement stops at max iterations."""
        from kttc.agents.refinement import IterativeRefinement

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="test",
                severity=ErrorSeverity.MAJOR,
                location=(0, 5),
                description="Error",
            )
        ]

        # All reports have low score with errors
        reports = [
            QAReport(mqm_score=70.0, errors=errors, task=task, status="fail"),
            QAReport(mqm_score=75.0, errors=errors, task=task, status="fail"),
            QAReport(mqm_score=80.0, errors=errors, task=task, status="fail"),
        ]

        mock_llm = MockLLMProvider(["Перевод 1", "Перевод 2", "Перевод 3"])
        mock_orchestrator = MockOrchestrator(reports=reports)

        refinement = IterativeRefinement(
            llm_provider=mock_llm,
            max_iterations=3,
            convergence_threshold=99.0,
        )

        result = await refinement.refine_until_convergence(task, mock_orchestrator)

        assert result.converged is False
        assert "Max iterations" in result.convergence_reason
        assert result.iterations == 3


@pytest.mark.unit
class TestRefinementResultModel:
    """Tests for RefinementResult model."""

    def test_refinement_result_fields(self) -> None:
        """Test RefinementResult field validation."""
        from kttc.agents.refinement import RefinementResult

        result = RefinementResult(
            final_translation="Test",
            iterations=2,
            initial_score=80.0,
            final_score=95.0,
            improvement=15.0,
            converged=True,
            convergence_reason="Test reason",
        )

        assert result.final_translation == "Test"
        assert result.iterations == 2
        assert result.improvement == 15.0

    def test_refinement_result_defaults(self) -> None:
        """Test RefinementResult default values."""
        from kttc.agents.refinement import RefinementResult

        result = RefinementResult(
            final_translation="Test",
            iterations=1,
            initial_score=90.0,
            final_score=90.0,
            improvement=0.0,
        )

        assert result.converged is False
        assert result.convergence_reason == ""
        assert result.qa_reports == []
