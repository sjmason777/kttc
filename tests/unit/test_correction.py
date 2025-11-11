"""Tests for AutoCorrector (automatic post-editing).

Comprehensive tests for error correction functionality.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask
from kttc.core.correction import AutoCorrector


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    llm = Mock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def corrector(mock_llm):
    """Create AutoCorrector instance."""
    return AutoCorrector(mock_llm)


@pytest.fixture
def task():
    """Create sample translation task."""
    return TranslationTask(
        source_text="Hello world",
        translation="Hola mundo incorrecto",
        source_lang="en",
        target_lang="es",
    )


@pytest.fixture
def critical_error():
    """Create critical error."""
    return ErrorAnnotation(
        category="accuracy",
        subcategory="mistranslation",
        severity=ErrorSeverity.CRITICAL,
        description="Wrong word used",
        suggestion="Use 'correcto' instead",
        location=(12, 22),
    )


@pytest.fixture
def major_error():
    """Create major error."""
    return ErrorAnnotation(
        category="fluency",
        subcategory="grammar",
        severity=ErrorSeverity.MAJOR,
        description="Grammar issue",
        suggestion="Fix grammar",
        location=(6, 11),
    )


@pytest.fixture
def minor_error():
    """Create minor error."""
    return ErrorAnnotation(
        category="style",
        subcategory="punctuation",
        severity=ErrorSeverity.MINOR,
        description="Missing comma",
        suggestion="Add comma",
        location=(5, 6),
    )


class TestAutoCorrect:
    """Test auto_correct method."""

    @pytest.mark.asyncio
    async def test_auto_correct_no_errors(self, corrector, task, mock_llm):
        """Test correction with no errors returns original."""
        result = await corrector.auto_correct(task, errors=[])

        assert result == task.translation
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_correct_light_mode(
        self, corrector, task, mock_llm, critical_error, major_error, minor_error
    ):
        """Test light mode corrects only critical and major errors."""
        mock_llm.complete.return_value = "Hola mundo correcto"

        errors = [critical_error, major_error, minor_error]
        result = await corrector.auto_correct(task, errors, correction_level="light")

        assert result == "Hola mundo correcto"
        mock_llm.complete.assert_called_once()

        # Verify prompt contains only critical and major errors
        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]
        assert "CRITICAL" in prompt
        assert "MAJOR" in prompt
        # Minor error should be filtered out in light mode

    @pytest.mark.asyncio
    async def test_auto_correct_full_mode(
        self, corrector, task, mock_llm, critical_error, major_error, minor_error
    ):
        """Test full mode corrects all errors."""
        mock_llm.complete.return_value = "Hola mundo correcto"

        errors = [critical_error, major_error, minor_error]
        result = await corrector.auto_correct(task, errors, correction_level="full")

        assert result == "Hola mundo correcto"
        mock_llm.complete.assert_called_once()

        # Verify prompt contains all errors
        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]
        assert "CRITICAL" in prompt
        assert "MAJOR" in prompt
        assert "MINOR" in prompt

    @pytest.mark.asyncio
    async def test_auto_correct_light_mode_only_minor_errors(
        self, corrector, task, mock_llm, minor_error
    ):
        """Test light mode with only minor errors returns original."""
        result = await corrector.auto_correct(task, errors=[minor_error], correction_level="light")

        assert result == task.translation
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_correct_removes_markdown(self, corrector, task, mock_llm, critical_error):
        """Test that markdown code blocks are removed from LLM response."""
        mock_llm.complete.return_value = "```\nHola mundo correcto\n```"

        result = await corrector.auto_correct(task, errors=[critical_error])

        assert result == "Hola mundo correcto"
        assert "```" not in result

    @pytest.mark.asyncio
    async def test_auto_correct_error_handling(self, corrector, task, critical_error):
        """Test error handling when LLM fails."""
        corrector.llm_provider.complete.side_effect = Exception("LLM error")

        result = await corrector.auto_correct(task, errors=[critical_error])

        # Should return original translation on error
        assert result == task.translation

    @pytest.mark.asyncio
    async def test_auto_correct_with_suggestion(self, corrector, task, mock_llm, critical_error):
        """Test correction uses error suggestion."""
        mock_llm.complete.return_value = "Hola mundo correcto"

        await corrector.auto_correct(task, errors=[critical_error])

        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]
        assert critical_error.suggestion in prompt


class TestCorrectAndReevaluate:
    """Test correct_and_reevaluate method."""

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_converges(
        self, corrector, task, mock_llm, critical_error
    ):
        """Test iterative correction converges on quality."""
        mock_llm.complete.return_value = "Hola mundo correcto"

        mock_orchestrator = Mock()
        mock_orchestrator.evaluate = AsyncMock()
        mock_orchestrator.evaluate.return_value = QAReport(
            task=task,
            mqm_score=95.0,
            errors=[],
            error_count=0,
            status="pass",
        )

        final, reports = await corrector.correct_and_reevaluate(
            task, [critical_error], mock_orchestrator
        )

        assert final == "Hola mundo correcto"
        assert len(reports) == 1
        assert reports[0].status == "pass"

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_max_iterations(
        self, corrector, task, mock_llm, critical_error
    ):
        """Test reaches max iterations without converging."""
        mock_llm.complete.return_value = "Hola mundo parcialmente corregido"

        mock_orchestrator = Mock()
        mock_orchestrator.evaluate = AsyncMock()
        mock_orchestrator.evaluate.return_value = QAReport(
            task=task,
            mqm_score=70.0,
            errors=[critical_error],
            error_count=1,
            status="fail",
        )

        final, reports = await corrector.correct_and_reevaluate(
            task, [critical_error], mock_orchestrator, max_iterations=2
        )

        assert len(reports) <= 2
        assert reports[-1].mqm_score == 70.0

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_no_change(self, corrector, task, critical_error):
        """Test stops when no changes made."""
        corrector.llm_provider.complete.return_value = task.translation

        mock_orchestrator = Mock()
        mock_orchestrator.evaluate = AsyncMock()

        final, reports = await corrector.correct_and_reevaluate(
            task, [critical_error], mock_orchestrator
        )

        # Should stop after first iteration if no change
        assert final == task.translation
        assert len(reports) == 0

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_stagnation(
        self, corrector, task, mock_llm, critical_error
    ):
        """Test stops when improvement stagnates."""
        mock_llm.complete.side_effect = ["Corrected v1", "Corrected v2"]

        mock_orchestrator = Mock()
        mock_orchestrator.evaluate = AsyncMock()
        # First report: 70.0, Second report: 70.5 (< 1.0 improvement)
        mock_orchestrator.evaluate.side_effect = [
            QAReport(
                task=task,
                mqm_score=70.0,
                errors=[critical_error],
                error_count=1,
                status="fail",
            ),
            QAReport(
                task=task,
                mqm_score=70.5,
                errors=[critical_error],
                error_count=1,
                status="fail",
            ),
        ]

        final, reports = await corrector.correct_and_reevaluate(
            task, [critical_error], mock_orchestrator, max_iterations=3
        )

        # Should stop due to stagnation after 2 iterations
        assert len(reports) == 2
        assert abs(reports[1].mqm_score - reports[0].mqm_score) < 1.0


class TestGetCorrectionSummary:
    """Test get_correction_summary method."""

    def test_get_correction_summary(self, corrector, critical_error, major_error, minor_error):
        """Test correction summary generation."""
        original = "Original translation text"
        corrected = "Corrected translation text with more words"
        errors = [critical_error, major_error, minor_error]

        summary = corrector.get_correction_summary(original, corrected, errors)

        assert summary["errors_fixed"] == 3
        assert summary["error_breakdown"]["critical"] == 1
        assert summary["error_breakdown"]["major"] == 1
        assert summary["error_breakdown"]["minor"] == 1
        assert "accuracy" in summary["categories"]
        assert "fluency" in summary["categories"]
        assert "style" in summary["categories"]
        assert summary["original_length"] == len(original)
        assert summary["corrected_length"] == len(corrected)
        assert summary["length_change"] == len(corrected) - len(original)

    def test_get_correction_summary_no_errors(self, corrector):
        """Test summary with no errors."""
        original = "Text"
        corrected = "Text"

        summary = corrector.get_correction_summary(original, corrected, [])

        assert summary["errors_fixed"] == 0
        assert summary["error_breakdown"]["critical"] == 0
        assert summary["error_breakdown"]["major"] == 0
        assert summary["error_breakdown"]["minor"] == 0
        assert summary["categories"] == []
        assert summary["length_change"] == 0


class TestGenerateCorrectedTranslation:
    """Test _generate_corrected_translation private method."""

    @pytest.mark.asyncio
    async def test_generate_includes_all_error_info(
        self, corrector, task, mock_llm, critical_error
    ):
        """Test that generated prompt includes all error information."""
        mock_llm.complete.return_value = "Corrected"

        await corrector._generate_corrected_translation(task, [critical_error], 0.2, 2000)

        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]

        # Verify all error components are in prompt
        assert str(critical_error.location[0]) in prompt
        assert str(critical_error.location[1]) in prompt
        assert critical_error.description in prompt
        assert critical_error.suggestion in prompt
        assert critical_error.severity.value.upper() in prompt
        assert critical_error.category in prompt
        assert critical_error.subcategory in prompt

    @pytest.mark.asyncio
    async def test_generate_without_suggestion(self, corrector, task, mock_llm):
        """Test generation when error has no suggestion."""
        error_no_suggestion = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            description="Wrong word",
            suggestion=None,
            location=(0, 5),
        )
        mock_llm.complete.return_value = "Corrected"

        result = await corrector._generate_corrected_translation(
            task, [error_no_suggestion], 0.2, 2000
        )

        assert result == "Corrected"
        # Should not crash without suggestion

    @pytest.mark.asyncio
    async def test_generate_fallback_on_error(self, corrector, task, critical_error):
        """Test fallback to original on LLM error."""
        corrector.llm_provider.complete.side_effect = Exception("LLM failed")

        result = await corrector._generate_corrected_translation(task, [critical_error], 0.2, 2000)

        assert result == task.translation
