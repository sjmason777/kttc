"""Unit tests for correction module.

Tests automatic error correction functionality with mocked LLM.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kttc.core.correction import AutoCorrector
from kttc.core.models import ErrorAnnotation, ErrorSeverity, TranslationTask


@pytest.mark.unit
class TestAutoCorrector:
    """Test AutoCorrector functionality."""

    @pytest.fixture
    def sample_task(self) -> TranslationTask:
        """Provide a sample translation task for testing."""
        return TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

    @pytest.fixture
    def sample_errors(self) -> list[ErrorAnnotation]:
        """Provide sample errors for testing."""
        return [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 4),
                description="Incorrect greeting",
                suggestion="Use 'Hola' instead",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MAJOR,
                location=(5, 10),
                description="Grammar issue",
                suggestion="Use proper grammar",
            ),
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistency",
                severity=ErrorSeverity.MINOR,
                location=(0, 10),
                description="Inconsistent term",
            ),
        ]

    def test_initialization(self) -> None:
        """Test AutoCorrector initialization."""
        # Arrange
        mock_llm = MagicMock()

        # Act
        corrector = AutoCorrector(llm_provider=mock_llm)

        # Assert
        assert corrector.llm_provider == mock_llm

    @pytest.mark.asyncio
    async def test_auto_correct_no_errors(self, sample_task: TranslationTask) -> None:
        """Test auto correction with no errors."""
        # Arrange
        mock_llm = AsyncMock()
        corrector = AutoCorrector(llm_provider=mock_llm)

        # Act
        result = await corrector.auto_correct(sample_task, [])

        # Assert
        assert result == sample_task.translation
        # LLM should not be called when there are no errors
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_correct_light_mode(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test auto correction in light mode (critical/major only)."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "¡Hola mundo!"
        corrector = AutoCorrector(llm_provider=mock_llm)

        # Act
        result = await corrector.auto_correct(sample_task, sample_errors, correction_level="light")

        # Assert
        assert result == "¡Hola mundo!"
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_correct_full_mode(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test auto correction in full mode (all errors)."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "¡Hola mundo!"
        corrector = AutoCorrector(llm_provider=mock_llm)

        # Act
        result = await corrector.auto_correct(sample_task, sample_errors, correction_level="full")

        # Assert
        assert result == "¡Hola mundo!"
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_correct_light_mode_no_critical_or_major(self) -> None:
        """Test auto correction in light mode with only minor errors."""
        # Arrange
        mock_llm = AsyncMock()
        corrector = AutoCorrector(llm_provider=mock_llm)

        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        minor_errors = [
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistency",
                severity=ErrorSeverity.MINOR,
                location=(0, 10),
                description="Minor inconsistency",
            )
        ]

        # Act
        result = await corrector.auto_correct(task, minor_errors, correction_level="light")

        # Assert
        assert result == task.translation  # Should return original since no critical/major errors
        # LLM should not be called when no errors need fixing
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_corrected_translation(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test generating corrected translation with LLM."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "¡Hola mundo!"
        corrector = AutoCorrector(llm_provider=mock_llm)

        # Act
        result = await corrector._generate_corrected_translation(
            sample_task, sample_errors[:2], 0.2, 2000
        )

        # Assert
        assert result == "¡Hola mundo!"
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_corrected_translation_with_markdown(self) -> None:
        """Test generating corrected translation with markdown response."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "```text\n¡Hola mundo!\n```"
        corrector = AutoCorrector(llm_provider=mock_llm)

        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 4),
                description="Incorrect greeting",
                suggestion="Use 'Hola' instead",
            )
        ]

        # Act
        result = await corrector._generate_corrected_translation(task, errors, 0.2, 2000)

        # Assert
        assert result == "¡Hola mundo!"

    @pytest.mark.asyncio
    async def test_generate_corrected_translation_with_exception(self) -> None:
        """Test generating corrected translation when LLM fails."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = Exception("LLM error")
        corrector = AutoCorrector(llm_provider=mock_llm)

        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 4),
                description="Incorrect greeting",
            )
        ]

        # Act
        result = await corrector._generate_corrected_translation(task, errors, 0.2, 2000)

        # Assert
        assert result == task.translation  # Should return original on error

    def test_get_correction_summary(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test getting correction summary."""
        # Arrange
        mock_llm = MagicMock()
        corrector = AutoCorrector(llm_provider=mock_llm)

        # Act
        summary = corrector.get_correction_summary(
            sample_task.translation, "¡Hola mundo!", sample_errors
        )

        # Assert
        assert summary["errors_fixed"] == 3
        assert summary["error_breakdown"]["critical"] == 1
        assert summary["error_breakdown"]["major"] == 1
        assert summary["error_breakdown"]["minor"] == 1
        assert "accuracy" in summary["categories"]
        assert "fluency" in summary["categories"]
        assert "terminology" in summary["categories"]
        assert summary["original_length"] == len(sample_task.translation)
        assert summary["corrected_length"] == len("¡Hola mundo!")


@pytest.mark.unit
class TestCorrectAndReevaluate:
    """Test correct_and_reevaluate method."""

    @pytest.fixture
    def sample_task(self) -> TranslationTask:
        """Provide a sample translation task."""
        return TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

    @pytest.fixture
    def sample_errors(self) -> list[ErrorAnnotation]:
        """Provide sample errors for testing."""
        return [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 4),
                description="Wrong translation",
            ),
        ]

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_success(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test correct_and_reevaluate stops when quality threshold reached."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "¡Hola mundo!"

        mock_orchestrator = AsyncMock()
        mock_report = MagicMock()
        mock_report.mqm_score = 98.0
        mock_report.status = "pass"
        mock_report.error_count = 0
        mock_report.errors = []
        mock_orchestrator.evaluate.return_value = mock_report

        corrector = AutoCorrector(llm_provider=mock_llm)

        final, reports = await corrector.correct_and_reevaluate(
            task=sample_task,
            errors=sample_errors,
            orchestrator=mock_orchestrator,
            correction_level="light",
            max_iterations=3,
        )

        assert final == "¡Hola mundo!"
        assert len(reports) == 1
        assert reports[0].status == "pass"

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_no_changes(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test correct_and_reevaluate stops when no changes made."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = sample_task.translation  # No change

        mock_orchestrator = AsyncMock()
        corrector = AutoCorrector(llm_provider=mock_llm)

        final, reports = await corrector.correct_and_reevaluate(
            task=sample_task,
            errors=sample_errors,
            orchestrator=mock_orchestrator,
            correction_level="light",
        )

        # Should return original since no changes
        assert final == sample_task.translation
        assert len(reports) == 0

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_stagnation(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test correct_and_reevaluate stops when improvement stagnates."""
        mock_llm = AsyncMock()
        # Returns different corrections each time
        mock_llm.complete.side_effect = ["¡Hola mundo!", "Hola mundo!", "Hello mundo!"]

        mock_orchestrator = AsyncMock()

        # First report - some improvement
        mock_report1 = MagicMock()
        mock_report1.mqm_score = 85.0
        mock_report1.status = "fail"
        mock_report1.error_count = 2
        mock_report1.errors = sample_errors

        # Second report - stagnated (less than 1 point improvement)
        mock_report2 = MagicMock()
        mock_report2.mqm_score = 85.5
        mock_report2.status = "fail"
        mock_report2.error_count = 2
        mock_report2.errors = sample_errors

        mock_orchestrator.evaluate.side_effect = [mock_report1, mock_report2]

        corrector = AutoCorrector(llm_provider=mock_llm)

        final, reports = await corrector.correct_and_reevaluate(
            task=sample_task,
            errors=sample_errors,
            orchestrator=mock_orchestrator,
            correction_level="full",
            max_iterations=5,
        )

        # Should stop after stagnation
        assert len(reports) == 2

    @pytest.mark.asyncio
    async def test_correct_and_reevaluate_max_iterations(
        self, sample_task: TranslationTask, sample_errors: list[ErrorAnnotation]
    ) -> None:
        """Test correct_and_reevaluate respects max_iterations."""
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            "Version 1",
            "Version 2",
            "Version 3",
        ]

        mock_orchestrator = AsyncMock()

        def make_report(score: float) -> MagicMock:
            report = MagicMock()
            report.mqm_score = score
            report.status = "fail"
            report.error_count = 1
            report.errors = sample_errors
            return report

        # Each iteration improves by 5 points to avoid stagnation
        mock_orchestrator.evaluate.side_effect = [
            make_report(70.0),
            make_report(75.0),
        ]

        corrector = AutoCorrector(llm_provider=mock_llm)

        final, reports = await corrector.correct_and_reevaluate(
            task=sample_task,
            errors=sample_errors,
            orchestrator=mock_orchestrator,
            correction_level="full",
            max_iterations=2,  # Limit to 2 iterations
        )

        # Should respect max_iterations
        assert len(reports) == 2
