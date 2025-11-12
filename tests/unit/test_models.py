"""Unit tests for core data models.

Tests Pydantic models for validation, properties, and serialization.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask


@pytest.mark.unit
class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_severity_values(self) -> None:
        """Test all severity levels exist."""
        assert ErrorSeverity.NEUTRAL.value == "neutral"
        assert ErrorSeverity.MINOR.value == "minor"
        assert ErrorSeverity.MAJOR.value == "major"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_penalty_values(self) -> None:
        """Test penalty values for each severity level."""
        assert ErrorSeverity.NEUTRAL.penalty_value == 0.0
        assert ErrorSeverity.MINOR.penalty_value == 1.0
        assert ErrorSeverity.MAJOR.penalty_value == 5.0
        assert ErrorSeverity.CRITICAL.penalty_value == 10.0


@pytest.mark.unit
class TestErrorAnnotation:
    """Tests for ErrorAnnotation model."""

    def test_create_valid_error(self) -> None:
        """Test creating valid error annotation."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Test error",
        )
        assert error.category == "accuracy"
        assert error.severity == ErrorSeverity.MAJOR
        assert error.location == (0, 5)
        assert error.suggestion is None

    def test_create_error_with_suggestion(self) -> None:
        """Test creating error with suggestion."""
        error = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(10, 15),
            description="Grammar issue",
            suggestion="Fix verb tense",
        )
        assert error.suggestion == "Fix verb tense"

    def test_error_serialization(self) -> None:
        """Test error can be serialized to dict."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Test",
        )
        data = error.model_dump()
        assert data["category"] == "accuracy"
        assert data["severity"] == "major"

    def test_error_from_dict(self) -> None:
        """Test error can be created from dict."""
        data: dict[str, Any] = {
            "category": "terminology",
            "subcategory": "inconsistency",
            "severity": "minor",
            "location": (5, 10),
            "description": "Test",
        }
        error = ErrorAnnotation(**data)
        assert error.category == "terminology"
        assert error.severity == ErrorSeverity.MINOR


@pytest.mark.unit
class TestTranslationTask:
    """Tests for TranslationTask model."""

    def test_create_valid_task(self) -> None:
        """Test creating valid translation task."""
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )
        assert task.source_text == "Hello"
        assert task.translation == "Hola"
        assert task.source_lang == "en"
        assert task.target_lang == "es"
        assert task.context is None

    def test_create_task_with_context(self) -> None:
        """Test creating task with context metadata."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
            context={"domain": "technical"},
        )
        assert task.context == {"domain": "technical"}

    def test_word_count_property(self) -> None:
        """Test word count calculation."""
        task = TranslationTask(
            source_text="Hello world this is a test",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )
        assert task.word_count == 6

    def test_word_count_single_word(self) -> None:
        """Test word count for single word."""
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )
        assert task.word_count == 1

    def test_empty_text_validation(self) -> None:
        """Test validation fails for empty text."""
        with pytest.raises(ValidationError) as exc_info:
            TranslationTask(
                source_text="",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            )
        assert "source_text" in str(exc_info.value)

    def test_invalid_language_code(self) -> None:
        """Test validation fails for invalid language codes."""
        with pytest.raises(ValidationError) as exc_info:
            TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="eng",  # Should be 2 chars
                target_lang="es",
            )
        assert "source_lang" in str(exc_info.value)

    def test_task_serialization(self) -> None:
        """Test task can be serialized."""
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )
        data = task.model_dump()
        assert data["source_text"] == "Hello"
        assert data["source_lang"] == "en"


@pytest.mark.unit
class TestQAReport:
    """Tests for QAReport model."""

    def test_create_valid_report(self, sample_translation_task: Any) -> None:
        """Test creating valid QA report."""
        report = QAReport(
            task=sample_translation_task,
            mqm_score=95.5,
            errors=[],
            status="pass",
        )
        assert report.mqm_score == 95.5
        assert report.status == "pass"
        assert len(report.errors) == 0

    def test_create_report_with_errors(
        self,
        sample_translation_task: Any,
        sample_error_annotations: Any,
    ) -> None:
        """Test creating report with errors."""
        report = QAReport(
            task=sample_translation_task,
            mqm_score=85.0,
            errors=sample_error_annotations,
            status="fail",
        )
        assert len(report.errors) == 3

    def test_error_count_property(self, sample_qa_report_fail: Any) -> None:
        """Test error count property."""
        assert sample_qa_report_fail.error_count == 3

    def test_critical_error_count(self, sample_qa_report_fail: Any) -> None:
        """Test critical error count."""
        assert sample_qa_report_fail.critical_error_count == 1

    def test_major_error_count(self, sample_qa_report_fail: Any) -> None:
        """Test major error count."""
        assert sample_qa_report_fail.major_error_count == 1

    def test_minor_error_count(self, sample_qa_report_fail: Any) -> None:
        """Test minor error count."""
        assert sample_qa_report_fail.minor_error_count == 1

    def test_no_errors_counts(self, sample_qa_report_pass: Any) -> None:
        """Test error counts when no errors."""
        assert sample_qa_report_pass.error_count == 0
        assert sample_qa_report_pass.critical_error_count == 0
        assert sample_qa_report_pass.major_error_count == 0
        assert sample_qa_report_pass.minor_error_count == 0

    def test_invalid_mqm_score(self, sample_translation_task: Any) -> None:
        """Test validation fails for invalid MQM score."""
        with pytest.raises(ValidationError) as exc_info:
            QAReport(
                task=sample_translation_task,
                mqm_score=105.0,  # Must be <= 100
                errors=[],
                status="pass",
            )
        assert "mqm_score" in str(exc_info.value)

    def test_invalid_status(self, sample_translation_task: Any) -> None:
        """Test validation fails for invalid status."""
        with pytest.raises(ValidationError) as exc_info:
            QAReport(
                task=sample_translation_task,
                mqm_score=95.0,
                errors=[],
                status="maybe",  # Must be 'pass' or 'fail'
            )
        assert "status" in str(exc_info.value)

    def test_report_serialization(self, sample_qa_report_pass: Any) -> None:
        """Test report can be serialized."""
        data = sample_qa_report_pass.model_dump()
        assert data["mqm_score"] == 98.5
        assert data["status"] == "pass"
        assert "task" in data
