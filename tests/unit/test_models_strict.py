"""Strict validation tests for data models.

These tests focus on edge cases, boundary conditions, type validation,
and ensuring models behave correctly under unusual inputs.
"""

import pytest
from pydantic import ValidationError

from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask


@pytest.mark.unit
class TestStrictModelValidation:
    """Strict validation tests for all models."""

    # ==================== TranslationTask Strict Tests ====================

    def test_translation_task_empty_strings_rejected(self) -> None:
        """Test that empty strings are rejected."""
        with pytest.raises(ValidationError):
            TranslationTask(
                source_text="",
                translation="valid",
                source_lang="en",
                target_lang="es",
            )

        with pytest.raises(ValidationError):
            TranslationTask(
                source_text="valid",
                translation="",
                source_lang="en",
                target_lang="es",
            )

    def test_translation_task_whitespace_only_allowed(self) -> None:
        """Test that whitespace-only strings are allowed (Pydantic min_length checks bytes, not stripped)."""
        # This is actually valid - Pydantic's min_length doesn't strip
        task = TranslationTask(
            source_text="   ",
            translation="valid",
            source_lang="en",
            target_lang="es",
        )
        # Whitespace is preserved
        assert len(task.source_text) == 3
        # But word count will be 0 (after split)
        assert task.word_count == 0

    def test_translation_task_invalid_lang_codes(self) -> None:
        """Test various invalid language code formats."""
        # Too short
        with pytest.raises(ValidationError):
            TranslationTask(
                source_text="test",
                translation="test",
                source_lang="e",
                target_lang="es",
            )

        # Too long
        with pytest.raises(ValidationError):
            TranslationTask(
                source_text="test",
                translation="test",
                source_lang="eng",
                target_lang="es",
            )

        # Numbers
        with pytest.raises(ValidationError):
            TranslationTask(
                source_text="test",
                translation="test",
                source_lang="e1",
                target_lang="es",
            )

        # Uppercase (should fail due to pattern)
        with pytest.raises(ValidationError):
            TranslationTask(
                source_text="test",
                translation="test",
                source_lang="EN",
                target_lang="es",
            )

        # Special characters
        with pytest.raises(ValidationError):
            TranslationTask(
                source_text="test",
                translation="test",
                source_lang="e-",
                target_lang="es",
            )

    def test_translation_task_word_count_edge_cases(self) -> None:
        """Test word count with various edge cases."""
        # Single word
        task = TranslationTask(
            source_text="word",
            translation="palabra",
            source_lang="en",
            target_lang="es",
        )
        assert task.word_count == 1

        # Multiple spaces
        task = TranslationTask(
            source_text="word1    word2     word3",
            translation="test",
            source_lang="en",
            target_lang="es",
        )
        # Python's split() handles multiple spaces correctly
        assert task.word_count == 3

        # Leading/trailing spaces
        task = TranslationTask(
            source_text="  word1 word2  ",
            translation="test",
            source_lang="en",
            target_lang="es",
        )
        assert task.word_count == 2

        # Newlines and tabs
        task = TranslationTask(
            source_text="word1\nword2\tword3",
            translation="test",
            source_lang="en",
            target_lang="es",
        )
        assert task.word_count == 3

    def test_translation_task_very_long_text(self) -> None:
        """Test with very long text."""
        long_text = "word " * 10000  # 10,000 words
        task = TranslationTask(
            source_text=long_text,
            translation="test",
            source_lang="en",
            target_lang="es",
        )
        assert task.word_count == 10000

    def test_translation_task_special_characters(self) -> None:
        """Test with special characters in text."""
        task = TranslationTask(
            source_text="Hello! How are you? 你好",
            translation="¡Hola! ¿Cómo estás?",
            source_lang="en",
            target_lang="es",
        )
        assert task.word_count > 0  # Should count words correctly

    def test_translation_task_context_types(self) -> None:
        """Test various context dictionary formats."""
        # Nested dict
        task = TranslationTask(
            source_text="test",
            translation="test",
            source_lang="en",
            target_lang="es",
            context={
                "domain": "medical",
                "metadata": {"source": "document.pdf", "page": 5},
                "glossary": ["term1", "term2"],
            },
        )
        assert task.context is not None
        assert isinstance(task.context["metadata"], dict)

        # Empty context
        task = TranslationTask(
            source_text="test",
            translation="test",
            source_lang="en",
            target_lang="es",
            context={},
        )
        assert task.context == {}

    # ==================== ErrorAnnotation Strict Tests ====================

    def test_error_annotation_location_validation(self) -> None:
        """Test location tuple validation."""
        # Valid location
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Error",
        )
        assert error.location == (0, 10)

        # Negative indices (should be allowed - might be used for positions from end)
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(-1, 5),
            description="Error",
        )
        assert error.location[0] == -1

        # Same start and end
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(5, 5),
            description="Error",
        )
        assert error.location[0] == error.location[1]

    def test_error_annotation_category_case_sensitivity(self) -> None:
        """Test that category names are case-sensitive."""
        # Lowercase
        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test",
        )
        assert error1.category == "accuracy"

        # Uppercase (should work - no case restriction)
        error2 = ErrorAnnotation(
            category="ACCURACY",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test",
        )
        assert error2.category == "ACCURACY"

    def test_error_annotation_long_description(self) -> None:
        """Test with very long description."""
        long_desc = "A" * 10000  # 10,000 characters
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description=long_desc,
        )
        assert len(error.description) == 10000

    def test_error_annotation_special_chars_in_strings(self) -> None:
        """Test special characters in string fields."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test\n\t\r",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Error with 'quotes' and \"double quotes\" and unicode: 你好",
            suggestion="Fix with <tags> & special=chars",
        )
        assert "\n" in error.subcategory
        assert "你好" in error.description
        assert error.suggestion is not None and "&" in error.suggestion

    def test_error_annotation_none_suggestion(self) -> None:
        """Test that None suggestion is handled correctly."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test",
            suggestion=None,
        )
        assert error.suggestion is None

        # Omitting suggestion (should default to None)
        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test",
        )
        assert error2.suggestion is None

    # ==================== QAReport Strict Tests ====================

    def test_qa_report_mqm_score_boundaries(self) -> None:
        """Test MQM score at exact boundaries."""
        task = TranslationTask(
            source_text="test",
            translation="test",
            source_lang="en",
            target_lang="es",
        )

        # Exactly 0
        report = QAReport(task=task, mqm_score=0.0, errors=[], status="fail")
        assert report.mqm_score == 0.0

        # Exactly 100
        report = QAReport(task=task, mqm_score=100.0, errors=[], status="pass")
        assert report.mqm_score == 100.0

        # Just below 100
        report = QAReport(task=task, mqm_score=99.99, errors=[], status="pass")
        assert report.mqm_score == 99.99

        # Just above 0
        report = QAReport(task=task, mqm_score=0.01, errors=[], status="fail")
        assert report.mqm_score == 0.01

    def test_qa_report_invalid_mqm_scores(self) -> None:
        """Test that invalid MQM scores are rejected."""
        task = TranslationTask(
            source_text="test",
            translation="test",
            source_lang="en",
            target_lang="es",
        )

        # Above 100
        with pytest.raises(ValidationError):
            QAReport(task=task, mqm_score=100.01, errors=[], status="pass")

        # Below 0
        with pytest.raises(ValidationError):
            QAReport(task=task, mqm_score=-0.01, errors=[], status="fail")

        # Way above
        with pytest.raises(ValidationError):
            QAReport(task=task, mqm_score=1000.0, errors=[], status="pass")

    def test_qa_report_invalid_status_values(self) -> None:
        """Test that only 'pass' or 'fail' are accepted for status."""
        task = TranslationTask(
            source_text="test",
            translation="test",
            source_lang="en",
            target_lang="es",
        )

        # Invalid statuses
        invalid_statuses = ["Pass", "PASS", "Fail", "FAIL", "pending", "unknown", ""]

        for status in invalid_statuses:
            with pytest.raises(ValidationError):
                QAReport(task=task, mqm_score=95.0, errors=[], status=status)

    def test_qa_report_error_count_properties(self) -> None:
        """Test error count properties are accurate."""
        task = TranslationTask(
            source_text="test",
            translation="test",
            source_lang="en",
            target_lang="es",
        )

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="test",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 5),
                description="Critical 1",
            ),
            ErrorAnnotation(
                category="accuracy",
                subcategory="test",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 5),
                description="Critical 2",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MAJOR,
                location=(0, 5),
                description="Major 1",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MAJOR,
                location=(0, 5),
                description="Major 2",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MAJOR,
                location=(0, 5),
                description="Major 3",
            ),
            ErrorAnnotation(
                category="style",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 5),
                description="Minor 1",
            ),
            ErrorAnnotation(
                category="style",
                subcategory="test",
                severity=ErrorSeverity.NEUTRAL,
                location=(0, 5),
                description="Neutral 1",
            ),
        ]

        report = QAReport(task=task, mqm_score=50.0, errors=errors, status="fail")

        # Test all counts are exact
        assert report.error_count == 7
        assert report.critical_error_count == 2
        assert report.major_error_count == 3
        assert report.minor_error_count == 1
        # Neutral errors should not be counted in minor
        assert (
            sum([report.critical_error_count, report.major_error_count, report.minor_error_count])
            == 6
        )

    def test_qa_report_large_error_list(self) -> None:
        """Test with a large number of errors."""
        task = TranslationTask(
            source_text="test",
            translation="test",
            source_lang="en",
            target_lang="es",
        )

        # Create 1000 errors
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(i, i + 5),
                description=f"Error {i}",
            )
            for i in range(1000)
        ]

        report = QAReport(task=task, mqm_score=0.0, errors=errors, status="fail")
        assert report.error_count == 1000
        assert report.minor_error_count == 1000

    def test_qa_report_serialization_deserialization(self) -> None:
        """Test that reports can be serialized and deserialized correctly."""
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Test error",
            suggestion="Fix it",
        )

        original = QAReport(
            task=task,
            mqm_score=92.5,
            errors=[error],
            status="pass",
            agent_details={"agent1": {"score": 95}},
            score_breakdown={"total_penalty": 5.0},
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize from dict
        restored = QAReport(**data)

        # Verify all fields match
        assert restored.mqm_score == original.mqm_score
        assert restored.status == original.status
        assert len(restored.errors) == len(original.errors)
        assert restored.errors[0].severity == original.errors[0].severity
        assert restored.task.source_text == original.task.source_text
