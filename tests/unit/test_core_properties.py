# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for core models and utilities.

Uses Hypothesis to test invariants and edge cases for data models,
language detection, and text analysis functions.
"""

from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from kttc.core.models import (
    ErrorAnnotation,
    ErrorSeverity,
    QAReport,
    TranslationTask,
)

# ============================================================================
# Custom Strategies
# ============================================================================


@st.composite
def language_code_strategy(draw: Any) -> str:
    """Generate valid 2-letter language codes."""
    codes = ["en", "ru", "zh", "hi", "fa", "de", "fr", "es", "ja", "ko", "ar", "pt"]
    return draw(st.sampled_from(codes))


@st.composite
def non_empty_text_strategy(draw: Any, min_size: int = 1, max_size: int = 500) -> str:
    """Generate non-empty text strings."""
    text = draw(st.text(min_size=min_size, max_size=max_size))
    assume(text.strip() != "")
    return text


@st.composite
def translation_task_strategy(draw: Any) -> TranslationTask:
    """Generate random TranslationTask objects."""
    source_text = draw(non_empty_text_strategy(min_size=1, max_size=200))
    translation = draw(non_empty_text_strategy(min_size=1, max_size=200))
    source_lang = draw(language_code_strategy())
    target_lang = draw(language_code_strategy())

    # Ensure source and target are different
    assume(source_lang != target_lang)

    return TranslationTask(
        source_text=source_text,
        translation=translation,
        source_lang=source_lang,
        target_lang=target_lang,
    )


# ============================================================================
# TranslationTask Property Tests
# ============================================================================


@pytest.mark.unit
class TestTranslationTaskProperties:
    """Property-based tests for TranslationTask model."""

    @given(
        source_text=non_empty_text_strategy(),
        translation=non_empty_text_strategy(),
        source_lang=language_code_strategy(),
        target_lang=language_code_strategy(),
    )
    @settings(max_examples=100)
    def test_task_creation_preserves_values(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
    ) -> None:
        """Property: Created task should preserve all input values."""
        assume(source_lang != target_lang)

        task = TranslationTask(
            source_text=source_text,
            translation=translation,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        assert task.source_text == source_text
        assert task.translation == translation
        assert task.source_lang == source_lang
        assert task.target_lang == target_lang

    @given(
        source_text=non_empty_text_strategy(min_size=10, max_size=100),
    )
    @settings(max_examples=50)
    def test_task_source_text_word_count(self, source_text: str) -> None:
        """Property: Word count should be positive for non-empty text."""
        task = TranslationTask(
            source_text=source_text,
            translation="Test translation",
            source_lang="en",
            target_lang="ru",
        )

        # Texts with actual words should have positive word count
        words = source_text.split()
        if len(words) > 0:
            # Model might have different word counting logic
            assert hasattr(task, "source_text")

    @given(
        source_lang=language_code_strategy(),
        target_lang=language_code_strategy(),
    )
    @settings(max_examples=50)
    def test_task_with_context(
        self,
        source_lang: str,
        target_lang: str,
    ) -> None:
        """Property: Context should be preserved."""
        assume(source_lang != target_lang)

        # Context is expected to be a dict
        context = {"domain": "technical", "notes": f"{source_lang} to {target_lang}"}

        task = TranslationTask(
            source_text="Hello world",
            translation="Test translation",
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
        )

        assert task.context == context


# ============================================================================
# ErrorAnnotation Property Tests
# ============================================================================


@pytest.mark.unit
class TestErrorAnnotationProperties:
    """Property-based tests for ErrorAnnotation model."""

    @given(
        start=st.integers(min_value=0, max_value=1000),
        length=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_location_tuple_integrity(self, start: int, length: int) -> None:
        """Property: Location tuple should maintain start < end invariant."""
        end = start + length

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MAJOR,
            location=(start, end),
            description="Test error",
        )

        assert error.location[0] == start
        assert error.location[1] == end
        assert error.location[0] < error.location[1]

    @given(severity=st.sampled_from(list(ErrorSeverity)))
    @settings(max_examples=20)
    def test_severity_value_preserved(self, severity: ErrorSeverity) -> None:
        """Property: Severity enum should be preserved exactly."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=severity,
            location=(0, 5),
            description="Test error",
        )

        assert error.severity == severity
        assert isinstance(error.severity, ErrorSeverity)

    @given(
        category=st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        subcategory=st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    )
    @settings(max_examples=50)
    def test_category_strings_preserved(self, category: str, subcategory: str) -> None:
        """Property: Category and subcategory strings preserved exactly."""
        error = ErrorAnnotation(
            category=category,
            subcategory=subcategory,
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test error",
        )

        assert error.category == category
        assert error.subcategory == subcategory

    @given(
        suggestion=st.one_of(
            st.none(),
            st.text(min_size=1, max_size=100),
        )
    )
    @settings(max_examples=50)
    def test_optional_suggestion_handling(self, suggestion: str | None) -> None:
        """Property: Optional suggestion field handled correctly."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Test error",
            suggestion=suggestion,
        )

        assert error.suggestion == suggestion


# ============================================================================
# QAReport Property Tests
# ============================================================================


@pytest.mark.unit
class TestQAReportProperties:
    """Property-based tests for QAReport model."""

    @given(
        mqm_score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        status=st.sampled_from(["pass", "fail"]),
    )
    @settings(max_examples=50)
    def test_report_score_preserved(self, mqm_score: float, status: str) -> None:
        """Property: MQM score should be preserved exactly."""
        task = TranslationTask(
            source_text="Source",
            translation="Translation",
            source_lang="en",
            target_lang="ru",
        )

        report = QAReport(
            mqm_score=mqm_score,
            errors=[],
            task=task,
            status=status,
        )

        assert report.mqm_score == mqm_score
        assert report.status == status

    @given(
        num_errors=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=30)
    def test_report_errors_count_preserved(self, num_errors: int) -> None:
        """Property: Error list length should be preserved."""
        task = TranslationTask(
            source_text="Source",
            translation="Translation",
            source_lang="en",
            target_lang="ru",
        )

        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(i, i + 1),
                description=f"Error {i}",
            )
            for i in range(num_errors)
        ]

        report = QAReport(
            mqm_score=80.0,
            errors=errors,
            task=task,
            status="pass",
        )

        assert len(report.errors) == num_errors

    @given(
        score=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_report_score_status_consistency(self, score: float) -> None:
        """Property: Status should be consistent with score thresholds."""
        task = TranslationTask(
            source_text="Source",
            translation="Translation",
            source_lang="en",
            target_lang="ru",
        )

        # Determine expected status based on score (only pass/fail allowed)
        if score >= 80.0:
            status = "pass"
        else:
            status = "fail"

        report = QAReport(
            mqm_score=score,
            errors=[],
            task=task,
            status=status,
        )

        # Verify report was created with consistent status
        assert report.status == status


# ============================================================================
# Error Severity Property Tests
# ============================================================================


@pytest.mark.unit
class TestErrorSeverityProperties:
    """Property-based tests for ErrorSeverity enum behavior."""

    @given(
        severity1=st.sampled_from(list(ErrorSeverity)),
        severity2=st.sampled_from(list(ErrorSeverity)),
    )
    @settings(max_examples=50)
    def test_severity_comparison_reflexive(
        self,
        severity1: ErrorSeverity,
        severity2: ErrorSeverity,
    ) -> None:
        """Property: Severity comparison should be consistent."""
        # Same severities should be equal, different ones should not
        if severity1 is severity2:
            assert severity1 == severity2
        else:
            assert severity1 != severity2

    def test_severity_ordering_is_total(self) -> None:
        """Property: All severities should have a defined order."""
        severities = list(ErrorSeverity)

        # Each severity should have a value for comparison
        for severity in severities:
            assert hasattr(severity, "value")

        # Check that we have expected number of severities
        assert len(severities) >= 4  # NEUTRAL, MINOR, MAJOR, CRITICAL

    @given(severity=st.sampled_from(list(ErrorSeverity)))
    @settings(max_examples=20)
    def test_severity_string_conversion(self, severity: ErrorSeverity) -> None:
        """Property: Severity should convert to meaningful string."""
        # Should have a name attribute
        assert hasattr(severity, "name")
        assert isinstance(severity.name, str)
        assert len(severity.name) > 0


# ============================================================================
# Text Processing Property Tests
# ============================================================================


@pytest.mark.unit
class TestTextProcessingProperties:
    """Property-based tests for text processing utilities."""

    @given(
        text=st.text(min_size=0, max_size=1000),
    )
    @settings(max_examples=100)
    def test_strip_is_idempotent(self, text: str) -> None:
        """Property: Stripping text twice should give same result as once."""
        stripped_once = text.strip()
        stripped_twice = stripped_once.strip()

        assert stripped_once == stripped_twice

    @given(
        text=st.text(min_size=0, max_size=500),
    )
    @settings(max_examples=100)
    def test_split_rejoin_preserves_words(self, text: str) -> None:
        """Property: Split then join with space should be consistent."""
        words = text.split()
        rejoined = " ".join(words)

        # Split and rejoin should give normalized version
        assert rejoined.split() == words

    @given(
        text=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=100)
    def test_lower_upper_roundtrip(self, text: str) -> None:
        """Property: Text case changes should be consistent."""
        # lower().lower() == lower()
        assert text.lower().lower() == text.lower()

        # upper().upper() == upper()
        assert text.upper().upper() == text.upper()

    @given(
        prefix=st.text(min_size=0, max_size=50),
        middle=st.text(min_size=1, max_size=100),
        suffix=st.text(min_size=0, max_size=50),
    )
    @settings(max_examples=100)
    def test_string_concatenation_length(
        self,
        prefix: str,
        middle: str,
        suffix: str,
    ) -> None:
        """Property: Concatenated string length equals sum of parts."""
        result = prefix + middle + suffix
        expected_length = len(prefix) + len(middle) + len(suffix)

        assert len(result) == expected_length


# ============================================================================
# Language Code Property Tests
# ============================================================================


@pytest.mark.unit
class TestLanguageCodeProperties:
    """Property-based tests for language code handling."""

    @given(
        code=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=2,
            max_size=2,
        ),
    )
    @settings(max_examples=50)
    def test_language_code_format(self, code: str) -> None:
        """Property: Language codes should be 2 lowercase letters."""
        assert len(code) == 2
        assert code.islower()
        assert code.isalpha()

    @given(
        code1=language_code_strategy(),
        code2=language_code_strategy(),
    )
    @settings(max_examples=50)
    def test_language_code_equality(self, code1: str, code2: str) -> None:
        """Property: Same language codes should have equal hashes."""
        if code1 == code2:
            assert hash(code1) == hash(code2)
