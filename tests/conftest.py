"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import pytest

from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask

# Configure anyio to use only asyncio backend (not trio)
pytest_plugins = ("anyio",)


@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to use asyncio backend only."""
    return "asyncio"


@pytest.fixture
def sample_translation_task() -> TranslationTask:
    """Sample translation task for testing.

    Returns:
        Simple English to Spanish translation task
    """
    return TranslationTask(
        source_text="Hello, world!",
        translation="Hola, mundo!",
        source_lang="en",
        target_lang="es",
    )


@pytest.fixture
def sample_translation_task_with_context() -> TranslationTask:
    """Sample translation task with context metadata.

    Returns:
        Translation task with domain context
    """
    return TranslationTask(
        source_text="The patient presented with acute symptoms.",
        translation="El paciente presentó síntomas agudos.",
        source_lang="en",
        target_lang="es",
        context={
            "domain": "medical",
            "style": "formal",
        },
    )


@pytest.fixture
def sample_error_annotation() -> ErrorAnnotation:
    """Sample error annotation for testing.

    Returns:
        Major accuracy error with suggestion
    """
    return ErrorAnnotation(
        category="accuracy",
        subcategory="mistranslation",
        severity=ErrorSeverity.MAJOR,
        location=(0, 5),
        description="Incorrect translation of 'hello'",
        suggestion="Use 'hola' instead of 'ola'",
    )


@pytest.fixture
def sample_error_annotations() -> list[ErrorAnnotation]:
    """Multiple error annotations for testing.

    Returns:
        List of errors with different severities
    """
    return [
        ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.CRITICAL,
            location=(0, 5),
            description="Critical mistranslation",
            suggestion="Fix meaning",
        ),
        ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MAJOR,
            location=(10, 15),
            description="Grammar error",
            suggestion="Fix verb conjugation",
        ),
        ErrorAnnotation(
            category="terminology",
            subcategory="inconsistency",
            severity=ErrorSeverity.MINOR,
            location=(20, 25),
            description="Inconsistent term usage",
            suggestion="Use glossary term",
        ),
    ]


@pytest.fixture
def sample_qa_report_pass(
    sample_translation_task: TranslationTask,
) -> QAReport:
    """Sample QA report with passing score.

    Returns:
        Report with high MQM score and no errors
    """
    return QAReport(
        task=sample_translation_task,
        mqm_score=98.5,
        comet_score=0.95,
        errors=[],
        status="pass",
    )


@pytest.fixture
def sample_qa_report_fail(
    sample_translation_task: TranslationTask,
    sample_error_annotations: list[ErrorAnnotation],
) -> QAReport:
    """Sample QA report with failing score.

    Returns:
        Report with low MQM score and multiple errors
    """
    return QAReport(
        task=sample_translation_task,
        mqm_score=82.3,
        comet_score=0.65,
        errors=sample_error_annotations,
        status="fail",
    )


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
