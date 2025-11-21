"""Unit tests for ConsoleFormatter.

Tests compact and verbose output modes for all CLI commands.
"""

import re
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from kttc.cli.formatters.console import ConsoleFormatter
from kttc.core.models import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask
from kttc.i18n import set_language


@pytest.fixture(autouse=True)
def set_english_language() -> None:
    """Ensure English language is used for all tests."""
    set_language("en")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def mock_task() -> TranslationTask:
    """Create mock translation task."""
    return TranslationTask(
        source_text="This is a test.",
        translation="Это тест.",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def mock_qa_report(mock_task: TranslationTask) -> QAReport:
    """Create mock QA report for testing."""
    return QAReport(
        task=mock_task,
        mqm_score=87.5,
        status="fail",
        errors=[
            ErrorAnnotation(
                category="Accuracy",
                subcategory="Mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=[0, 10],
                description="Incorrect translation of key term",
            ),
            ErrorAnnotation(
                category="Grammar",
                subcategory="Agreement",
                severity=ErrorSeverity.MAJOR,
                location=[15, 25],
                description="Case agreement error",
            ),
            ErrorAnnotation(
                category="Fluency",
                subcategory="Awkward",
                severity=ErrorSeverity.MINOR,
                location=[30, 40],
                description="Awkward phrasing",
            ),
        ],
        critical_error_count=1,
        major_error_count=1,
        minor_error_count=1,
        confidence=0.85,
        agent_agreement=0.90,
    )


@pytest.fixture
def mock_lightweight_scores() -> MagicMock:
    """Create mock lightweight scores."""
    scores = MagicMock()
    scores.chrf = 68.5
    scores.bleu = 42.3
    scores.ter = 71.9
    scores.composite_score = 65.9
    return scores


@pytest.mark.unit
class TestConsoleFormatterCompact:
    """Test ConsoleFormatter compact mode (default)."""

    def test_check_result_compact(
        self, mock_qa_report: QAReport, mock_lightweight_scores: MagicMock
    ) -> None:
        """Test check command output is compact."""
        # Arrange
        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter.print_check_result(
                report=mock_qa_report,
                source_lang="en",
                target_lang="ru",
                lightweight_scores=mock_lightweight_scores,
                rule_based_score=85.0,
                rule_based_errors=[],
                nlp_insights=None,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert: Check key elements are present
        assert "Translation Quality Check" in output_stripped
        assert "en → ru" in output_stripped
        assert "87.5" in output_stripped  # MQM score
        assert "FAIL" in output_stripped
        assert "C:1 M:1 m:1" in output_stripped  # Error counts
        assert "chrF: 68.5" in output_stripped
        assert "BLEU: 42.3" in output_stripped
        assert "TER: 71.9" in output_stripped
        assert "Rule-based: 85" in output_stripped

        # Assert: Compact output (should be <= 20 lines)
        lines = [line for line in output_stripped.split("\n") if line.strip()]
        assert len(lines) <= 20, f"Output too long: {len(lines)} lines"

    def test_check_result_no_errors_compact(self) -> None:
        """Test check output with no errors is even more compact."""
        # Arrange
        task = TranslationTask(
            source_text="Test",
            translation="Тест",
            source_lang="en",
            target_lang="ru",
        )
        report = QAReport(
            task=task,
            mqm_score=98.5,
            status="pass",
            errors=[],
            critical_error_count=0,
            major_error_count=0,
            minor_error_count=0,
        )

        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter.print_check_result(
                report=report,
                source_lang="en",
                target_lang="ru",
                lightweight_scores=None,
                rule_based_score=None,
                rule_based_errors=None,
                nlp_insights=None,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert
        assert "PASS" in output_stripped
        assert "98.5" in output_stripped
        assert "Errors: 0" in output_stripped

        # Should be very compact (< 10 lines)
        lines = [line for line in output_stripped.split("\n") if line.strip()]
        assert len(lines) <= 10, f"Output too long for no errors: {len(lines)} lines"

    def test_compare_result_compact(self) -> None:
        """Test compare command output is compact."""
        # Arrange
        results = [
            {
                "name": "translation1",
                "mqm_score": 92.5,
                "status": "pass",
                "critical_errors": 0,
                "major_errors": 1,
                "minor_errors": 2,
            },
            {
                "name": "translation2",
                "mqm_score": 88.0,
                "status": "fail",
                "critical_errors": 1,
                "major_errors": 0,
                "minor_errors": 1,
            },
            {
                "name": "translation3",
                "mqm_score": 95.0,
                "status": "pass",
                "critical_errors": 0,
                "major_errors": 0,
                "minor_errors": 1,
            },
        ]

        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter.print_compare_result(
                source_lang="en",
                target_lang="ru",
                results=results,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert
        assert "Translation Comparison" in output_stripped
        assert "en → ru" in output_stripped
        assert "Compared: 3 translations" in output_stripped
        assert "Best: translation3" in output_stripped
        assert "95.0" in output_stripped  # Best score
        assert "translation1" in output_stripped
        assert "translation2" in output_stripped

        # Compact output
        lines = [line for line in output_stripped.split("\n") if line.strip()]
        assert len(lines) <= 15, f"Output too long: {len(lines)} lines"

    def test_benchmark_result_compact(self) -> None:
        """Test benchmark command output is compact."""
        # Arrange
        results = [
            {
                "name": "openai",
                "success": True,
                "mqm_score": 92.5,
                "status": "pass",
                "critical_errors": 0,
                "major_errors": 1,
                "minor_errors": 2,
                "duration": 3.5,
            },
            {
                "name": "anthropic",
                "success": True,
                "mqm_score": 94.0,
                "status": "pass",
                "critical_errors": 0,
                "major_errors": 0,
                "minor_errors": 1,
                "duration": 2.8,
            },
            {
                "name": "gigachat",
                "success": False,
                "mqm_score": 0.0,
                "status": "error",
                "critical_errors": 0,
                "major_errors": 0,
                "minor_errors": 0,
                "duration": 1.2,
            },
        ]

        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter.print_benchmark_result(
                source_lang="en",
                target_lang="ru",
                results=results,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert
        assert "Provider Benchmark" in output_stripped
        assert "en → ru" in output_stripped
        assert "Tested: 3 providers" in output_stripped
        assert "Best: anthropic" in output_stripped
        assert "openai" in output_stripped
        assert "gigachat" in output_stripped

        # Compact output
        lines = [line for line in output_stripped.split("\n") if line.strip()]
        assert len(lines) <= 18, f"Output too long: {len(lines)} lines"

    def test_batch_result_compact(self) -> None:
        """Test batch command output is compact."""
        # Arrange
        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter.print_batch_result(
                total=100,
                passed=85,
                failed=15,
                avg_score=89.5,
                total_errors=45,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert
        assert "Batch Processing Complete" in output_stripped
        assert "Total: 100" in output_stripped
        assert "Passed: 85" in output_stripped
        assert "Failed: 15" in output_stripped
        assert "Avg MQM: 89.5" in output_stripped
        assert "Total errors: 45" in output_stripped
        assert "Pass rate: 85%" in output_stripped

        # Very compact output
        lines = [line for line in output_stripped.split("\n") if line.strip()]
        assert len(lines) <= 8, f"Output too long: {len(lines)} lines"


@pytest.mark.unit
class TestConsoleFormatterVerbose:
    """Test ConsoleFormatter verbose mode."""

    def test_check_result_verbose(
        self, mock_qa_report: QAReport, mock_lightweight_scores: MagicMock
    ) -> None:
        """Test check command verbose output includes additional details."""
        # Arrange
        mock_qa_report.agent_scores = {"accuracy_agent": 90.0, "fluency_agent": 85.0}
        mock_qa_report.agent_details = {
            "detected_domain": "medical",
            "domain_confidence": 0.85,
        }

        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter.print_check_result(
                report=mock_qa_report,
                source_lang="en",
                target_lang="ru",
                lightweight_scores=mock_lightweight_scores,
                rule_based_score=85.0,
                rule_based_errors=[],
                nlp_insights=None,
                verbose=True,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert: All compact elements are present
        assert "Translation Quality Check" in output_stripped
        assert "87.5" in output_stripped

        # Assert: Verbose-specific elements
        assert "Detailed Metrics" in output_stripped
        assert "Confidence: 0.85" in output_stripped
        assert "Agent Agreement: 90%" in output_stripped
        assert "Domain: Medical" in output_stripped
        assert "Per-Agent Scores" in output_stripped
        assert "accuracy_agent" in output_stripped or "Accuracy Agent" in output_stripped
        assert "90.00" in output_stripped

        # Verbose output can be longer (20-30 lines)
        lines = [line for line in output_stripped.split("\n") if line.strip()]
        assert len(lines) >= 15, "Verbose output should have more details"
        assert len(lines) <= 35, f"Verbose output too long: {len(lines)} lines"


@pytest.mark.unit
class TestConsoleFormatterErrorTable:
    """Test error table formatting."""

    def test_issues_table_compact(self, mock_qa_report: QAReport) -> None:
        """Test issues table in compact mode."""
        # Arrange
        translation_text = "Это тест с ошибками перевода."
        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter._print_issues_compact(
                errors=mock_qa_report.errors,
                translation_text=translation_text,
                nlp_insights=None,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert: Table structure
        assert "Issues Found" in output_stripped
        assert "Location" in output_stripped
        assert "Fragment" in output_stripped
        assert "Issue" in output_stripped

        # Assert: Errors present (severity badges C/M/m)
        assert "C " in output_stripped or "M " in output_stripped or "m " in output_stripped

    def test_issues_table_with_nlp_insights(self, mock_qa_report: QAReport) -> None:
        """Test issues table includes NLP insights."""
        # Arrange
        translation_text = "Это тест с ошибками перевода."
        nlp_insights = {
            "issues": [
                {
                    "category": "Linguistic",
                    "severity": "minor",
                    "description": "Case agreement issue",
                    "location": [0, 10],
                }
            ]
        }

        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter._print_issues_compact(
                errors=mock_qa_report.errors,
                translation_text=translation_text,
                nlp_insights=nlp_insights,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert: NLP issues included
        assert "Case agreement issue" in output_stripped

    def test_issues_table_with_long_descriptions(self) -> None:
        """Test long descriptions are handled correctly (wrapped by Rich)."""
        # Arrange
        translation_text = "Test text for translation with some errors in it."
        long_error = ErrorAnnotation(
            category="Test",
            subcategory="Test",
            severity=ErrorSeverity.MINOR,
            location=[0, 10],
            description="This is a very long description that should be handled "
            "by the Rich table wrapping feature in compact mode",
        )

        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Act
            ConsoleFormatter._print_issues_compact(
                errors=[long_error],
                translation_text=translation_text,
                nlp_insights=None,
                verbose=False,
            )

            # Get output
            output = console.file.getvalue()
            output_stripped = strip_ansi(output)

        # Assert: Table structure present
        assert "Issues Found" in output_stripped
        # Assert: Description is present (may be wrapped)
        assert "long description" in output_stripped


@pytest.mark.unit
class TestConsoleFormatterColorCoding:
    """Test color coding and styling."""

    def test_score_color_coding(self) -> None:
        """Test MQM scores are color-coded correctly."""
        # Test excellent score (green)
        assert ConsoleFormatter._get_score_color(96.0) == "green"

        # Test good score (yellow)
        assert ConsoleFormatter._get_score_color(90.0) == "yellow"

        # Test poor score (red)
        assert ConsoleFormatter._get_score_color(80.0) == "red"

    def test_status_color_coding(self) -> None:
        """Test status colors."""
        assert ConsoleFormatter._get_status_color("pass") == "green"
        assert ConsoleFormatter._get_status_color("fail") == "red"

    def test_error_format(self) -> None:
        """Test error count formatting."""
        formatted = ConsoleFormatter._format_errors(1, 2, 3)
        assert formatted == "C:1 M:2 m:3"


@pytest.mark.unit
class TestConsoleFormatterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_results(self) -> None:
        """Test handling of empty results."""
        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Should not crash
            ConsoleFormatter.print_compare_result(
                source_lang="en",
                target_lang="ru",
                results=[],
                verbose=False,
            )

            output = console.file.getvalue()

        # Should handle gracefully (may show error or empty state)
        assert output is not None

    def test_none_values(self) -> None:
        """Test handling of None values."""
        task = TranslationTask(
            source_text="Test",
            translation="Тест",
            source_lang="en",
            target_lang="ru",
        )
        report = QAReport(
            task=task,
            mqm_score=90.0,
            status="pass",
            errors=[],
            critical_error_count=0,
            major_error_count=0,
            minor_error_count=0,
            confidence=None,  # None confidence
            agent_agreement=None,  # None agreement
        )

        console = Console(file=StringIO(), width=120, legacy_windows=False)

        with patch("kttc.cli.formatters.console.console", console):
            # Should not crash
            ConsoleFormatter.print_check_result(
                report=report,
                source_lang="en",
                target_lang="ru",
                lightweight_scores=None,
                rule_based_score=None,
                rule_based_errors=None,
                nlp_insights=None,
                verbose=True,
            )

            output = console.file.getvalue()

        # Should handle None gracefully
        assert "90.0" in strip_ansi(output)
