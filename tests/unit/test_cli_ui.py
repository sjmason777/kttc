"""Unit tests for CLI UI module.

Tests Rich UI components and display functions.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestStyleConstants:
    """Test UI style constants."""

    def test_style_constants_defined(self) -> None:
        """Test that style constants are defined."""
        from kttc.cli.ui import STYLE_BOLD_CYAN, STYLE_DIM_YELLOW

        assert STYLE_BOLD_CYAN == "bold cyan"
        assert STYLE_DIM_YELLOW == "dim yellow"

    def test_status_label_constants(self) -> None:
        """Test status label constants."""
        from kttc.cli.ui import STATUS_ACCEPTABLE, STATUS_GOOD, STATUS_POOR

        assert "Good" in STATUS_GOOD
        assert "Acceptable" in STATUS_ACCEPTABLE
        assert "Poor" in STATUS_POOR


@pytest.mark.unit
class TestPrintHeader:
    """Test print_header function."""

    @patch("kttc.cli.ui.console")
    def test_print_header_title_only(self, mock_console: MagicMock) -> None:
        """Test printing header with title only."""
        from kttc.cli.ui import print_header

        print_header("Test Title")

        # Should print empty line, title, and another empty line
        assert mock_console.print.call_count >= 2
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Test Title" in str(c) for c in calls)

    @patch("kttc.cli.ui.console")
    def test_print_header_with_subtitle(self, mock_console: MagicMock) -> None:
        """Test printing header with subtitle."""
        from kttc.cli.ui import print_header

        print_header("Main Title", "Subtitle text")

        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Main Title" in str(c) for c in calls)
        assert any("Subtitle text" in str(c) for c in calls)


@pytest.mark.unit
class TestPrintStartupInfo:
    """Test print_startup_info function."""

    @patch("kttc.cli.ui.console")
    def test_print_startup_info(self, mock_console: MagicMock) -> None:
        """Test printing startup information."""
        from kttc.cli.ui import print_startup_info

        info = {
            "Model": "gpt-4",
            "Temperature": "0.1",
            "Threshold": "95.0",
        }

        print_startup_info(info)

        # Should print a panel
        mock_console.print.assert_called()


@pytest.mark.unit
class TestAnalyzeAdjNounPairs:
    """Test adjective-noun pair analysis."""

    def test_analyze_empty_pairs(self) -> None:
        """Test analysis with empty pairs list."""
        from kttc.cli.ui import _analyze_adj_noun_pairs

        insights: dict = {"good_indicators": [], "issues": []}
        _analyze_adj_noun_pairs([], insights)

        assert insights["good_indicators"] == []
        assert insights["issues"] == []

    def test_analyze_all_correct_pairs(self) -> None:
        """Test analysis when all pairs have correct agreement."""
        from kttc.cli.ui import _analyze_adj_noun_pairs

        pairs = [
            {"agreement": "correct", "text": "красивый дом"},
            {"agreement": "correct", "text": "большая машина"},
        ]
        insights: dict = {"good_indicators": [], "issues": []}

        _analyze_adj_noun_pairs(pairs, insights)

        assert len(insights["good_indicators"]) == 1
        assert "2 pairs verified" in insights["good_indicators"][0]
        assert len(insights["issues"]) == 0

    def test_analyze_incorrect_pairs(self) -> None:
        """Test analysis with incorrect agreement."""
        from kttc.cli.ui import _analyze_adj_noun_pairs

        pairs = [
            {
                "agreement": "incorrect",
                "text": "красивая дом",
                "adjective": {"text": "красивая"},
                "noun": {"text": "дом"},
                "location": [0, 12],
            },
        ]
        insights: dict = {"good_indicators": [], "issues": []}

        _analyze_adj_noun_pairs(pairs, insights)

        assert len(insights["issues"]) == 1
        assert insights["issues"][0]["category"] == "Linguistic"
        assert insights["issues"][0]["subcategory"] == "Case Agreement"

    def test_analyze_pair_without_details(self) -> None:
        """Test analysis with pair missing adjective/noun details."""
        from kttc.cli.ui import _analyze_adj_noun_pairs

        pairs = [
            {"agreement": "incorrect", "text": "test phrase"},
        ]
        insights: dict = {"good_indicators": [], "issues": []}

        _analyze_adj_noun_pairs(pairs, insights)

        # Should skip pairs without adj/noun details
        assert len(insights["issues"]) == 0


@pytest.mark.unit
class TestAnalyzeEntities:
    """Test entity analysis."""

    def test_analyze_entities_no_helper(self) -> None:
        """Test entity analysis without extract_entities method."""
        from kttc.cli.ui import _analyze_entities

        helper = MagicMock(spec=[])  # No extract_entities
        insights: dict = {"entities": []}

        _analyze_entities(helper, "Test text", insights)

        # Should not crash and not modify insights
        assert "entities" in insights


@pytest.mark.unit
class TestModuleExports:
    """Test module exports and re-exports."""

    def test_console_exported(self) -> None:
        """Test console is exported."""
        from kttc.cli.ui import console

        assert console is not None

    def test_print_functions_exported(self) -> None:
        """Test print functions are exported."""
        from kttc.cli.ui import (
            print_error,
            print_header,
            print_info,
            print_startup_info,
            print_success,
            print_warning,
        )

        assert callable(print_error)
        assert callable(print_info)
        assert callable(print_success)
        assert callable(print_warning)
        assert callable(print_header)
        assert callable(print_startup_info)

    def test_all_exports(self) -> None:
        """Test __all__ contains expected exports."""
        from kttc.cli.ui import __all__

        expected = [
            "console",
            "print_error",
            "print_info",
            "print_success",
            "print_warning",
            "print_header",
            "print_startup_info",
        ]
        for name in expected:
            assert name in __all__


@pytest.mark.unit
class TestConsoleUtilityFunctions:
    """Test console utility functions."""

    @patch("kttc.utils.console.console")
    def test_print_error(self, mock_console: MagicMock) -> None:
        """Test print_error function."""
        from kttc.cli.ui import print_error

        print_error("Test error message")

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "Test error message" in call_args

    @patch("kttc.utils.console.console")
    def test_print_success(self, mock_console: MagicMock) -> None:
        """Test print_success function."""
        from kttc.cli.ui import print_success

        print_success("Success message")

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "Success message" in call_args

    @patch("kttc.utils.console.console")
    def test_print_warning(self, mock_console: MagicMock) -> None:
        """Test print_warning function."""
        from kttc.cli.ui import print_warning

        print_warning("Warning message")

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "Warning message" in call_args

    @patch("kttc.utils.console.console")
    def test_print_info(self, mock_console: MagicMock) -> None:
        """Test print_info function."""
        from kttc.cli.ui import print_info

        print_info("Info message")

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "Info message" in call_args


@pytest.mark.unit
class TestScoreColorHelpers:
    """Test score color helper functions."""

    def test_get_score_color_green(self) -> None:
        """Test green color for high score."""
        from kttc.cli.ui import _get_score_color

        assert _get_score_color(95.0) == "green"
        assert _get_score_color(100.0) == "green"

    def test_get_score_color_yellow(self) -> None:
        """Test yellow color for medium score."""
        from kttc.cli.ui import _get_score_color

        assert _get_score_color(90.0) == "yellow"
        assert _get_score_color(85.0) == "yellow"

    def test_get_score_color_red(self) -> None:
        """Test red color for low score."""
        from kttc.cli.ui import _get_score_color

        assert _get_score_color(80.0) == "red"
        assert _get_score_color(50.0) == "red"


@pytest.mark.unit
class TestConfidenceColorHelpers:
    """Test confidence color helper functions."""

    def test_get_confidence_color_green(self) -> None:
        """Test green color for high confidence."""
        from kttc.cli.ui import _get_confidence_color

        assert _get_confidence_color(0.9) == "green"
        assert _get_confidence_color(0.8) == "green"

    def test_get_confidence_color_yellow(self) -> None:
        """Test yellow color for medium confidence."""
        from kttc.cli.ui import _get_confidence_color

        assert _get_confidence_color(0.7) == "yellow"
        assert _get_confidence_color(0.6) == "yellow"

    def test_get_confidence_color_red(self) -> None:
        """Test red color for low confidence."""
        from kttc.cli.ui import _get_confidence_color

        assert _get_confidence_color(0.5) == "red"
        assert _get_confidence_color(0.3) == "red"

    def test_get_confidence_label_high(self) -> None:
        """Test high confidence label."""
        from kttc.cli.ui import _get_confidence_label

        assert _get_confidence_label(0.9) == "high"
        assert _get_confidence_label(0.8) == "high"

    def test_get_confidence_label_medium(self) -> None:
        """Test medium confidence label."""
        from kttc.cli.ui import _get_confidence_label

        assert _get_confidence_label(0.7) == "medium"
        assert _get_confidence_label(0.6) == "medium"

    def test_get_confidence_label_low(self) -> None:
        """Test low confidence label."""
        from kttc.cli.ui import _get_confidence_label

        assert _get_confidence_label(0.5) == "low"
        assert _get_confidence_label(0.3) == "low"


@pytest.mark.unit
class TestMetricColorHelpers:
    """Test metric color helper functions."""

    def test_get_metric_color_chrf(self) -> None:
        """Test metric color for chrF thresholds."""
        from kttc.cli.ui import _METRIC_THRESHOLDS, _get_metric_color

        thresholds = _METRIC_THRESHOLDS["chrf"]
        assert _get_metric_color(85, thresholds) == "green"
        assert _get_metric_color(70, thresholds) == "yellow"
        assert _get_metric_color(55, thresholds) == "dim yellow"
        assert _get_metric_color(40, thresholds) == "red"

    def test_get_quality_status(self) -> None:
        """Test quality status helper."""
        from kttc.cli.ui import (
            STATUS_ACCEPTABLE,
            STATUS_GOOD,
            STATUS_POOR,
            _get_quality_status,
        )

        assert _get_quality_status(80, 70, 50) == STATUS_GOOD
        assert _get_quality_status(60, 70, 50) == STATUS_ACCEPTABLE
        assert _get_quality_status(40, 70, 50) == STATUS_POOR


@pytest.mark.unit
class TestFormatErrorHelpers:
    """Test error formatting helper functions."""

    def test_format_error_severity_critical(self) -> None:
        """Test critical severity formatting."""
        from kttc.cli.ui import _format_error_severity

        result = _format_error_severity("critical")
        assert "CRITICAL" in str(result)

    def test_format_error_severity_major(self) -> None:
        """Test major severity formatting."""
        from kttc.cli.ui import _format_error_severity

        result = _format_error_severity("major")
        assert "MAJOR" in str(result)

    def test_format_error_severity_minor(self) -> None:
        """Test minor severity formatting."""
        from kttc.cli.ui import _format_error_severity

        result = _format_error_severity("minor")
        assert "MINOR" in str(result)

    def test_format_error_location_tuple(self) -> None:
        """Test location formatting with tuple."""
        from kttc.cli.ui import _format_error_location

        assert _format_error_location((0, 10)) == "0-10"
        assert _format_error_location([5, 15]) == "5-15"

    def test_format_error_location_invalid(self) -> None:
        """Test location formatting with invalid input."""
        from kttc.cli.ui import _format_error_location

        assert _format_error_location([1]) == "N/A"
        assert _format_error_location("invalid") == "N/A"

    def test_format_error_suggestion_empty(self) -> None:
        """Test formatting empty suggestion."""
        from kttc.cli.ui import _format_error_suggestion

        result = _format_error_suggestion("", None, False)
        assert "[dim]-[/dim]" in result

    def test_format_error_suggestion_with_text(self) -> None:
        """Test formatting suggestion with text."""
        from kttc.cli.ui import _format_error_suggestion

        result = _format_error_suggestion("Use 'их' instead", None, False)
        assert "Use 'их' instead" in result

    def test_format_error_suggestion_truncated(self) -> None:
        """Test truncation of long suggestion."""
        from kttc.cli.ui import _format_error_suggestion

        long_suggestion = "A" * 100
        result = _format_error_suggestion(long_suggestion, None, False)
        assert "..." in result
        assert len(result) < 100

    def test_format_error_suggestion_with_confidence(self) -> None:
        """Test suggestion with confidence."""
        from kttc.cli.ui import _format_error_suggestion

        result = _format_error_suggestion("Fix this", 0.95, True)
        assert "95%" in result


@pytest.mark.unit
class TestSeverityColorHelpers:
    """Test severity color helper functions."""

    def test_get_severity_color_critical(self) -> None:
        """Test critical severity color."""
        from kttc.cli.ui import _get_severity_color

        assert _get_severity_color("critical") == "red"

    def test_get_severity_color_major(self) -> None:
        """Test major severity color."""
        from kttc.cli.ui import _get_severity_color

        assert _get_severity_color("major") == "yellow"

    def test_get_severity_color_minor(self) -> None:
        """Test minor severity color."""
        from kttc.cli.ui import _get_severity_color

        assert _get_severity_color("minor") == "dim"

    def test_get_score_color_and_icon_high(self) -> None:
        """Test high score color and icon."""
        from kttc.cli.ui import _get_score_color_and_icon

        color, icon = _get_score_color_and_icon(85)
        assert color == "green"
        assert icon == "✓"

    def test_get_score_color_and_icon_medium(self) -> None:
        """Test medium score color and icon."""
        from kttc.cli.ui import _get_score_color_and_icon

        color, icon = _get_score_color_and_icon(70)
        assert color == "yellow"
        assert icon == "⚠"

    def test_get_score_color_and_icon_low(self) -> None:
        """Test low score color and icon."""
        from kttc.cli.ui import _get_score_color_and_icon

        color, icon = _get_score_color_and_icon(50)
        assert color == "red"
        assert icon == "✗"


@pytest.mark.unit
class TestGetNlpInsights:
    """Test NLP insights collection."""

    def test_get_nlp_insights_no_helper(self) -> None:
        """Test with no helper."""
        from kttc.cli.ui import get_nlp_insights

        task = MagicMock()
        result = get_nlp_insights(task, None)
        assert result is None

    def test_get_nlp_insights_unavailable(self) -> None:
        """Test with unavailable helper."""
        from kttc.cli.ui import get_nlp_insights

        task = MagicMock()
        helper = MagicMock()
        helper.is_available.return_value = False
        result = get_nlp_insights(task, helper)
        assert result is None

    def test_get_nlp_insights_no_morphology(self) -> None:
        """Test when morphology is not available."""
        from kttc.cli.ui import get_nlp_insights

        task = MagicMock()
        task.translation = "Test"
        helper = MagicMock()
        helper.is_available.return_value = True
        helper.get_enrichment_data.return_value = {"has_morphology": False}
        result = get_nlp_insights(task, helper)
        assert result is None

    def test_get_nlp_insights_with_data(self) -> None:
        """Test with full NLP data."""
        from kttc.cli.ui import get_nlp_insights

        task = MagicMock()
        task.translation = "Привет мир"
        helper = MagicMock()
        helper.is_available.return_value = True
        helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 2,
            "verb_aspects": {"verb1": "perfective"},
            "adjective_noun_pairs": [],
        }
        helper.extract_entities.return_value = []
        result = get_nlp_insights(task, helper)
        assert result is not None
        assert result["word_count"] == 2


@pytest.mark.unit
class TestBuildErrorBreakdown:
    """Test error breakdown building."""

    def test_build_error_breakdown_all_types(self) -> None:
        """Test breakdown with all error types."""
        from kttc.cli.ui import _build_error_breakdown

        report = MagicMock()
        report.critical_error_count = 1
        report.major_error_count = 2
        report.minor_error_count = 3
        result = _build_error_breakdown(report, 4, 5)
        assert "Critical: 1" in result
        assert "Major: 2" in result
        assert "Minor: 3" in result

    def test_build_error_breakdown_no_errors(self) -> None:
        """Test breakdown with no errors."""
        from kttc.cli.ui import _build_error_breakdown

        report = MagicMock()
        report.critical_error_count = 0
        report.major_error_count = 0
        report.minor_error_count = 0
        result = _build_error_breakdown(report, 0, 0)
        assert result == ""


@pytest.mark.unit
class TestCollectAllIssues:
    """Test issue collection."""

    def test_collect_api_errors(self) -> None:
        """Test collecting API errors."""
        from kttc.cli.ui import _collect_all_issues

        report = MagicMock()
        report.errors = []
        api_errors = ["Connection failed"]
        result = _collect_all_issues(report, None, api_errors)
        assert len(result) == 1
        assert result[0]["category"] == "System Error"

    def test_collect_nlp_issues(self) -> None:
        """Test collecting NLP issues."""
        from kttc.cli.ui import _collect_all_issues

        report = MagicMock()
        report.errors = []
        nlp_insights = {"issues": [{"category": "Linguistic"}]}
        result = _collect_all_issues(report, nlp_insights, None)
        assert len(result) == 1


@pytest.mark.unit
class TestProgressCreation:
    """Test progress spinner creation."""

    def test_create_progress(self) -> None:
        """Test create_progress returns Progress instance."""
        from kttc.cli.ui import create_progress

        progress = create_progress()
        assert progress is not None

    def test_create_step_progress(self) -> None:
        """Test create_step_progress returns Progress instance."""
        from kttc.cli.ui import create_step_progress

        progress = create_step_progress()
        assert progress is not None


@pytest.mark.unit
class TestCheckModels:
    """Test model checking."""

    def test_check_models_with_loader(self) -> None:
        """Test check_models_with_loader always returns True."""
        from kttc.cli.ui import check_models_with_loader

        assert check_models_with_loader() is True


@pytest.mark.unit
class TestPrintTranslationPreview:
    """Test translation preview printing."""

    @patch("kttc.cli.ui.console")
    def test_print_translation_preview(self, mock_console: MagicMock) -> None:
        """Test print_translation_preview function."""
        from kttc.cli.ui import print_translation_preview

        print_translation_preview("Hello", "Привет")
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_translation_preview_long(self, mock_console: MagicMock) -> None:
        """Test print_translation_preview with long text."""
        from kttc.cli.ui import print_translation_preview

        long_text = "x" * 200
        print_translation_preview(long_text, long_text, max_length=50)
        assert mock_console.print.called


@pytest.mark.unit
class TestPrintComparisonTable:
    """Test comparison table printing."""

    @patch("kttc.cli.ui.console")
    def test_print_comparison_table(self, mock_console: MagicMock) -> None:
        """Test print_comparison_table function."""
        from kttc.cli.ui import print_comparison_table

        comparisons = [
            {
                "name": "Provider A",
                "mqm_score": 95.0,
                "critical_errors": 0,
                "major_errors": 1,
                "minor_errors": 2,
                "status": "pass",
                "duration": 1.5,
            },
        ]
        print_comparison_table(comparisons)
        assert mock_console.print.called


@pytest.mark.unit
class TestPrintBenchmarkSummary:
    """Test benchmark summary printing."""

    @patch("kttc.cli.ui.console")
    def test_print_benchmark_summary(self, mock_console: MagicMock) -> None:
        """Test print_benchmark_summary function."""
        from kttc.cli.ui import print_benchmark_summary

        results = {
            "total_providers": 3,
            "test_sentences": 100,
            "avg_mqm": 85.5,
            "avg_duration": 1.2,
            "best_provider": "Provider A",
            "fastest_provider": "Provider C",
            "pass_rate": "2/3",
        }
        print_benchmark_summary(results)
        assert mock_console.print.called


@pytest.mark.unit
class TestBuildResultsTable:
    """Test results table building."""

    def test_build_results_table_pass(self) -> None:
        """Test building results table for passing report."""
        from kttc.cli.ui import _build_results_table

        report = MagicMock()
        report.status = "pass"
        report.mqm_score = 95.0
        report.confidence = 0.9
        report.agent_agreement = 0.95
        report.agent_details = {}
        report.critical_error_count = 0
        report.major_error_count = 0
        report.minor_error_count = 0
        table = _build_results_table(report, 0, 0, 0, False)
        assert table is not None

    def test_build_results_table_fail(self) -> None:
        """Test building results table for failing report."""
        from kttc.cli.ui import _build_results_table

        report = MagicMock()
        report.status = "fail"
        report.mqm_score = 70.0
        report.confidence = 0.6
        report.agent_agreement = 0.7
        report.agent_details = {"detected_domain": "legal", "domain_confidence": 0.8}
        report.critical_error_count = 1
        report.major_error_count = 2
        report.minor_error_count = 3
        table = _build_results_table(report, 6, 0, 0, True)
        assert table is not None


@pytest.mark.unit
class TestBuildRuleBasedTables:
    """Test rule-based table building."""

    def test_build_rule_based_summary_table(self) -> None:
        """Test building rule-based summary table."""
        from kttc.cli.ui import _build_rule_based_summary_table

        errors = [MagicMock()]
        severity_counts = {"critical": 1, "major": 0, "minor": 0}
        table = _build_rule_based_summary_table(85.0, errors, severity_counts)
        assert table is not None

    def test_build_rule_based_error_table(self) -> None:
        """Test building rule-based error table."""
        from kttc.cli.ui import _build_rule_based_error_table

        error = MagicMock()
        error.severity = "major"
        error.check_type = "length_check"
        error.description = "Text too short"
        table = _build_rule_based_error_table([error], False)
        assert table is not None


@pytest.mark.unit
class TestPrintQAReport:
    """Test QA report printing."""

    @patch("kttc.cli.ui.console")
    @patch("kttc.cli.ui._print_unified_error_table")
    def test_print_qa_report_basic(
        self, mock_error_table: MagicMock, mock_console: MagicMock
    ) -> None:
        """Test basic QA report printing."""
        from kttc.cli.ui import print_qa_report
        from kttc.core import QAReport, TranslationTask

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )
        report = QAReport(
            task=task,
            mqm_score=90.0,
            status="pass",
            errors=[],
        )
        print_qa_report(report)
        assert mock_console.print.called


@pytest.mark.unit
class TestPrintRuleBasedErrors:
    """Test rule-based error printing."""

    @patch("kttc.cli.ui.console")
    def test_print_rule_based_errors_no_errors(self, mock_console: MagicMock) -> None:
        """Test printing rule-based errors with no errors."""
        from kttc.cli.ui import print_rule_based_errors

        print_rule_based_errors([], 100.0)
        assert mock_console.print.called

    @patch("kttc.cli.ui.console")
    def test_print_rule_based_errors_with_errors(self, mock_console: MagicMock) -> None:
        """Test printing rule-based errors with errors."""
        from kttc.cli.ui import print_rule_based_errors

        error = MagicMock()
        error.severity = "critical"
        error.check_type = "length_check"
        error.description = "Text too short"
        print_rule_based_errors([error], 75.0, verbose=True)
        assert mock_console.print.called


@pytest.mark.unit
class TestPrintLightweightMetrics:
    """Test lightweight metrics printing."""

    @patch("kttc.cli.ui.console")
    def test_print_lightweight_metrics(self, mock_console: MagicMock) -> None:
        """Test printing lightweight metrics."""
        from kttc.cli.ui import print_lightweight_metrics

        scores = MagicMock()
        scores.chrf = 75.0
        scores.bleu = 45.0
        scores.ter = 65.0
        scores.length_ratio = 0.95
        scores.composite_score = 70.0
        scores.quality_level = "good"
        print_lightweight_metrics(scores)
        assert mock_console.print.called


@pytest.mark.unit
class TestPrintNlpInsights:
    """Test NLP insights printing."""

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_none(self, mock_console: MagicMock) -> None:
        """Test print_nlp_insights with None helper."""
        from kttc.cli.ui import print_nlp_insights

        task = MagicMock()
        print_nlp_insights(task, None)
        # Should return early without printing panel

    @patch("kttc.cli.ui.console")
    def test_print_nlp_insights_with_data(self, mock_console: MagicMock) -> None:
        """Test print_nlp_insights with data."""
        from kttc.cli.ui import print_nlp_insights

        task = MagicMock()
        task.translation = "Test"
        helper = MagicMock()
        helper.is_available.return_value = True
        helper.get_enrichment_data.return_value = {
            "has_morphology": True,
            "word_count": 5,
            "verb_aspects": {},
            "adjective_noun_pairs": [],
        }
        helper.extract_entities.return_value = []
        print_nlp_insights(task, helper)
        assert mock_console.print.called


@pytest.mark.unit
class TestPrintAvailableExtensions:
    """Test available extensions printing."""

    @patch("kttc.cli.ui.console")
    @patch("kttc.utils.dependencies.has_benchmark")
    @patch("kttc.utils.dependencies.has_webui")
    def test_print_available_extensions(
        self, mock_webui: MagicMock, mock_benchmark: MagicMock, mock_console: MagicMock
    ) -> None:
        """Test printing available extensions."""
        mock_benchmark.return_value = True
        mock_webui.return_value = False
        from kttc.cli.ui import print_available_extensions

        print_available_extensions()
        assert mock_console.print.called
