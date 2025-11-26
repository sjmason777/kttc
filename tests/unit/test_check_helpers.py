"""Unit tests for check_helpers module.

Tests helper functions for the check CLI command including:
- Mode detection (single, compare, batch)
- Report saving in different formats
- Header display functions
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock heavy dependencies before importing check_helpers
sys.modules["spacy"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["jieba"] = MagicMock()
sys.modules["hanlp"] = MagicMock()

# Import after mocking
from kttc.cli.commands.check_helpers import (  # noqa: E402
    calculate_lightweight_metrics,
    detect_check_mode,
    display_check_header,
    perform_smart_routing,
    print_verbose_autodetect_info,
    run_nlp_analysis,
    run_style_analysis,
    save_report,
)
from kttc.core.models import (  # noqa: E402
    ErrorAnnotation,
    ErrorSeverity,
    QAReport,
    TranslationTask,
)


@pytest.fixture
def sample_task() -> TranslationTask:
    """Create sample translation task."""
    return TranslationTask(
        source_text="Hello world",
        translation="Привет мир",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def sample_report(sample_task: TranslationTask) -> QAReport:
    """Create sample QA report."""
    error = ErrorAnnotation(
        category="accuracy",
        subcategory="mistranslation",
        severity=ErrorSeverity.MAJOR,
        location=(0, 5),
        description="Test error",
        suggestion="Fix it",
    )
    return QAReport(
        task=sample_task,
        mqm_score=85.0,
        errors=[error],
        status="pass",
    )


@pytest.fixture
def sample_report_no_errors(sample_task: TranslationTask) -> QAReport:
    """Create sample QA report without errors."""
    return QAReport(
        task=sample_task,
        mqm_score=100.0,
        errors=[],
        status="pass",
    )


class TestDetectCheckMode:
    """Tests for detect_check_mode function."""

    def test_single_file_mode(self, tmp_path: Path) -> None:
        """Test detection of single file mode."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"
        source.write_text("Hello", encoding="utf-8")
        translation.write_text("Привет", encoding="utf-8")

        mode, params = detect_check_mode(str(source), [str(translation)])

        assert mode == "single"
        assert params["source"] == str(source)
        assert params["translation"] == str(translation)

    def test_compare_mode_multiple_translations(self, tmp_path: Path) -> None:
        """Test detection of compare mode with multiple translations."""
        source = tmp_path / "source.txt"
        trans1 = tmp_path / "translation1.txt"
        trans2 = tmp_path / "translation2.txt"
        source.write_text("Hello", encoding="utf-8")
        trans1.write_text("Привет", encoding="utf-8")
        trans2.write_text("Здравствуйте", encoding="utf-8")

        mode, params = detect_check_mode(str(source), [str(trans1), str(trans2)])

        assert mode == "compare"
        assert params["source"] == str(source)
        assert params["translations"] == [str(trans1), str(trans2)]

    def test_batch_csv_mode(self, tmp_path: Path) -> None:
        """Test detection of batch mode with CSV file."""
        csv_file = tmp_path / "batch.csv"
        csv_file.write_text("source,translation\nHello,Привет", encoding="utf-8")

        mode, params = detect_check_mode(str(csv_file), None)

        assert mode == "batch_file"
        assert params["file_path"] == str(csv_file)

    def test_batch_json_mode(self, tmp_path: Path) -> None:
        """Test detection of batch mode with JSON file."""
        json_file = tmp_path / "batch.json"
        json_file.write_text('[{"source": "Hello", "translation": "Привет"}]', encoding="utf-8")

        mode, params = detect_check_mode(str(json_file), None)

        assert mode == "batch_file"
        assert params["file_path"] == str(json_file)

    def test_batch_jsonl_mode(self, tmp_path: Path) -> None:
        """Test detection of batch mode with JSONL file."""
        jsonl_file = tmp_path / "batch.jsonl"
        jsonl_file.write_text('{"source": "Hello", "translation": "Привет"}', encoding="utf-8")

        mode, params = detect_check_mode(str(jsonl_file), None)

        assert mode == "batch_file"
        assert params["file_path"] == str(jsonl_file)

    def test_batch_directory_mode(self, tmp_path: Path) -> None:
        """Test detection of batch mode with directories."""
        source_dir = tmp_path / "source"
        trans_dir = tmp_path / "translation"
        source_dir.mkdir()
        trans_dir.mkdir()

        mode, params = detect_check_mode(str(source_dir), [str(trans_dir)])

        assert mode == "batch_dir"
        assert params["source_dir"] == str(source_dir)
        assert params["translation_dir"] == str(trans_dir)

    def test_directory_mode_requires_one_translation(self, tmp_path: Path) -> None:
        """Test that directory mode requires exactly one translation directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        with pytest.raises(ValueError, match="exactly one translation directory"):
            detect_check_mode(str(source_dir), None)

        with pytest.raises(ValueError, match="exactly one translation directory"):
            detect_check_mode(str(source_dir), ["dir1", "dir2"])

    def test_directory_mode_translation_must_be_dir(self, tmp_path: Path) -> None:
        """Test that translation must be directory when source is directory."""
        source_dir = tmp_path / "source"
        trans_file = tmp_path / "translation.txt"
        source_dir.mkdir()
        trans_file.write_text("Привет", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a directory"):
            detect_check_mode(str(source_dir), [str(trans_file)])

    def test_nonexistent_source_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent source file raises error."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="not found"):
            detect_check_mode(str(nonexistent), ["some_trans.txt"])

    def test_no_translations_raises_error(self, tmp_path: Path) -> None:
        """Test that missing translations raises error."""
        source = tmp_path / "source.txt"
        source.write_text("Hello", encoding="utf-8")

        with pytest.raises(ValueError, match="At least one translation"):
            detect_check_mode(str(source), None)

        with pytest.raises(ValueError, match="At least one translation"):
            detect_check_mode(str(source), [])


class TestSaveReport:
    """Tests for save_report function."""

    def test_save_json_by_format(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test saving report as JSON via format parameter."""
        output = tmp_path / "report.out"

        save_report(sample_report, str(output), "json")

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["mqm_score"] == 85.0
        assert data["status"] == "pass"

    def test_save_json_by_extension(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test saving report as JSON via file extension."""
        output = tmp_path / "report.json"

        save_report(sample_report, str(output), "auto")

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["mqm_score"] == 85.0

    def test_save_markdown_by_format(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test saving report as Markdown via format parameter."""
        output = tmp_path / "report.out"

        save_report(sample_report, str(output), "markdown")

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Translation Quality Report" in content

    def test_save_markdown_by_extension(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test saving report as Markdown via file extension."""
        output = tmp_path / "report.md"

        save_report(sample_report, str(output), "auto")

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Translation Quality Report" in content

    def test_save_html_by_format(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test saving report as HTML via format parameter."""
        output = tmp_path / "report.out"

        save_report(sample_report, str(output), "html")

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_save_html_by_extension(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test saving report as HTML via file extension."""
        output = tmp_path / "report.html"

        save_report(sample_report, str(output), "auto")

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_save_xlsx_fallback_without_openpyxl(
        self, sample_report: QAReport, tmp_path: Path
    ) -> None:
        """Test XLSX saving falls back to JSON when openpyxl unavailable."""
        output = tmp_path / "report.xlsx"

        with patch("kttc.cli.commands.check_helpers.XLSXFormatter") as mock_formatter:
            mock_formatter.is_available.return_value = False
            save_report(sample_report, str(output), "xlsx")

        # Should have created JSON file instead
        json_output = tmp_path / "report.json"
        assert json_output.exists()

    def test_save_xlsx_success(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test XLSX saving when openpyxl is available."""
        output = tmp_path / "report.xlsx"

        with patch("kttc.cli.commands.check_helpers.XLSXFormatter") as mock_formatter:
            mock_formatter.is_available.return_value = True
            save_report(sample_report, str(output), "xlsx")
            mock_formatter.format_report.assert_called_once()

    def test_save_default_format_is_json(self, sample_report: QAReport, tmp_path: Path) -> None:
        """Test that unknown format defaults to JSON."""
        output = tmp_path / "report.unknown"

        save_report(sample_report, str(output), "unknown_format")

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["mqm_score"] == 85.0


class TestDisplayCheckHeader:
    """Tests for display_check_header function."""

    def test_display_header_basic(self) -> None:
        """Test basic header display."""
        with patch("kttc.cli.commands.check_helpers.print_header") as mock_header, patch("kttc.cli.commands.check_helpers.print_startup_info") as mock_info:
            display_check_header(
                source="source.txt",
                translation="trans.txt",
                source_lang="en",
                target_lang="ru",
                threshold=0.8,
                auto_select_model=False,
                auto_correct=False,
                correction_level="minimal",
            )

            mock_header.assert_called_once()
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert call_args["Source File"] == "source.txt"
            assert call_args["Translation File"] == "trans.txt"

    def test_display_header_with_auto_model(self) -> None:
        """Test header display with auto model selection."""
        with patch("kttc.cli.commands.check_helpers.print_header"), patch("kttc.cli.commands.check_helpers.print_startup_info") as mock_info:
            display_check_header(
                source="source.txt",
                translation="trans.txt",
                source_lang="en",
                target_lang="ru",
                threshold=0.8,
                auto_select_model=True,
                auto_correct=False,
                correction_level="minimal",
            )

            call_args = mock_info.call_args[0][0]
            assert "Model Selection" in call_args

    def test_display_header_with_auto_correct(self) -> None:
        """Test header display with auto correction enabled."""
        with patch("kttc.cli.commands.check_helpers.print_header"), patch("kttc.cli.commands.check_helpers.print_startup_info") as mock_info:
            display_check_header(
                source="source.txt",
                translation="trans.txt",
                source_lang="en",
                target_lang="ru",
                threshold=0.8,
                auto_select_model=False,
                auto_correct=True,
                correction_level="full",
            )

            call_args = mock_info.call_args[0][0]
            assert "Auto-Correct" in call_args
            assert "full" in call_args["Auto-Correct"]


class TestPrintVerboseAutodetectInfo:
    """Tests for print_verbose_autodetect_info function."""

    def test_print_info_basic(self) -> None:
        """Test basic info printing."""
        with patch("kttc.cli.commands.check_helpers.console") as mock_console:
            print_verbose_autodetect_info(
                mode="single",
                detected_glossary=None,
                smart_routing=False,
                detected_format="json",
            )

            # Should print mode and format
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("single" in c for c in calls)
            assert any("json" in c for c in calls)

    def test_print_info_with_glossary(self) -> None:
        """Test info printing with glossary."""
        with patch("kttc.cli.commands.check_helpers.console") as mock_console:
            print_verbose_autodetect_info(
                mode="single",
                detected_glossary="tech_glossary",
                smart_routing=False,
                detected_format="json",
            )

            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("tech_glossary" in c for c in calls)

    def test_print_info_with_smart_routing(self) -> None:
        """Test info printing with smart routing enabled."""
        with patch("kttc.cli.commands.check_helpers.console") as mock_console:
            print_verbose_autodetect_info(
                mode="single",
                detected_glossary=None,
                smart_routing=True,
                detected_format="json",
            )

            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("Smart routing" in c or "routing" in c for c in calls)


class TestCalculateLightweightMetrics:
    """Tests for calculate_lightweight_metrics function."""

    def test_metrics_without_reference(self) -> None:
        """Test metrics calculation without reference file."""
        source = "Hello world"
        translation = "Привет мир"

        with patch("kttc.evaluation.LightweightMetrics") as mock_metrics, patch("kttc.evaluation.ErrorDetector") as mock_detector:
            mock_metrics_instance = MagicMock()
            mock_metrics_instance.evaluate.return_value = {"bleu": 0.5}
            mock_metrics.return_value = mock_metrics_instance

            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_all_errors.return_value = []
            mock_detector_instance.calculate_rule_based_score.return_value = 1.0
            mock_detector.return_value = mock_detector_instance

            scores, errors, rule_score = calculate_lightweight_metrics(
                source, translation, None, verbose=False
            )

            assert scores == {"bleu": 0.5}
            assert errors == []
            assert rule_score == 1.0

    def test_metrics_with_reference_file(self, tmp_path: Path) -> None:
        """Test metrics calculation with reference file."""
        source = "Hello world"
        translation = "Привет мир"
        reference_file = tmp_path / "reference.txt"
        reference_file.write_text("Привет мир!", encoding="utf-8")

        with patch("kttc.evaluation.LightweightMetrics") as mock_metrics, patch("kttc.evaluation.ErrorDetector") as mock_detector:
            mock_metrics_instance = MagicMock()
            mock_metrics_instance.evaluate.return_value = {"bleu": 0.9}
            mock_metrics.return_value = mock_metrics_instance

            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_all_errors.return_value = []
            mock_detector_instance.calculate_rule_based_score.return_value = 1.0
            mock_detector.return_value = mock_detector_instance

            scores, errors, rule_score = calculate_lightweight_metrics(
                source, translation, str(reference_file), verbose=True
            )

            assert scores is not None
            mock_metrics_instance.evaluate.assert_called_once()

    def test_metrics_with_nonexistent_reference(self, tmp_path: Path) -> None:
        """Test metrics with nonexistent reference file."""
        source = "Hello world"
        translation = "Привет мир"

        with patch("kttc.evaluation.LightweightMetrics") as mock_metrics, patch("kttc.evaluation.ErrorDetector") as mock_detector, patch("kttc.cli.commands.check_helpers.console"):
            mock_metrics_instance = MagicMock()
            mock_metrics_instance.evaluate.return_value = {"bleu": 0.5}
            mock_metrics.return_value = mock_metrics_instance

            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_all_errors.return_value = []
            mock_detector_instance.calculate_rule_based_score.return_value = 1.0
            mock_detector.return_value = mock_detector_instance

            scores, errors, rule_score = calculate_lightweight_metrics(
                source, translation, str(tmp_path / "nonexistent.txt"), verbose=True
            )

            # Should still return scores using translation as reference
            assert scores is not None

    def test_metrics_exception_handling(self) -> None:
        """Test metrics calculation handles exceptions gracefully."""
        source = "Hello world"
        translation = "Привет мир"

        with patch("kttc.evaluation.LightweightMetrics") as mock_metrics, patch("kttc.evaluation.ErrorDetector"):
            mock_metrics.return_value.evaluate.side_effect = Exception("Calculation failed")

            with patch("kttc.cli.commands.check_helpers.console"):
                scores, errors, rule_score = calculate_lightweight_metrics(
                    source, translation, None, verbose=True
                )

                assert scores is None
                assert errors is None
                assert rule_score is None


class TestPerformSmartRouting:
    """Tests for perform_smart_routing function."""

    def test_smart_routing_success(self, sample_task: TranslationTask) -> None:
        """Test successful smart routing."""
        with patch("kttc.llm.ComplexityRouter") as mock_router, patch("kttc.cli.commands.check_helpers.get_available_providers") as mock_providers:
            mock_providers.return_value = ["openai", "anthropic"]

            mock_score = MagicMock()
            mock_score.overall = 0.5
            mock_score.sentence_length = 0.3
            mock_score.rare_words = 0.2
            mock_score.syntactic = 0.4
            mock_score.domain_specific = 0.1

            mock_router_instance = MagicMock()
            mock_router_instance.route.return_value = ("gpt-4", mock_score)
            mock_router.return_value = mock_router_instance

            settings = MagicMock()

            model, score = perform_smart_routing(
                source_text="Hello world",
                source_lang="en",
                target_lang="ru",
                task=sample_task,
                settings=settings,
                show_routing_info=False,
            )

            assert model == "gpt-4"
            assert score.overall == 0.5

    def test_smart_routing_with_verbose(self, sample_task: TranslationTask) -> None:
        """Test smart routing with verbose output."""
        with patch("kttc.llm.ComplexityRouter") as mock_router, patch("kttc.cli.commands.check_helpers.get_available_providers") as mock_providers, patch("kttc.cli.commands.check_helpers.console") as mock_console:
            mock_providers.return_value = ["openai"]

            mock_score = MagicMock()
            mock_score.overall = 0.5
            mock_score.sentence_length = 0.3
            mock_score.rare_words = 0.2
            mock_score.syntactic = 0.4
            mock_score.domain_specific = 0.1

            mock_router_instance = MagicMock()
            mock_router_instance.route.return_value = ("gpt-4", mock_score)
            mock_router.return_value = mock_router_instance

            settings = MagicMock()

            perform_smart_routing(
                source_text="Hello world",
                source_lang="en",
                target_lang="ru",
                task=sample_task,
                settings=settings,
                show_routing_info=True,
            )

            # Should print complexity info
            assert mock_console.print.called

    def test_smart_routing_failure(self, sample_task: TranslationTask) -> None:
        """Test smart routing handles failures gracefully."""
        with patch("kttc.llm.ComplexityRouter") as mock_router, patch("kttc.cli.commands.check_helpers.console"):
            mock_router.side_effect = Exception("Routing failed")

            settings = MagicMock()

            model, score = perform_smart_routing(
                source_text="Hello world",
                source_lang="en",
                target_lang="ru",
                task=sample_task,
                settings=settings,
                show_routing_info=False,
            )

            assert model is None
            assert score is None


class TestRunNlpAnalysis:
    """Tests for run_nlp_analysis function."""

    def test_nlp_analysis_no_helper(self, sample_task: TranslationTask) -> None:
        """Test NLP analysis when no helper available."""
        with patch("kttc.helpers.get_helper_for_language") as mock_helper:
            mock_helper.return_value = None
            api_errors: list[str] = []

            result = run_nlp_analysis(sample_task, verbose=True, api_errors=api_errors)

            assert result is None

    def test_nlp_analysis_helper_not_available(self, sample_task: TranslationTask) -> None:
        """Test NLP analysis when helper not available."""
        with patch("kttc.helpers.get_helper_for_language") as mock_helper, patch("kttc.cli.commands.check_helpers.console"):
            mock_helper_instance = MagicMock()
            mock_helper_instance.is_available.return_value = False
            mock_helper.return_value = mock_helper_instance
            api_errors: list[str] = []

            result = run_nlp_analysis(sample_task, verbose=True, api_errors=api_errors)

            assert result is None

    def test_nlp_analysis_success(self, sample_task: TranslationTask) -> None:
        """Test successful NLP analysis."""
        with patch("kttc.helpers.get_helper_for_language") as mock_helper, patch("kttc.cli.commands.check_helpers.get_nlp_insights") as mock_insights, patch("kttc.cli.commands.check_helpers.create_step_progress"), patch("kttc.cli.commands.check_helpers.console"):
            mock_helper_instance = MagicMock()
            mock_helper_instance.is_available.return_value = True
            mock_helper.return_value = mock_helper_instance

            mock_insights.return_value = {"tokens": 10}
            api_errors: list[str] = []

            result = run_nlp_analysis(sample_task, verbose=True, api_errors=api_errors)

            assert result == {"tokens": 10}

    def test_nlp_analysis_exception(self, sample_task: TranslationTask) -> None:
        """Test NLP analysis handles exceptions."""
        with patch("kttc.helpers.get_helper_for_language") as mock_helper, patch("kttc.cli.commands.check_helpers.get_nlp_insights") as mock_insights, patch("kttc.cli.commands.check_helpers.create_step_progress"), patch("kttc.cli.commands.check_helpers.console"):
            mock_helper_instance = MagicMock()
            mock_helper_instance.is_available.return_value = True
            mock_helper.return_value = mock_helper_instance

            mock_insights.side_effect = Exception("NLP failed")
            api_errors: list[str] = []

            result = run_nlp_analysis(sample_task, verbose=True, api_errors=api_errors)

            assert result is None
            assert len(api_errors) == 1
            assert "NLP analysis failed" in api_errors[0]


class TestRunStyleAnalysis:
    """Tests for run_style_analysis function."""

    def test_style_analysis_success(self) -> None:
        """Test successful style analysis."""
        with patch("kttc.style.StyleFingerprint") as mock_style:
            mock_profile = MagicMock()
            mock_profile.is_literary = False
            mock_style.return_value.analyze.return_value = mock_profile

            result = run_style_analysis("Hello world", "en", verbose=False)

            assert result == mock_profile

    def test_style_analysis_literary_text(self) -> None:
        """Test style analysis with literary text detection."""
        with patch("kttc.style.StyleFingerprint") as mock_style, patch("kttc.cli.commands.check_helpers.console") as mock_console:
            mock_profile = MagicMock()
            mock_profile.is_literary = True
            mock_profile.detected_pattern.value = "poetic_verse"
            mock_style.return_value.analyze.return_value = mock_profile

            result = run_style_analysis("Once upon a time...", "en", verbose=True)

            assert result == mock_profile
            mock_console.print.assert_called()

    def test_style_analysis_exception(self) -> None:
        """Test style analysis handles exceptions."""
        with patch("kttc.style.StyleFingerprint") as mock_style:
            mock_style.side_effect = Exception("Style analysis failed")

            result = run_style_analysis("Hello world", "en", verbose=False)

            assert result is None


class TestRunNlpAnalysisNonVerbose:
    """Tests for run_nlp_analysis non-verbose branch."""

    def test_nlp_analysis_non_verbose_success(self, sample_task: TranslationTask) -> None:
        """Test NLP analysis in non-verbose mode."""
        with patch("kttc.helpers.get_helper_for_language") as mock_helper, patch("kttc.cli.commands.check_helpers.get_nlp_insights") as mock_insights:
            mock_helper_instance = MagicMock()
            mock_helper_instance.is_available.return_value = True
            mock_helper.return_value = mock_helper_instance

            mock_insights.return_value = {"tokens": 10}
            api_errors: list[str] = []

            result = run_nlp_analysis(sample_task, verbose=False, api_errors=api_errors)

            assert result == {"tokens": 10}

    def test_nlp_analysis_non_verbose_exception(self, sample_task: TranslationTask) -> None:
        """Test NLP analysis exception in non-verbose mode."""
        with patch("kttc.helpers.get_helper_for_language") as mock_helper, patch("kttc.cli.commands.check_helpers.get_nlp_insights") as mock_insights:
            mock_helper_instance = MagicMock()
            mock_helper_instance.is_available.return_value = True
            mock_helper.return_value = mock_helper_instance

            mock_insights.side_effect = Exception("NLP failed")
            api_errors: list[str] = []

            result = run_nlp_analysis(sample_task, verbose=False, api_errors=api_errors)

            assert result is None


class TestRunQualityEvaluation:
    """Tests for run_quality_evaluation async function."""

    @pytest.mark.asyncio
    async def test_run_quality_evaluation_verbose(self, sample_task: TranslationTask) -> None:
        """Test quality evaluation in verbose mode."""
        from kttc.cli.commands.check_helpers import run_quality_evaluation

        mock_llm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.default_temperature = 0.7
        mock_settings.default_max_tokens = 1000

        mock_report = MagicMock()
        mock_report.mqm_score = 85.0

        with patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class, patch("kttc.cli.commands.check_helpers.create_step_progress") as mock_progress, patch("kttc.cli.commands.check_helpers.console"):
            mock_orch = MagicMock()
            mock_orch.evaluate = AsyncMock(return_value=mock_report)
            mock_orch_class.return_value = mock_orch

            mock_progress.return_value.__enter__ = MagicMock()
            mock_progress.return_value.__exit__ = MagicMock()

            api_errors: list[str] = []
            report, orchestrator = await run_quality_evaluation(
                llm_provider=mock_llm,
                task=sample_task,
                threshold=0.8,
                settings=mock_settings,
                verbose=True,
                api_errors=api_errors,
            )

            assert report == mock_report
            assert orchestrator == mock_orch

    @pytest.mark.asyncio
    async def test_run_quality_evaluation_non_verbose(self, sample_task: TranslationTask) -> None:
        """Test quality evaluation in non-verbose mode."""
        from kttc.cli.commands.check_helpers import run_quality_evaluation

        mock_llm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.default_temperature = 0.7
        mock_settings.default_max_tokens = 1000

        mock_report = MagicMock()
        mock_report.mqm_score = 85.0

        with patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class, patch("kttc.cli.commands.check_helpers.create_step_progress") as mock_progress, patch("kttc.cli.commands.check_helpers.console"):
            mock_orch = MagicMock()
            mock_orch.evaluate = AsyncMock(return_value=mock_report)
            mock_orch_class.return_value = mock_orch

            mock_progress.return_value.__enter__ = MagicMock()
            mock_progress.return_value.__exit__ = MagicMock()

            api_errors: list[str] = []
            report, orchestrator = await run_quality_evaluation(
                llm_provider=mock_llm,
                task=sample_task,
                threshold=0.8,
                settings=mock_settings,
                verbose=False,
                api_errors=api_errors,
            )

            assert report == mock_report

    @pytest.mark.asyncio
    async def test_run_quality_evaluation_with_profile(self, sample_task: TranslationTask) -> None:
        """Test quality evaluation with MQM profile."""
        from kttc.cli.commands.check_helpers import run_quality_evaluation

        mock_llm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.default_temperature = 0.7
        mock_settings.default_max_tokens = 1000

        mock_profile = MagicMock()
        mock_profile.agent_weights = {"accuracy": 2.0}

        mock_report = MagicMock()
        mock_report.mqm_score = 85.0

        with patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class, patch("kttc.cli.commands.check_helpers.create_step_progress") as mock_progress, patch("kttc.cli.commands.check_helpers.console"):
            mock_orch = MagicMock()
            mock_orch.evaluate = AsyncMock(return_value=mock_report)
            mock_orch_class.return_value = mock_orch

            mock_progress.return_value.__enter__ = MagicMock()
            mock_progress.return_value.__exit__ = MagicMock()

            api_errors: list[str] = []
            report, orchestrator = await run_quality_evaluation(
                llm_provider=mock_llm,
                task=sample_task,
                threshold=0.8,
                settings=mock_settings,
                verbose=False,
                api_errors=api_errors,
                profile=mock_profile,
            )

            mock_orch_class.assert_called_once()
            call_kwargs = mock_orch_class.call_args[1]
            assert call_kwargs["agent_weights"] == {"accuracy": 2.0}

    @pytest.mark.asyncio
    async def test_run_quality_evaluation_exception(self, sample_task: TranslationTask) -> None:
        """Test quality evaluation handles exceptions."""
        from kttc.cli.commands.check_helpers import run_quality_evaluation

        mock_llm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.default_temperature = 0.7
        mock_settings.default_max_tokens = 1000

        with patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class, patch("kttc.cli.commands.check_helpers.create_step_progress") as mock_progress, patch("kttc.cli.commands.check_helpers.console"):
            mock_orch = MagicMock()
            mock_orch.evaluate = AsyncMock(side_effect=Exception("Evaluation error"))
            mock_orch_class.return_value = mock_orch

            mock_progress.return_value.__enter__ = MagicMock()
            mock_progress.return_value.__exit__ = MagicMock(return_value=False)

            api_errors: list[str] = []
            with pytest.raises(RuntimeError, match="Evaluation failed"):
                await run_quality_evaluation(
                    llm_provider=mock_llm,
                    task=sample_task,
                    threshold=0.8,
                    settings=mock_settings,
                    verbose=True,
                    api_errors=api_errors,
                )


class TestHandleAutoCorrection:
    """Tests for handle_auto_correction async function."""

    @pytest.mark.asyncio
    async def test_auto_correction_disabled(
        self, sample_report: QAReport, sample_task: TranslationTask
    ) -> None:
        """Test that auto correction does nothing when disabled."""
        from kttc.cli.commands.check_helpers import handle_auto_correction

        mock_orch = MagicMock()
        mock_llm = MagicMock()
        mock_settings = MagicMock()

        # Should return immediately when auto_correct is False
        await handle_auto_correction(
            auto_correct=False,
            report=sample_report,
            task=sample_task,
            orchestrator=mock_orch,
            llm_provider=mock_llm,
            translation="trans.txt",
            source_text="Hello",
            source_lang="en",
            target_lang="ru",
            correction_level="minimal",
            settings=mock_settings,
            verbose=False,
        )

        # No corrections should have been attempted
        mock_orch.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_correction_no_errors(self, sample_task: TranslationTask) -> None:
        """Test that auto correction does nothing when no errors."""
        from kttc.cli.commands.check_helpers import handle_auto_correction

        mock_report = MagicMock()
        mock_report.errors = []  # No errors

        mock_orch = MagicMock()
        mock_llm = MagicMock()
        mock_settings = MagicMock()

        await handle_auto_correction(
            auto_correct=True,
            report=mock_report,
            task=sample_task,
            orchestrator=mock_orch,
            llm_provider=mock_llm,
            translation="trans.txt",
            source_text="Hello",
            source_lang="en",
            target_lang="ru",
            correction_level="minimal",
            settings=mock_settings,
            verbose=False,
        )

        mock_orch.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_correction_success(
        self, sample_report: QAReport, sample_task: TranslationTask, tmp_path: Path
    ) -> None:
        """Test successful auto correction."""
        from kttc.cli.commands.check_helpers import handle_auto_correction

        trans_file = tmp_path / "translation.txt"
        trans_file.write_text("Original translation", encoding="utf-8")

        mock_orch = MagicMock()
        mock_corrected_report = MagicMock()
        mock_corrected_report.mqm_score = 95.0
        mock_corrected_report.errors = []
        mock_corrected_report.status = "pass"
        mock_orch.evaluate = AsyncMock(return_value=mock_corrected_report)

        mock_llm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.default_temperature = 0.7

        with patch("kttc.core.correction.AutoCorrector") as mock_corrector_class, patch("kttc.cli.commands.check_helpers.console"), patch(
            "kttc.cli.commands.check_helpers.create_translation_task"
        ) as mock_create_task:
            mock_corrector = MagicMock()
            mock_corrector.auto_correct = AsyncMock(return_value="Corrected text")
            mock_corrector_class.return_value = mock_corrector

            mock_create_task.return_value = sample_task

            await handle_auto_correction(
                auto_correct=True,
                report=sample_report,
                task=sample_task,
                orchestrator=mock_orch,
                llm_provider=mock_llm,
                translation=str(trans_file),
                source_text="Hello",
                source_lang="en",
                target_lang="ru",
                correction_level="full",
                settings=mock_settings,
                verbose=True,
            )

            mock_corrector.auto_correct.assert_called_once()
            corrected_file = tmp_path / "translation_corrected.txt"
            assert corrected_file.exists()

    @pytest.mark.asyncio
    async def test_auto_correction_exception(
        self, sample_report: QAReport, sample_task: TranslationTask, tmp_path: Path
    ) -> None:
        """Test auto correction handles exceptions."""
        from kttc.cli.commands.check_helpers import handle_auto_correction

        trans_file = tmp_path / "translation.txt"
        trans_file.write_text("Original translation", encoding="utf-8")

        mock_orch = MagicMock()
        mock_llm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.default_temperature = 0.7

        with patch("kttc.core.correction.AutoCorrector") as mock_corrector_class, patch("kttc.cli.commands.check_helpers.console"):
            mock_corrector = MagicMock()
            mock_corrector.auto_correct = AsyncMock(side_effect=Exception("Correction failed"))
            mock_corrector_class.return_value = mock_corrector

            # Should not raise, just log warning
            await handle_auto_correction(
                auto_correct=True,
                report=sample_report,
                task=sample_task,
                orchestrator=mock_orch,
                llm_provider=mock_llm,
                translation=str(trans_file),
                source_text="Hello",
                source_lang="en",
                target_lang="ru",
                correction_level="minimal",
                settings=mock_settings,
                verbose=True,
            )
