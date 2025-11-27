"""Unit tests for batch command module.

Tests batch processing helper functions and utilities.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kttc.cli.commands.batch import (
    _build_batch_config_info,
    _get_batch_identifier,
    _load_and_apply_glossaries,
    _print_batch_result,
    _save_batch_report,
    _scan_batch_directories,
)
from kttc.core import QAReport
from kttc.core.models import ErrorAnnotation


@pytest.mark.unit
class TestScanBatchDirectories:
    """Test directory scanning for batch processing."""

    def test_scan_finds_matching_pairs(self, tmp_path: Path) -> None:
        """Test scanning finds matching source-translation pairs."""
        # Create source and translation directories
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        # Create matching files
        (source_dir / "file1.txt").write_text("Hello")
        (source_dir / "file2.txt").write_text("World")
        (translation_dir / "file1.txt").write_text("Hola")
        (translation_dir / "file2.txt").write_text("Mundo")

        pairs = _scan_batch_directories(str(source_dir), str(translation_dir), verbose=False)

        assert len(pairs) == 2
        assert all(src.name == trg.name for src, trg in pairs)

    def test_scan_skips_unmatched_files(self, tmp_path: Path) -> None:
        """Test scanning skips files without matching translations."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        # Create source files, only some have translations
        (source_dir / "file1.txt").write_text("Hello")
        (source_dir / "file2.txt").write_text("World")
        (translation_dir / "file1.txt").write_text("Hola")
        # file2.txt has no translation

        pairs = _scan_batch_directories(str(source_dir), str(translation_dir), verbose=False)

        assert len(pairs) == 1
        assert pairs[0][0].name == "file1.txt"

    def test_scan_raises_for_nonexistent_source_dir(self, tmp_path: Path) -> None:
        """Test raises FileNotFoundError for nonexistent source directory."""
        translation_dir = tmp_path / "translations"
        translation_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Source directory not found"):
            _scan_batch_directories(
                str(tmp_path / "nonexistent"), str(translation_dir), verbose=False
            )

    def test_scan_raises_for_nonexistent_translation_dir(self, tmp_path: Path) -> None:
        """Test raises FileNotFoundError for nonexistent translation directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test")

        with pytest.raises(FileNotFoundError, match="Translation directory not found"):
            _scan_batch_directories(str(source_dir), str(tmp_path / "nonexistent"), verbose=False)

    def test_scan_raises_for_empty_source_dir(self, tmp_path: Path) -> None:
        """Test raises ValueError for empty source directory."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        with pytest.raises(ValueError, match="No .txt files found"):
            _scan_batch_directories(str(source_dir), str(translation_dir), verbose=False)

    def test_scan_raises_for_no_matching_pairs(self, tmp_path: Path) -> None:
        """Test raises ValueError when no matching pairs found."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        (source_dir / "file1.txt").write_text("Hello")
        (translation_dir / "other.txt").write_text("Different file")

        with pytest.raises(ValueError, match="No matching source-translation file pairs"):
            _scan_batch_directories(str(source_dir), str(translation_dir), verbose=False)


@pytest.mark.unit
class TestBuildBatchConfigInfo:
    """Test batch config info builder."""

    def test_basic_config_info(self) -> None:
        """Test basic configuration info dictionary."""
        translations = [MagicMock(), MagicMock()]
        groups = {("en", "es"): [MagicMock()]}

        info = _build_batch_config_info(
            file_path="test.csv",
            translations=translations,
            groups=groups,
            threshold=95.0,
            parallel=4,
            batch_size=None,
            smart_routing=False,
            glossary=None,
        )

        assert info["Input File"] == "test.csv"
        assert info["Total Translations"] == "2"
        assert info["Language Pairs"] == "1"
        assert info["Quality Threshold"] == "95.0"
        assert info["Parallel Workers"] == "4"
        assert "Batch Size" not in info
        assert "Smart Routing" not in info
        assert "Glossaries" not in info

    def test_config_info_with_options(self) -> None:
        """Test config info with optional parameters."""
        translations = [MagicMock()]
        groups = {("en", "es"): [MagicMock()], ("en", "fr"): [MagicMock()]}

        info = _build_batch_config_info(
            file_path="batch.json",
            translations=translations,
            groups=groups,
            threshold=90.0,
            parallel=8,
            batch_size=50,
            smart_routing=True,
            glossary="medical,legal",
        )

        assert info["Batch Size"] == "50"
        assert info["Smart Routing"] == "Enabled"
        assert info["Glossaries"] == "medical,legal"


@pytest.mark.unit
class TestGetBatchIdentifier:
    """Test batch translation identifier generation."""

    def test_identifier_without_metadata(self) -> None:
        """Test identifier generation without file metadata."""
        batch_translation = MagicMock()
        batch_translation.metadata = None

        identifier = _get_batch_identifier(0, batch_translation)
        assert identifier == "#1"

        identifier = _get_batch_identifier(4, batch_translation)
        assert identifier == "#5"

    def test_identifier_with_file_metadata(self) -> None:
        """Test identifier generation with file metadata."""
        batch_translation = MagicMock()
        batch_translation.metadata = {"file": "/path/to/document.txt"}

        identifier = _get_batch_identifier(2, batch_translation)
        assert identifier == "document.txt:#3"

    def test_identifier_with_empty_metadata(self) -> None:
        """Test identifier generation with empty metadata dict."""
        batch_translation = MagicMock()
        batch_translation.metadata = {}

        identifier = _get_batch_identifier(0, batch_translation)
        assert identifier == "#1"


@pytest.mark.unit
class TestPrintBatchResult:
    """Test batch result printing."""

    def test_print_passed_result(self) -> None:
        """Test printing passed batch result."""
        report = MagicMock(spec=QAReport)
        report.status = "pass"
        report.mqm_score = 98.5
        mock_console = MagicMock()

        _print_batch_result("test.txt:#1", report, mock_console)

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "green" in call_args
        assert "test.txt:#1" in call_args
        assert "98.50" in call_args

    def test_print_failed_result(self) -> None:
        """Test printing failed batch result."""
        report = MagicMock(spec=QAReport)
        report.status = "fail"
        report.mqm_score = 75.0
        mock_console = MagicMock()

        _print_batch_result("doc.txt:#2", report, mock_console)

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "red" in call_args
        assert "doc.txt:#2" in call_args
        assert "75.00" in call_args


@pytest.mark.unit
class TestLoadAndApplyGlossaries:
    """Test glossary loading and application."""

    def test_no_glossary_does_nothing(self) -> None:
        """Test no action when glossary is None."""
        translations = [MagicMock()]
        # Should not raise or modify anything
        _load_and_apply_glossaries(None, translations)

    def test_loads_and_applies_glossary(self) -> None:
        """Test glossary loading and term application."""
        with patch("kttc.core.GlossaryManager") as mock_manager_cls:
            mock_manager = MagicMock()
            mock_manager_cls.return_value = mock_manager

            # Create mock term
            mock_term = MagicMock()
            mock_term.source = "hello"
            mock_term.target = "hola"
            mock_term.do_not_translate = False
            mock_manager.find_in_text.return_value = [mock_term]

            translation = MagicMock()
            translation.source_text = "hello world"
            translation.source_lang = "en"
            translation.target_lang = "es"
            translation.context = None

            _load_and_apply_glossaries("base,medical", [translation])

            mock_manager.load_multiple.assert_called_once_with(["base", "medical"])
            assert translation.context is not None
            assert "glossary_terms" in translation.context

    def test_handles_glossary_load_error(self) -> None:
        """Test graceful handling of glossary loading errors."""
        with patch("kttc.core.GlossaryManager") as mock_manager_cls:
            mock_manager_cls.side_effect = Exception("Failed to load")

            translations = [MagicMock()]
            # Should not raise, just warn
            _load_and_apply_glossaries("invalid", translations)


@pytest.mark.unit
class TestSaveBatchReport:
    """Test batch report saving."""

    def _create_mock_report(
        self, status: str = "pass", score: float = 95.0, errors: list | None = None
    ) -> MagicMock:
        """Create a mock QAReport."""
        report = MagicMock(spec=QAReport)
        report.status = status
        report.mqm_score = score
        report.errors = errors or []
        return report

    def _create_mock_error(
        self,
        category: str = "accuracy",
        subcategory: str = "mistranslation",
        severity: str = "major",
    ) -> MagicMock:
        """Create a mock error annotation."""
        error = MagicMock(spec=ErrorAnnotation)
        error.category = category
        error.subcategory = subcategory
        error.severity = MagicMock()
        error.severity.value = severity
        error.location = "word"
        error.description = "Test error"
        return error

    def test_save_json_format(self, tmp_path: Path) -> None:
        """Test saving report in JSON format."""
        output_file = tmp_path / "report.json"
        results = [
            ("file1.txt", self._create_mock_report("pass", 98.0)),
            ("file2.txt", self._create_mock_report("fail", 75.0, [self._create_mock_error()])),
        ]

        _save_batch_report(results, str(output_file), 95.0)

        assert output_file.exists()
        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert "summary" in data
        assert data["summary"]["total_files"] == 2
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert len(data["files"]) == 2

    def test_save_text_format(self, tmp_path: Path) -> None:
        """Test saving report in text format."""
        output_file = tmp_path / "report.txt"
        results = [("file1.txt", self._create_mock_report("pass", 95.0))]

        _save_batch_report(results, str(output_file), 95.0)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "BATCH TRANSLATION QUALITY REPORT" in content
        assert "Total files:    1" in content
        assert "PASS" in content

    def test_save_markdown_format(self, tmp_path: Path) -> None:
        """Test saving report in Markdown format."""
        output_file = tmp_path / "report.md"
        results = [
            ("file1.txt", self._create_mock_report("fail", 80.0, [self._create_mock_error()]))
        ]

        _save_batch_report(results, str(output_file), 95.0)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# Batch Translation Quality Report" in content
        assert "| Metric | Value |" in content
        assert "file1.txt" in content

    def test_save_html_format(self, tmp_path: Path) -> None:
        """Test saving report in HTML format."""
        output_file = tmp_path / "report.html"
        results = [("file1.txt", self._create_mock_report("pass", 98.0))]

        _save_batch_report(results, str(output_file), 95.0)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Batch Translation Quality Report" in content
        assert "file1.txt" in content

    def test_auto_detect_format_from_extension(self, tmp_path: Path) -> None:
        """Test automatic format detection from file extension."""
        results = [("file.txt", self._create_mock_report())]

        # Test .md extension
        md_file = tmp_path / "report.md"
        _save_batch_report(results, str(md_file), 95.0)
        assert "# Batch Translation Quality Report" in md_file.read_text(encoding="utf-8")

        # Test .html extension
        html_file = tmp_path / "report.html"
        _save_batch_report(results, str(html_file), 95.0)
        assert "<!DOCTYPE html>" in html_file.read_text(encoding="utf-8")

    def test_explicit_format_overrides_extension(self, tmp_path: Path) -> None:
        """Test explicit format parameter overrides file extension."""
        output_file = tmp_path / "report.json"  # .json extension
        results = [("file.txt", self._create_mock_report())]

        _save_batch_report(results, str(output_file), 95.0, output_format="markdown")

        content = output_file.read_text(encoding="utf-8")
        assert "# Batch Translation Quality Report" in content  # Markdown content

    def test_unknown_format_defaults_to_json(self, tmp_path: Path) -> None:
        """Test unknown format falls back to JSON."""
        output_file = tmp_path / "report.xyz"
        results = [("file.txt", self._create_mock_report())]

        _save_batch_report(results, str(output_file), 95.0, output_format="invalid_format")

        content = output_file.read_text(encoding="utf-8")
        data = json.loads(content)  # Should be valid JSON
        assert "summary" in data
