"""Comprehensive tests for CLI batch and report commands."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from kttc.cli.main import (
    _display_batch_summary,
    _generate_batch_html_report,
    _generate_batch_markdown_report,
    _save_batch_report,
    app,
)
from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask

runner = CliRunner()


@pytest.mark.unit
class TestBatchCommand:
    """Tests for the batch command implementation."""

    def test_batch_command_missing_source_dir(self, tmp_path: Path) -> None:
        """Test batch command with non-existent source directory."""
        translation_dir = tmp_path / "translations"
        translation_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "batch",
                "--source-dir",
                str(tmp_path / "nonexistent"),
                "--translation-dir",
                str(translation_dir),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.stdout or "Error" in result.stdout

    def test_batch_command_missing_translation_dir(self, tmp_path: Path) -> None:
        """Test batch command with non-existent translation directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "batch",
                "--source-dir",
                str(source_dir),
                "--translation-dir",
                str(tmp_path / "nonexistent"),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.stdout or "Error" in result.stdout

    def test_batch_command_no_source_files(self, tmp_path: Path) -> None:
        """Test batch command with no .txt files in source directory."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "batch",
                "--source-dir",
                str(source_dir),
                "--translation-dir",
                str(translation_dir),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        assert result.exit_code == 1
        assert "No .txt files found" in result.stdout or "Error" in result.stdout

    def test_batch_command_no_matching_pairs(self, tmp_path: Path) -> None:
        """Test batch command with no matching translation files."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        # Create source file without matching translation
        (source_dir / "test.txt").write_text("Hello", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "batch",
                "--source-dir",
                str(source_dir),
                "--translation-dir",
                str(translation_dir),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        assert result.exit_code == 1
        assert "No matching" in result.stdout or "Error" in result.stdout

    def test_batch_command_success_all_pass(self, tmp_path: Path) -> None:
        """Test batch command with all passing translations."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        output = tmp_path / "batch_report.json"

        source_dir.mkdir()
        translation_dir.mkdir()

        # Create test files
        (source_dir / "test1.txt").write_text("Hello world", encoding="utf-8")
        (translation_dir / "test1.txt").write_text("Hola mundo", encoding="utf-8")
        (source_dir / "test2.txt").write_text("Good morning", encoding="utf-8")
        (translation_dir / "test2.txt").write_text("Buenos días", encoding="utf-8")

        # Mock reports
        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.0,
            errors=[],
            status="pass",
        )

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(return_value=mock_report)
            mock_orchestrator_class.return_value = mock_orchestrator

            with patch("kttc.cli.main.get_settings") as mock_settings:
                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.default_model = "gpt-4"
                settings.default_temperature = 0.1
                settings.default_max_tokens = 2000
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.main.OpenAIProvider"):
                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "--source-dir",
                            str(source_dir),
                            "--translation-dir",
                            str(translation_dir),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--output",
                            str(output),
                        ],
                    )

                    assert result.exit_code == 0
                    assert "ALL PASS" in result.stdout
                    assert output.exists()

                    # Verify report content
                    data = json.loads(output.read_text())
                    assert data["summary"]["total_files"] == 2
                    assert data["summary"]["passed"] == 2
                    assert data["summary"]["failed"] == 0

    def test_batch_command_with_failures(self, tmp_path: Path) -> None:
        """Test batch command with some failing translations."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        output = tmp_path / "batch_report.json"

        source_dir.mkdir()
        translation_dir.mkdir()

        (source_dir / "test1.txt").write_text("Hello", encoding="utf-8")
        (translation_dir / "test1.txt").write_text("Hola", encoding="utf-8")

        # Create passing and failing reports
        pass_report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.0,
            errors=[],
            status="pass",
        )

        fail_report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=85.0,
            errors=[
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity=ErrorSeverity.MAJOR,
                    location=(0, 5),
                    description="Error",
                )
            ],
            status="fail",
        )

        call_count = {"count": 0}

        async def mock_evaluate(*args: Any, **kwargs: Any) -> QAReport:
            call_count["count"] += 1
            return pass_report if call_count["count"] == 1 else fail_report

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(side_effect=mock_evaluate)
            mock_orchestrator_class.return_value = mock_orchestrator

            with patch("kttc.cli.main.get_settings") as mock_settings:
                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.default_model = "gpt-4"
                settings.default_temperature = 0.1
                settings.default_max_tokens = 2000
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.main.OpenAIProvider"):
                    # Add second file
                    (source_dir / "test2.txt").write_text("Good", encoding="utf-8")
                    (translation_dir / "test2.txt").write_text("Bueno", encoding="utf-8")

                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "--source-dir",
                            str(source_dir),
                            "--translation-dir",
                            str(translation_dir),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--output",
                            str(output),
                        ],
                    )

                    assert result.exit_code == 1
                    assert "FAILED" in result.stdout

    def test_batch_command_keyboard_interrupt(self, tmp_path: Path) -> None:
        """Test batch command handles keyboard interrupt."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        (source_dir / "test.txt").write_text("Hello", encoding="utf-8")
        (translation_dir / "test.txt").write_text("Hola", encoding="utf-8")

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(side_effect=KeyboardInterrupt())
            mock_orchestrator_class.return_value = mock_orchestrator

            with patch("kttc.cli.main.get_settings") as mock_settings:
                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.default_model = "gpt-4"
                settings.default_temperature = 0.1
                settings.default_max_tokens = 2000
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.main.OpenAIProvider"):
                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "--source-dir",
                            str(source_dir),
                            "--translation-dir",
                            str(translation_dir),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                        ],
                    )

                    assert result.exit_code == 130
                    assert "Interrupted" in result.stdout

    def test_batch_command_verbose_skipping_files(self, tmp_path: Path) -> None:
        """Test batch command in verbose mode shows skipped files."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        output = tmp_path / "batch_report.json"

        source_dir.mkdir()
        translation_dir.mkdir()

        # Create source files, but only one has matching translation
        (source_dir / "test1.txt").write_text("Hello", encoding="utf-8")
        (translation_dir / "test1.txt").write_text("Hola", encoding="utf-8")
        (source_dir / "test2.txt").write_text("Good", encoding="utf-8")
        # No translation for test2.txt

        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.0,
            errors=[],
            status="pass",
        )

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(return_value=mock_report)
            mock_orchestrator_class.return_value = mock_orchestrator

            with patch("kttc.cli.main.get_settings") as mock_settings:
                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.default_model = "gpt-4"
                settings.default_temperature = 0.1
                settings.default_max_tokens = 2000
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.main.OpenAIProvider"):
                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "--source-dir",
                            str(source_dir),
                            "--translation-dir",
                            str(translation_dir),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--output",
                            str(output),
                            "--verbose",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Skipping test2.txt" in result.stdout or "no matching" in result.stdout

    def test_batch_command_with_anthropic(self, tmp_path: Path) -> None:
        """Test batch command with Anthropic provider."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        output = tmp_path / "batch_report.json"

        source_dir.mkdir()
        translation_dir.mkdir()

        (source_dir / "test.txt").write_text("Hello", encoding="utf-8")
        (translation_dir / "test.txt").write_text("Hola", encoding="utf-8")

        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.0,
            errors=[],
            status="pass",
        )

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(return_value=mock_report)
            mock_orchestrator_class.return_value = mock_orchestrator

            with patch("kttc.cli.main.get_settings") as mock_settings:
                settings = MagicMock()
                settings.default_llm_provider = "anthropic"
                settings.default_model = "claude-3-opus-20240229"
                settings.default_temperature = 0.1
                settings.default_max_tokens = 2000
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.main.AnthropicProvider") as mock_provider:
                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "--source-dir",
                            str(source_dir),
                            "--translation-dir",
                            str(translation_dir),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--output",
                            str(output),
                            "--provider",
                            "anthropic",
                            "--verbose",
                        ],
                    )

                    assert result.exit_code == 0
                    mock_provider.assert_called_once()

    def test_batch_command_processing_error(self, tmp_path: Path) -> None:
        """Test batch command handles individual file processing errors."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        output = tmp_path / "batch_report.json"

        source_dir.mkdir()
        translation_dir.mkdir()

        (source_dir / "test1.txt").write_text("Hello", encoding="utf-8")
        (translation_dir / "test1.txt").write_text("Hola", encoding="utf-8")
        (source_dir / "test2.txt").write_text("Good", encoding="utf-8")
        (translation_dir / "test2.txt").write_text("Bueno", encoding="utf-8")

        # First file succeeds, second fails
        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.0,
            errors=[],
            status="pass",
        )

        call_count = {"count": 0}

        async def mock_evaluate(*args: Any, **kwargs: Any) -> QAReport:
            call_count["count"] += 1
            if call_count["count"] == 1:
                return mock_report
            else:
                raise RuntimeError("Processing failed")

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(side_effect=mock_evaluate)
            mock_orchestrator_class.return_value = mock_orchestrator

            with patch("kttc.cli.main.get_settings") as mock_settings:
                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.default_model = "gpt-4"
                settings.default_temperature = 0.1
                settings.default_max_tokens = 2000
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.main.OpenAIProvider"):
                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "--source-dir",
                            str(source_dir),
                            "--translation-dir",
                            str(translation_dir),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--output",
                            str(output),
                        ],
                    )

                    # Should still complete, but with error messages
                    assert (
                        "Error processing file" in result.stdout
                        or "Processing failed" in result.stdout
                    )

    def test_batch_command_unknown_provider(self, tmp_path: Path) -> None:
        """Test batch command with unknown provider."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"
        source_dir.mkdir()
        translation_dir.mkdir()

        (source_dir / "test.txt").write_text("Hello", encoding="utf-8")
        (translation_dir / "test.txt").write_text("Hola", encoding="utf-8")

        with patch("kttc.cli.main.get_settings") as mock_settings:
            settings = MagicMock()
            settings.default_llm_provider = "unknown"
            settings.default_model = "test-model"
            settings.default_temperature = 0.1
            settings.default_max_tokens = 2000
            settings.get_llm_provider_key.return_value = "test-key"
            mock_settings.return_value = settings

            result = runner.invoke(
                app,
                [
                    "batch",
                    "--source-dir",
                    str(source_dir),
                    "--translation-dir",
                    str(translation_dir),
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "es",
                ],
            )

            assert result.exit_code == 1
            assert "Unknown provider" in result.stdout or "Error" in result.stdout


@pytest.mark.unit
class TestReportCommand:
    """Tests for the report command implementation."""

    def test_report_command_missing_input_file(self) -> None:
        """Test report command with non-existent input file."""
        result = runner.invoke(app, ["report", "nonexistent.json"])

        assert result.exit_code == 1
        assert "not found" in result.stdout or "Error" in result.stdout

    def test_report_command_markdown_output(self, tmp_path: Path) -> None:
        """Test report command generates Markdown output."""
        input_file = tmp_path / "batch_report.json"
        output_file = tmp_path / "report.md"

        # Create sample batch report
        data = {
            "summary": {
                "total_files": 2,
                "passed": 1,
                "failed": 1,
                "average_score": 91.5,
                "total_errors": 1,
                "threshold": 95.0,
            },
            "files": [
                {
                    "filename": "test1.txt",
                    "status": "pass",
                    "mqm_score": 98.0,
                    "error_count": 0,
                    "errors": [],
                },
                {
                    "filename": "test2.txt",
                    "status": "fail",
                    "mqm_score": 85.0,
                    "error_count": 1,
                    "errors": [
                        {
                            "category": "accuracy",
                            "subcategory": "mistranslation",
                            "severity": "major",
                            "location": [0, 5],
                            "description": "Error description",
                        }
                    ],
                },
            ],
        }

        input_file.write_text(json.dumps(data), encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "report",
                str(input_file),
                "--format",
                "markdown",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        assert "# Batch Translation Quality Report" in content
        assert "test1.txt" in content
        assert "test2.txt" in content

    def test_report_command_html_output(self, tmp_path: Path) -> None:
        """Test report command generates HTML output."""
        input_file = tmp_path / "batch_report.json"
        output_file = tmp_path / "report.html"

        data = {
            "summary": {
                "total_files": 1,
                "passed": 1,
                "failed": 0,
                "average_score": 98.0,
                "total_errors": 0,
                "threshold": 95.0,
            },
            "files": [
                {
                    "filename": "test.txt",
                    "status": "pass",
                    "mqm_score": 98.0,
                    "error_count": 0,
                    "errors": [],
                }
            ],
        }

        input_file.write_text(json.dumps(data), encoding="utf-8")

        result = runner.invoke(
            app,
            ["report", str(input_file), "--format", "html", "--output", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "test.txt" in content

    def test_report_command_auto_output_filename(self, tmp_path: Path) -> None:
        """Test report command auto-generates output filename."""
        input_file = tmp_path / "batch_report.json"

        data = {
            "summary": {
                "total_files": 1,
                "passed": 1,
                "failed": 0,
                "average_score": 98.0,
                "total_errors": 0,
                "threshold": 95.0,
            },
            "files": [
                {
                    "filename": "test.txt",
                    "status": "pass",
                    "mqm_score": 98.0,
                    "error_count": 0,
                    "errors": [],
                }
            ],
        }

        input_file.write_text(json.dumps(data), encoding="utf-8")

        result = runner.invoke(app, ["report", str(input_file), "--format", "markdown"])

        assert result.exit_code == 0
        # Should create batch_report.md
        assert (tmp_path / "batch_report.md").exists()

    def test_report_command_unsupported_format(self, tmp_path: Path) -> None:
        """Test report command with unsupported format."""
        input_file = tmp_path / "batch_report.json"

        data: dict[str, Any] = {"summary": {}, "files": []}
        input_file.write_text(json.dumps(data), encoding="utf-8")

        result = runner.invoke(app, ["report", str(input_file), "--format", "pdf"])

        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout or "Error" in result.stdout


@pytest.mark.unit
class TestBatchHelpers:
    """Tests for batch command helper functions."""

    def test_display_batch_summary(self) -> None:
        """Test _display_batch_summary function."""
        reports = [
            (
                "test1.txt",
                QAReport(
                    task=TranslationTask(
                        source_text="Hello",
                        translation="Hola",
                        source_lang="en",
                        target_lang="es",
                    ),
                    mqm_score=98.0,
                    errors=[],
                    status="pass",
                    comet_score=None,
                ),
            ),
            (
                "test2.txt",
                QAReport(
                    task=TranslationTask(
                        source_text="Good",
                        translation="Bueno",
                        source_lang="en",
                        target_lang="es",
                    ),
                    mqm_score=85.0,
                    errors=[
                        ErrorAnnotation(
                            category="accuracy",
                            subcategory="mistranslation",
                            severity=ErrorSeverity.MAJOR,
                            location=(0, 4),
                            description="Error",
                        )
                    ],
                    status="fail",
                    comet_score=None,
                ),
            ),
        ]

        # Should not raise any exceptions
        _display_batch_summary(reports, 95.0)

    def test_save_batch_report(self, tmp_path: Path) -> None:
        """Test _save_batch_report function."""
        output = tmp_path / "report.json"

        reports = [
            (
                "test.txt",
                QAReport(
                    task=TranslationTask(
                        source_text="Hello",
                        translation="Hola",
                        source_lang="en",
                        target_lang="es",
                    ),
                    mqm_score=98.0,
                    errors=[],
                    status="pass",
                    comet_score=None,
                ),
            )
        ]

        _save_batch_report(reports, str(output), 95.0)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["summary"]["total_files"] == 1
        assert data["summary"]["passed"] == 1
        assert len(data["files"]) == 1

    def test_generate_batch_markdown_report(self) -> None:
        """Test _generate_batch_markdown_report function."""
        data = {
            "summary": {
                "total_files": 1,
                "passed": 1,
                "failed": 0,
                "average_score": 98.0,
                "total_errors": 0,
                "threshold": 95.0,
            },
            "files": [
                {
                    "filename": "test.txt",
                    "status": "pass",
                    "mqm_score": 98.0,
                    "error_count": 0,
                    "errors": [],
                }
            ],
        }

        markdown = _generate_batch_markdown_report(data)

        assert "# Batch Translation Quality Report" in markdown
        assert "test.txt" in markdown
        assert "98.00" in markdown

    def test_generate_batch_markdown_report_with_errors(self) -> None:
        """Test _generate_batch_markdown_report with errors."""
        data = {
            "summary": {
                "total_files": 1,
                "passed": 0,
                "failed": 1,
                "average_score": 85.0,
                "total_errors": 1,
                "threshold": 95.0,
            },
            "files": [
                {
                    "filename": "test.txt",
                    "status": "fail",
                    "mqm_score": 85.0,
                    "error_count": 1,
                    "errors": [
                        {
                            "category": "accuracy",
                            "subcategory": "mistranslation",
                            "severity": "major",
                            "location": [0, 5],
                            "description": "Test error with | pipe",
                        }
                    ],
                }
            ],
        }

        markdown = _generate_batch_markdown_report(data)

        assert "## Detailed Errors" in markdown
        assert "test.txt" in markdown
        assert "accuracy" in markdown
        assert "\\|" in markdown  # Pipe should be escaped

    def test_generate_batch_html_report(self) -> None:
        """Test _generate_batch_html_report function."""
        data = {
            "summary": {
                "total_files": 1,
                "passed": 1,
                "failed": 0,
                "average_score": 98.0,
                "total_errors": 0,
                "threshold": 95.0,
            },
            "files": [
                {
                    "filename": "test.txt",
                    "status": "pass",
                    "mqm_score": 98.0,
                    "error_count": 0,
                    "errors": [],
                }
            ],
        }

        html = _generate_batch_html_report(data)

        assert "<!DOCTYPE html>" in html
        assert "test.txt" in html
        assert "98.0" in html

    def test_generate_batch_html_report_with_errors(self) -> None:
        """Test _generate_batch_html_report with errors."""
        data = {
            "summary": {
                "total_files": 1,
                "passed": 0,
                "failed": 1,
                "average_score": 85.0,
                "total_errors": 2,
                "threshold": 95.0,
            },
            "files": [
                {
                    "filename": "test.txt",
                    "status": "fail",
                    "mqm_score": 85.0,
                    "error_count": 2,
                    "errors": [
                        {
                            "category": "accuracy",
                            "subcategory": "mistranslation",
                            "severity": "critical",
                            "location": [0, 5],
                            "description": "Critical error",
                        },
                        {
                            "category": "fluency",
                            "subcategory": "grammar",
                            "severity": "minor",
                            "location": [6, 10],
                            "description": "Minor error",
                        },
                    ],
                }
            ],
        }

        html = _generate_batch_html_report(data)

        assert "<h2>Detailed Errors</h2>" in html
        assert "accuracy" in html
        assert "fluency" in html
        assert "severity-critical" in html
        assert "severity-minor" in html

    def test_batch_command_error_with_verbose(self, tmp_path: Path) -> None:
        """Test batch command error handling with verbose flag (covers console.print_exception)."""
        source_dir = tmp_path / "source"
        translation_dir = tmp_path / "translations"

        source_dir.mkdir()
        translation_dir.mkdir()

        (source_dir / "test1.txt").write_text("Hello", encoding="utf-8")
        (translation_dir / "test1.txt").write_text("Hola", encoding="utf-8")

        # Mock _batch_async to raise an exception
        with patch("kttc.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = RuntimeError("Critical error in batch processing")

            result = runner.invoke(
                app,
                [
                    "batch",
                    "--source-dir",
                    str(source_dir),
                    "--translation-dir",
                    str(translation_dir),
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "es",
                    "--verbose",
                ],
            )

            # Should fail with error and print exception with verbose
            assert result.exit_code == 1
            assert "Error" in result.stdout or "error" in result.stdout.lower()
