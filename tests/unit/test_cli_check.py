"""Comprehensive tests for CLI check command.

Tests the complete implementation of the 'kttc check' command including:
- File loading and validation
- LLM provider integration
- Agent orchestrator integration
- Output formatting (text, JSON, Markdown)
- Error handling and exit codes
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from kttc.cli.formatters import MarkdownFormatter
from kttc.cli.main import _display_report, _save_report, app
from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask

runner = CliRunner()


@pytest.mark.unit
class TestCheckCommand:
    """Tests for the check command implementation."""

    def test_check_command_keyboard_interrupt(self, tmp_path: Path) -> None:
        """Test check command handles keyboard interrupt."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            # Simulate KeyboardInterrupt during evaluation
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                        ],
                    )

                    assert result.exit_code == 130
                    assert "Interrupted" in result.stdout

    def test_check_command_verbose_exception(self, tmp_path: Path) -> None:
        """Test check command shows full exception in verbose mode."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(side_effect=RuntimeError("Test error"))
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--verbose",
                        ],
                    )

                    assert result.exit_code == 1
                    assert "Error" in result.stdout

    def test_check_command_file_read_error(self, tmp_path: Path) -> None:
        """Test check command handles file read errors."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        # Mock Path.read_text to raise an exception
        with patch("kttc.cli.main.Path") as mock_path:
            mock_source_path = MagicMock()
            mock_source_path.exists.return_value = True
            mock_source_path.read_text.side_effect = UnicodeDecodeError(
                "utf-8", b"", 0, 1, "invalid"
            )

            def path_side_effect(arg: str) -> Path:
                if "source" in str(arg):
                    return mock_source_path
                # Return real Path for other files
                return Path(arg)

            mock_path.side_effect = path_side_effect

            result = runner.invoke(
                app,
                [
                    "check",
                    "--source",
                    str(source),
                    "--translation",
                    str(translation),
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "es",
                ],
            )

            assert result.exit_code == 1
            assert "Error" in result.stdout

    def test_check_command_task_creation_error(self, tmp_path: Path) -> None:
        """Test check command handles task creation errors."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        with patch("kttc.cli.main.TranslationTask") as mock_task:
            mock_task.side_effect = ValueError("Invalid task parameters")

            result = runner.invoke(
                app,
                [
                    "check",
                    "--source",
                    str(source),
                    "--translation",
                    str(translation),
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "es",
                ],
            )

            assert result.exit_code == 1
            assert "Error" in result.stdout

    def test_check_command_missing_source_file(self, tmp_path: Path) -> None:
        """Test check command with non-existent source file."""
        translation = tmp_path / "trans.txt"
        translation.write_text("Hola mundo", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "check",
                "--source",
                str(tmp_path / "nonexistent.txt"),
                "--translation",
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        assert result.exit_code == 1
        assert "Source file not found" in result.stdout or "Error" in result.stdout

    def test_check_command_missing_translation_file(self, tmp_path: Path) -> None:
        """Test check command with non-existent translation file."""
        source = tmp_path / "source.txt"
        source.write_text("Hello world", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "check",
                "--source",
                str(source),
                "--translation",
                str(tmp_path / "nonexistent.txt"),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        assert result.exit_code == 1
        assert "Translation file not found" in result.stdout or "Error" in result.stdout

    def test_check_command_success_pass(self, tmp_path: Path) -> None:
        """Test check command with passing translation."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        # Mock the orchestrator evaluation
        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "PASS" in result.stdout
                    assert "98.5" in result.stdout or "98.50" in result.stdout

    def test_check_command_fail_below_threshold(self, tmp_path: Path) -> None:
        """Test check command with failing translation."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        # Create error annotation
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.CRITICAL,
            location=(0, 5),
            description="Critical translation error",
        )

        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=85.0,
            errors=[error],
            status="fail",
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--threshold",
                            "95.0",
                        ],
                    )

                    assert result.exit_code == 1
                    assert "FAIL" in result.stdout
                    assert "85" in result.stdout

    def test_check_command_with_verbose(self, tmp_path: Path) -> None:
        """Test check command with verbose output."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        error = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Minor grammar issue",
        )

        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.0,
            errors=[error],
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--verbose",
                        ],
                    )

                    assert result.exit_code == 0
                    # In verbose mode, we should see loaded characters info
                    assert "Loaded" in result.stdout or "chars" in result.stdout

    def test_check_command_json_output(self, tmp_path: Path) -> None:
        """Test check command with JSON output."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        output = tmp_path / "report.json"

        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--output",
                            str(output),
                            "--format",
                            "json",
                        ],
                    )

                    assert result.exit_code == 0
                    assert output.exists()

                    # Verify JSON content
                    data = json.loads(output.read_text(encoding="utf-8"))
                    assert data["mqm_score"] == 98.5
                    assert data["status"] == "pass"

    def test_check_command_markdown_output(self, tmp_path: Path) -> None:
        """Test check command with Markdown output."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        output = tmp_path / "report.md"

        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Test error",
        )

        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=90.0,
            errors=[error],
            status="fail",
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--output",
                            str(output),
                            "--format",
                            "markdown",
                        ],
                    )

                    assert result.exit_code == 1  # Fails due to low score
                    assert output.exists()

                    # Verify Markdown content
                    content = output.read_text(encoding="utf-8")
                    assert "# Translation Quality Report" in content
                    assert "**MQM Score:**" in content
                    assert "90.00" in content
                    assert "| accuracy |" in content

    def test_check_command_anthropic_provider(self, tmp_path: Path) -> None:
        """Test check command with Anthropic provider."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        mock_report = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                            "--provider",
                            "anthropic",
                        ],
                    )

                    assert result.exit_code == 0
                    # Verify Anthropic provider was used
                    mock_provider.assert_called_once()

    def test_check_command_unknown_provider(self, tmp_path: Path) -> None:
        """Test check command with unknown provider."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

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
                    "check",
                    "--source",
                    str(source),
                    "--translation",
                    str(translation),
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "es",
                ],
            )

            assert result.exit_code == 1
            assert "Unknown provider" in result.stdout or "Error" in result.stdout

    def test_check_command_orchestrator_error(self, tmp_path: Path) -> None:
        """Test check command handles orchestrator errors."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))
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
                            "check",
                            "--source",
                            str(source),
                            "--translation",
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                        ],
                    )

                    assert result.exit_code == 1
                    assert "Error" in result.stdout


@pytest.mark.unit
class TestDisplayReport:
    """Tests for _display_report function."""

    def test_display_report_pass(self) -> None:
        """Test displaying a passing report."""
        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
            errors=[],
            status="pass",
        )

        # Should not raise any exceptions
        _display_report(report, "text", verbose=False)

    def test_display_report_fail(self) -> None:
        """Test displaying a failing report."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.CRITICAL,
            location=(0, 5),
            description="Critical error",
        )

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=80.0,
            errors=[error],
            status="fail",
        )

        # Should not raise any exceptions
        _display_report(report, "text", verbose=False)

    def test_display_report_verbose_with_errors(self) -> None:
        """Test displaying report in verbose mode with errors."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 5),
                description="Critical error",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MAJOR,
                location=(6, 10),
                description="Major error",
            ),
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistency",
                severity=ErrorSeverity.MINOR,
                location=(11, 15),
                description="Minor error",
            ),
        ]

        report = QAReport(
            task=TranslationTask(
                source_text="Hello world",
                translation="Hola mundo",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=85.0,
            errors=errors,
            status="fail",
        )

        # Should not raise any exceptions
        _display_report(report, "text", verbose=True)

    def test_display_report_long_description_truncation(self) -> None:
        """Test that long error descriptions are truncated."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="A" * 100,  # Very long description
        )

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=90.0,
            errors=[error],
            status="fail",
        )

        # Should not raise any exceptions
        _display_report(report, "text", verbose=True)


@pytest.mark.unit
class TestSaveReport:
    """Tests for _save_report function."""

    def test_save_report_json(self, tmp_path: Path) -> None:
        """Test saving report as JSON."""
        output = tmp_path / "report.json"

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
            errors=[],
            status="pass",
        )

        _save_report(report, str(output), "json")

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["mqm_score"] == 98.5
        assert data["status"] == "pass"

    def test_save_report_json_auto_detect(self, tmp_path: Path) -> None:
        """Test saving report as JSON with auto-detection."""
        output = tmp_path / "report.json"

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
            errors=[],
            status="pass",
        )

        _save_report(report, str(output), "text")  # Format is text, but .json extension

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["mqm_score"] == 98.5

    def test_save_report_markdown(self, tmp_path: Path) -> None:
        """Test saving report as Markdown."""
        output = tmp_path / "report.md"

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Test error",
        )

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=90.0,
            errors=[error],
            status="fail",
        )

        _save_report(report, str(output), "markdown")

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "# Translation Quality Report" in content
        assert "90.00" in content

    def test_save_report_markdown_auto_detect(self, tmp_path: Path) -> None:
        """Test saving report as Markdown with auto-detection."""
        output = tmp_path / "report.md"

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
            errors=[],
            status="pass",
        )

        _save_report(report, str(output), "text")  # Format is text, but .md extension

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "# Translation Quality Report" in content

    def test_save_report_default_json(self, tmp_path: Path) -> None:
        """Test saving report defaults to JSON."""
        output = tmp_path / "report.txt"

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
            errors=[],
            status="pass",
        )

        _save_report(report, str(output), "text")

        assert output.exists()
        # Should be JSON format by default
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["mqm_score"] == 98.5


@pytest.mark.unit
class TestGenerateMarkdownReport:
    """Tests for MarkdownFormatter.format_report function."""

    def test_generate_markdown_report_pass(self) -> None:
        """Test generating Markdown for passing report."""
        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=98.5,
            errors=[],
            status="pass",
        )

        markdown = MarkdownFormatter.format_report(report)

        assert "# Translation Quality Report" in markdown
        assert "✅ PASS" in markdown
        assert "98.50" in markdown
        assert "- **Status**: ✅ PASS" in markdown

    def test_generate_markdown_report_fail(self) -> None:
        """Test generating Markdown for failing report."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.CRITICAL,
            location=(0, 5),
            description="Critical error",
        )

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=80.0,
            errors=[error],
            status="fail",
        )

        markdown = MarkdownFormatter.format_report(report)

        assert "# Translation Quality Report" in markdown
        assert "❌ FAIL" in markdown
        assert "80.00" in markdown
        assert "## Issues Found (1)" in markdown
        assert "| accuracy |" in markdown
        assert "| CRITICAL |" in markdown

    def test_generate_markdown_report_escapes_pipes(self) -> None:
        """Test that pipe characters in descriptions are escaped."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Error with | pipe character",
        )

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=90.0,
            errors=[error],
            status="fail",
        )

        markdown = MarkdownFormatter.format_report(report)

        # Pipe should be escaped
        assert "\\|" in markdown
        assert "Error with \\| pipe character" in markdown

    def test_generate_markdown_report_with_long_descriptions(self) -> None:
        """Test that long descriptions are handled properly."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="A" * 100,  # Very long description
        )

        report = QAReport(
            task=TranslationTask(
                source_text="Hello",
                translation="Hola",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=90.0,
            errors=[error],
            status="fail",
        )

        markdown = MarkdownFormatter.format_report(report)

        # Verify the long description is in the markdown
        assert "A" * 50 in markdown  # At least part of the description should be there
        assert "| accuracy |" in markdown

    def test_generate_markdown_report_multiple_errors(self) -> None:
        """Test generating Markdown with multiple errors."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.CRITICAL,
                location=(0, 5),
                description="Critical error",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MAJOR,
                location=(6, 10),
                description="Major error",
            ),
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistency",
                severity=ErrorSeverity.MINOR,
                location=(11, 15),
                description="Minor error",
            ),
        ]

        report = QAReport(
            task=TranslationTask(
                source_text="Hello world test",
                translation="Hola mundo prueba",
                source_lang="en",
                target_lang="es",
            ),
            mqm_score=85.0,
            errors=errors,
            status="fail",
        )

        markdown = MarkdownFormatter.format_report(report)

        assert "## Issues Found (3)" in markdown
        assert "| accuracy |" in markdown
        assert "| fluency |" in markdown
        assert "| terminology |" in markdown
