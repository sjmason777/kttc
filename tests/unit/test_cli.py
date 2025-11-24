"""Unit tests for CLI interface.

Tests CLI commands, argument parsing, and output formatting.
Focus: Fast, isolated tests with mocked dependencies.
"""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from kttc import __version__
from kttc.cli.main import app
from kttc.core.models import QAReport

# Create CLI runner
runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text.

    Typer with Rich adds ANSI color codes to CLI output,
    which breaks simple string matching in tests.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.mark.unit
class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self) -> None:
        """Test --help flag shows usage information."""
        # Act
        result = runner.invoke(app, ["--help"])

        # Assert
        assert result.exit_code == 0
        assert "Knowledge Translation Transmutation Core" in result.stdout
        assert "check" in result.stdout
        assert "translate" in result.stdout

    def test_cli_version(self) -> None:
        """Test --version flag shows version."""
        # Act
        result = runner.invoke(app, ["--version"])

        # Assert
        assert result.exit_code == 0
        assert "KTTC version:" in result.stdout
        assert __version__ in result.stdout

    def test_invalid_command_fails(self) -> None:
        """Test invalid command shows error."""
        # Act
        result = runner.invoke(app, ["invalid-command"])

        # Assert
        assert result.exit_code != 0


@pytest.mark.unit
class TestCheckCommand:
    """Test 'kttc check' command."""

    def test_check_help(self) -> None:
        """Test check command help."""
        # Act
        result = runner.invoke(app, ["check", "--help"])

        # Assert
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "translation quality" in output.lower()
        assert "SOURCE" in output
        assert "--threshold" in output

    def test_check_missing_required_args(self) -> None:
        """Test check fails without required arguments."""
        # Act
        result = runner.invoke(app, ["check"])

        # Assert
        assert result.exit_code != 0

    def test_check_nonexistent_files(self) -> None:
        """Test check fails with nonexistent files."""
        # Act
        result = runner.invoke(
            app,
            [
                "check",
                "nonexistent.txt",
                "nonexistent_trans.txt",
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        # Assert
        assert result.exit_code != 0

    def test_check_success_with_mocks(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport
    ) -> None:
        """Test successful check with mocked orchestrator."""
        # Arrange
        source, translation = temp_text_files

        with patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class:
            with patch("kttc.cli.commands.check.get_settings") as mock_settings:
                # Setup mocks
                mock_orch = MagicMock()
                mock_orch.evaluate = AsyncMock(return_value=sample_qa_report)
                mock_orch_class.return_value = mock_orch

                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.utils.OpenAIProvider"):
                    # Act
                    result = runner.invoke(
                        app,
                        [
                            "check",
                            str(source),
                            str(translation),
                            "--source-lang",
                            "en",
                            "--target-lang",
                            "es",
                        ],
                    )

                    # Assert
                    assert result.exit_code == 0
                    assert "MQM:" in result.stdout or "MQM Score" in result.stdout


@pytest.mark.unit
class TestOutputFormats:
    """Test different output formats."""

    def test_json_output(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport, tmp_path: Path
    ) -> None:
        """Test JSON output format."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.json"

        with patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class:
            with patch("kttc.cli.commands.check.get_settings") as mock_settings:
                mock_orch = MagicMock()
                mock_orch.evaluate = AsyncMock(return_value=sample_qa_report)
                mock_orch_class.return_value = mock_orch

                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.utils.OpenAIProvider"):
                    # Act
                    result = runner.invoke(
                        app,
                        [
                            "check",
                            str(source),
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

                    # Assert
                    assert result.exit_code == 0
                    assert output.exists()

                    # Verify JSON content
                    data = json.loads(output.read_text(encoding="utf-8"))
                    assert "mqm_score" in data
                    assert data["mqm_score"] == 85.0

    def test_markdown_output(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport, tmp_path: Path
    ) -> None:
        """Test Markdown output format."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.md"

        with patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class:
            with patch("kttc.cli.commands.check.get_settings") as mock_settings:
                mock_orch = MagicMock()
                mock_orch.evaluate = AsyncMock(return_value=sample_qa_report)
                mock_orch_class.return_value = mock_orch

                settings = MagicMock()
                settings.default_llm_provider = "openai"
                settings.get_llm_provider_key.return_value = "test-key"
                mock_settings.return_value = settings

                with patch("kttc.cli.utils.OpenAIProvider"):
                    # Act
                    result = runner.invoke(
                        app,
                        [
                            "check",
                            str(source),
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

                    # Assert
                    assert result.exit_code == 0
                    assert output.exists()

                    # Verify Markdown content
                    content = output.read_text(encoding="utf-8")
                    assert "# Translation Quality Report" in content
                    assert "**MQM Score**:" in content or "MQM Score" in content


@pytest.mark.unit
class TestTranslateCommand:
    """Test 'kttc translate' command."""

    def test_translate_help(self) -> None:
        """Test translate command help."""
        # Act
        result = runner.invoke(app, ["translate", "--help"])

        # Assert
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Translate text" in output
        assert "--text" in output
        assert "--source-lang" in output
        assert "--target-lang" in output

    def test_translate_missing_required_args(self) -> None:
        """Test translate fails without required arguments."""
        # Act
        result = runner.invoke(app, ["translate"])

        # Assert
        assert result.exit_code != 0


@pytest.mark.unit
class TestBatchCommand:
    """Test 'kttc batch' command."""

    def test_batch_help(self) -> None:
        """Test batch command help."""
        # Act
        result = runner.invoke(app, ["batch", "--help"])

        # Assert
        assert result.exit_code == 0
        assert "batch" in result.stdout.lower()

    def test_batch_missing_required_args(self) -> None:
        """Test batch fails without required arguments."""
        # Act
        result = runner.invoke(app, ["batch"])

        # Assert
        assert result.exit_code != 0
