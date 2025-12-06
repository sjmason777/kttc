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

        with (
            patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
        ):
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

        with (
            patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
        ):
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

        with (
            patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
        ):
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


@pytest.mark.unit
class TestEnsembleMode:
    """Test ensemble mode CLI flags."""

    def test_check_help_shows_ensemble_options(self) -> None:
        """Test check --help shows ensemble-related options."""
        result = runner.invoke(app, ["check", "--help"])

        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        # Should show llm-count and llm options
        assert "--llm-count" in output or "llm_count" in output.replace("-", "_")
        assert "--llm" in output

    def test_llm_count_activates_ensemble(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport
    ) -> None:
        """Test --llm-count > 1 activates ensemble mode."""
        source, translation = temp_text_files

        # Add ensemble_metadata to the report to simulate ensemble mode result
        sample_qa_report.ensemble_metadata = {
            "ensemble_mode": True,
            "providers_total": 2,
            "providers_successful": 2,
        }

        with (
            patch("kttc.agents.MultiProviderAgentOrchestrator") as mock_multi_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
            patch("kttc.cli.utils.setup_multi_llm_providers") as mock_setup,
        ):
            # Setup mocks
            mock_multi_orch = MagicMock()
            mock_multi_orch.evaluate = AsyncMock(return_value=sample_qa_report)
            mock_multi_orch_class.return_value = mock_multi_orch

            mock_setup.return_value = {"openai": MagicMock(), "anthropic": MagicMock()}

            settings = MagicMock()
            settings.default_llm_provider = "openai"
            settings.get_llm_provider_key.return_value = "test-key"
            mock_settings.return_value = settings

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
                    "--llm-count",
                    "2",
                ],
            )

            # Assert
            assert result.exit_code == 0
            # MultiProviderAgentOrchestrator should be instantiated
            mock_multi_orch_class.assert_called_once()

    def test_multiple_llm_providers_activates_ensemble(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport
    ) -> None:
        """Test multiple --llm providers activates ensemble mode."""
        source, translation = temp_text_files

        sample_qa_report.ensemble_metadata = {
            "ensemble_mode": True,
            "providers_total": 2,
        }

        with (
            patch("kttc.agents.MultiProviderAgentOrchestrator") as mock_multi_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
            patch("kttc.cli.utils.setup_multi_llm_providers") as mock_setup,
        ):
            mock_multi_orch = MagicMock()
            mock_multi_orch.evaluate = AsyncMock(return_value=sample_qa_report)
            mock_multi_orch_class.return_value = mock_multi_orch

            mock_setup.return_value = {"openai": MagicMock(), "anthropic": MagicMock()}

            settings = MagicMock()
            settings.default_llm_provider = "openai"
            settings.get_llm_provider_key.return_value = "test-key"
            mock_settings.return_value = settings

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
                    "--llm",
                    "openai,anthropic",
                ],
            )

            # Assert
            assert result.exit_code == 0
            mock_multi_orch_class.assert_called_once()

    def test_single_provider_does_not_activate_ensemble(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport
    ) -> None:
        """Test single provider uses standard orchestrator, not ensemble."""
        source, translation = temp_text_files

        with (
            patch("kttc.cli.commands.check_helpers.AgentOrchestrator") as mock_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
        ):
            mock_orch = MagicMock()
            mock_orch.evaluate = AsyncMock(return_value=sample_qa_report)
            mock_orch_class.return_value = mock_orch

            settings = MagicMock()
            settings.default_llm_provider = "openai"
            settings.get_llm_provider_key.return_value = "test-key"
            mock_settings.return_value = settings

            with patch("kttc.cli.utils.OpenAIProvider"):
                # Act - single provider (no --llm-count, single --llm)
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
                        "--llm",
                        "openai",
                    ],
                )

                # Assert
                assert result.exit_code == 0
                # Standard AgentOrchestrator should be used
                mock_orch_class.assert_called_once()

    def test_ensemble_orchestrator_receives_correct_providers(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport
    ) -> None:
        """Test ensemble orchestrator receives providers from setup_multi_llm_providers."""
        source, translation = temp_text_files

        sample_qa_report.ensemble_metadata = {"ensemble_mode": True}

        mock_providers = {"openai": MagicMock(), "anthropic": MagicMock()}

        with (
            patch("kttc.agents.MultiProviderAgentOrchestrator") as mock_multi_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
            # Patch where the function is used, not where it's defined
            patch("kttc.cli.commands.check.setup_multi_llm_providers") as mock_setup,
        ):
            mock_multi_orch = MagicMock()
            mock_multi_orch.evaluate = AsyncMock(return_value=sample_qa_report)
            mock_multi_orch_class.return_value = mock_multi_orch

            mock_setup.return_value = mock_providers

            settings = MagicMock()
            settings.default_llm_provider = "openai"
            settings.get_llm_provider_key.return_value = "test-key"
            mock_settings.return_value = settings

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
                    "--llm-count",
                    "2",
                ],
            )

            # Assert
            assert result.exit_code == 0
            # Verify MultiProviderAgentOrchestrator was called with providers
            mock_multi_orch_class.assert_called_once()
            call_kwargs = mock_multi_orch_class.call_args
            assert call_kwargs is not None
            # Check providers were passed
            if call_kwargs.kwargs:
                assert "providers" in call_kwargs.kwargs
                assert call_kwargs.kwargs["providers"] == mock_providers

    def test_demo_mode_with_ensemble(
        self, temp_text_files: tuple[Path, Path], sample_qa_report: QAReport
    ) -> None:
        """Test --demo with ensemble mode still works."""
        source, translation = temp_text_files

        sample_qa_report.ensemble_metadata = {"ensemble_mode": True}

        with (
            patch("kttc.agents.MultiProviderAgentOrchestrator") as mock_multi_orch_class,
            patch("kttc.cli.commands.check.get_settings") as mock_settings,
        ):
            mock_multi_orch = MagicMock()
            mock_multi_orch.evaluate = AsyncMock(return_value=sample_qa_report)
            mock_multi_orch_class.return_value = mock_multi_orch

            settings = MagicMock()
            mock_settings.return_value = settings

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
                    "--llm-count",
                    "2",
                    "--demo",
                ],
            )

            # Assert - demo mode should still use ensemble orchestrator
            assert result.exit_code == 0
            mock_multi_orch_class.assert_called_once()
