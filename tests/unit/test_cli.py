"""Unit tests for CLI interface.

Tests CLI commands, argument parsing, and output formatting.
"""

import re

import pytest
from typer.testing import CliRunner

from kttc.cli.main import app

# Create CLI runner
runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text for consistent testing."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


@pytest.mark.unit
class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self) -> None:
        """Test that --help flag works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Knowledge Translation Transmutation Core" in result.stdout
        assert "check" in result.stdout
        assert "translate" in result.stdout
        assert "batch" in result.stdout
        assert "report" in result.stdout

    def test_cli_version(self) -> None:
        """Test that --version flag works."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "KTTC version:" in result.stdout
        assert "0.1.0" in result.stdout

    def test_cli_version_short_flag(self) -> None:
        """Test that -v flag shows version."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "KTTC version:" in result.stdout

    def test_check_command_help(self) -> None:
        """Test check command help."""
        result = runner.invoke(app, ["check", "--help"])
        output = strip_ansi(result.stdout)
        assert result.exit_code == 0
        assert "Check translation quality" in output
        assert "--source" in output
        assert "--translation" in output
        assert "--threshold" in output

    def test_translate_command_help(self) -> None:
        """Test translate command help."""
        result = runner.invoke(app, ["translate", "--help"])
        output = strip_ansi(result.stdout)
        assert result.exit_code == 0
        assert "Translate text with automatic quality checking" in output
        assert "--text" in output
        assert "--source-lang" in output
        assert "--target-lang" in output

    def test_batch_command_help(self) -> None:
        """Test batch command help."""
        result = runner.invoke(app, ["batch", "--help"])
        output = strip_ansi(result.stdout)
        assert result.exit_code == 0
        assert "Batch process multiple translation files" in output
        assert "--source-dir" in output
        assert "--translation-dir" in output

    def test_report_command_help(self) -> None:
        """Test report command help."""
        result = runner.invoke(app, ["report", "--help"])
        output = strip_ansi(result.stdout)
        assert result.exit_code == 0
        assert "Generate formatted report" in output
        assert "--format" in output

    def test_check_command_requires_files(self) -> None:
        """Test that check command requires valid file paths."""
        # Non-existent files should cause an error
        result = runner.invoke(
            app,
            [
                "check",
                "--source",
                "nonexistent_test.txt",
                "--translation",
                "nonexistent_trans.txt",
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )
        # Should fail with error
        assert result.exit_code != 0

    def test_batch_command_not_implemented(self) -> None:
        """Test that batch command shows not implemented message."""
        result = runner.invoke(
            app,
            [
                "batch",
                "--source-dir",
                "./source",
                "--translation-dir",
                "./trans",
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )
        assert "Not implemented yet" in result.stdout or result.exit_code != 0

    def test_report_command_not_implemented(self) -> None:
        """Test that report command shows not implemented message."""
        result = runner.invoke(app, ["report", "results.json"])
        assert "Not implemented yet" in result.stdout or result.exit_code != 0

    def test_invalid_command(self) -> None:
        """Test that invalid command shows error."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

    def test_check_command_missing_required_args(self) -> None:
        """Test check command fails without required arguments."""
        result = runner.invoke(app, ["check"])
        assert result.exit_code != 0

    def test_translate_command_missing_required_args(self) -> None:
        """Test translate command fails without required arguments."""
        result = runner.invoke(app, ["translate"])
        assert result.exit_code != 0

    def test_batch_command_missing_required_args(self) -> None:
        """Test batch command fails without required arguments."""
        result = runner.invoke(app, ["batch"])
        assert result.exit_code != 0

    def test_run_function_executes_app(self) -> None:
        """Test that run() function executes the app."""
        import sys
        from unittest import mock

        from kttc.cli.main import run

        # Mock sys.argv to simulate --help
        with mock.patch.object(sys, "argv", ["kttc", "--help"]):
            # run() will call app() which will exit with 0 for --help
            with pytest.raises(SystemExit) as exc_info:
                run()
            assert exc_info.value.code == 0

    def test_main_module_direct_execution(self) -> None:
        """Test that cli/main.py can be executed directly."""
        import runpy
        import sys
        from unittest import mock

        # Mock sys.argv to pass --help
        with mock.patch.object(sys, "argv", ["kttc", "--help"]):
            try:
                # Run the cli.main module as if executed directly
                runpy.run_module("kttc.cli.main", run_name="__main__")
            except SystemExit:
                # Expected - Typer exits after --help
                pass

        # Test passes if we got here without errors
        assert True
