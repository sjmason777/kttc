"""Unit tests for __main__ entry point.

Tests the python -m kttc entry point.
"""

import subprocess
import sys

import pytest


@pytest.mark.unit
class TestMain:
    """Tests for __main__ entry point."""

    def test_main_module_help(self) -> None:
        """Test that python -m kttc --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "kttc", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Knowledge Translation Transmutation Core" in result.stdout

    def test_main_module_version(self) -> None:
        """Test that python -m kttc --version works."""
        result = subprocess.run(
            [sys.executable, "-m", "kttc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "KTTC version:" in result.stdout
        assert "0.1.0" in result.stdout

    def test_main_module_import(self) -> None:
        """Test that __main__ module can be imported."""
        import kttc.__main__  # noqa: F401

        # Should not raise any errors
        assert True

    def test_main_module_executes_run(self) -> None:
        """Test that __main__ module executes run() when called directly."""
        import runpy
        import sys
        from unittest import mock

        # Mock sys.argv to pass --help (so it exits cleanly)
        with mock.patch.object(sys, "argv", ["kttc", "--help"]):
            try:
                # Run the __main__ module as if executed with python -m kttc
                runpy.run_module("kttc.__main__", run_name="__main__")
            except SystemExit:
                # Typer raises SystemExit, which is expected
                pass

        # The test passes if we got here without errors
        assert True
