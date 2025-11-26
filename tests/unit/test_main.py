"""Unit tests for main module entry point.

Tests the main entry point without actually running the CLI.
"""

import pytest

import kttc.__main__


@pytest.mark.unit
class TestMainModule:
    """Test main module functionality."""

    def test_main_imports_run_function(self) -> None:
        """Test that main module imports the run function."""
        # This test ensures the main module can be imported without errors
        # and that it properly imports the run function from cli.main
        assert hasattr(kttc.__main__, "run")

    def test_main_module_has_docstring(self) -> None:
        """Test that main module has proper documentation."""
        # Act & Assert
        assert kttc.__main__.__doc__ is not None
        assert "Entry point" in kttc.__main__.__doc__
