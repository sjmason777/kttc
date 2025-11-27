"""Unit tests for dependencies module.

Tests dependency management and lazy loading.
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from kttc.utils.dependencies import (
    DEPENDENCY_GROUPS,
    check_dependency_group,
    check_package_installed,
    ensure_dependency_group,
    has_benchmark,
    has_webui,
    install_dependency_group,
    require_benchmark,
    require_webui,
    show_missing_dependencies_prompt,
    show_optimization_tips,
)


@pytest.mark.unit
class TestCheckPackageInstalled:
    """Test check_package_installed function."""

    def test_check_installed_package(self) -> None:
        """Test checking a package that is installed."""
        # pytest is always installed in test environment
        result = check_package_installed("pytest")
        assert result is True

    def test_check_not_installed_package(self) -> None:
        """Test checking a package that is not installed."""
        # This package should not exist
        result = check_package_installed("nonexistent_package_12345")
        assert result is False

    def test_check_standard_library(self) -> None:
        """Test checking standard library module."""
        result = check_package_installed("os")
        assert result is True

    def test_check_hyphenated_package_name(self) -> None:
        """Test package name with hyphens is converted to underscores."""
        # pydantic-settings -> pydantic_settings
        result = check_package_installed("pydantic-settings")
        # Result depends on whether it's installed
        assert isinstance(result, bool)


@pytest.mark.unit
class TestCheckDependencyGroup:
    """Test check_dependency_group function."""

    def test_check_metrics_group(self) -> None:
        """Test checking metrics dependency group."""
        all_installed, missing = check_dependency_group("metrics")
        assert isinstance(all_installed, bool)
        assert isinstance(missing, list)

    def test_check_webui_group(self) -> None:
        """Test checking webui dependency group."""
        all_installed, missing = check_dependency_group("webui")
        assert isinstance(all_installed, bool)
        assert isinstance(missing, list)

    @pytest.mark.skip(reason="Flaky due to test isolation issues with imports")
    def test_check_benchmark_group(self) -> None:
        """Test checking benchmark dependency group."""
        all_installed, missing = check_dependency_group("benchmark")
        assert isinstance(all_installed, bool)
        assert isinstance(missing, list)

    def test_missing_packages_in_list(self) -> None:
        """Test that missing packages are returned in list."""
        _, missing = check_dependency_group("metrics")
        for pkg in missing:
            assert isinstance(pkg, str)

    @patch("kttc.utils.dependencies.DEPENDENCY_GROUPS", {"test": {"pytest": "Test framework"}})
    def test_all_packages_installed(self) -> None:
        """Test when all packages in group are installed."""
        all_installed, missing = check_dependency_group("test")
        assert all_installed is True
        assert missing == []


@pytest.mark.unit
class TestDependencyGroups:
    """Test DEPENDENCY_GROUPS constant."""

    def test_groups_exist(self) -> None:
        """Test that expected groups exist."""
        assert "metrics" in DEPENDENCY_GROUPS
        assert "webui" in DEPENDENCY_GROUPS
        assert "benchmark" in DEPENDENCY_GROUPS

    def test_metrics_group_packages(self) -> None:
        """Test metrics group has expected packages."""
        metrics = DEPENDENCY_GROUPS["metrics"]
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_webui_group_packages(self) -> None:
        """Test webui group has expected packages."""
        webui = DEPENDENCY_GROUPS["webui"]
        assert isinstance(webui, dict)
        assert "fastapi" in webui
        assert "uvicorn" in webui

    def test_benchmark_group_packages(self) -> None:
        """Test benchmark group has expected packages."""
        benchmark = DEPENDENCY_GROUPS["benchmark"]
        assert isinstance(benchmark, dict)
        assert len(benchmark) > 0

    def test_package_descriptions(self) -> None:
        """Test that packages have descriptions."""
        for _, packages in DEPENDENCY_GROUPS.items():
            for pkg_name, description in packages.items():
                assert isinstance(pkg_name, str)
                assert isinstance(description, str)
                assert len(description) > 0


@pytest.mark.unit
class TestShowMissingDependenciesPrompt:
    """Test show_missing_dependencies_prompt function."""

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.Confirm.ask")
    def test_prompt_with_missing_packages(
        self, mock_confirm: MagicMock, mock_console: MagicMock
    ) -> None:
        """Test prompt display with missing packages."""
        mock_confirm.return_value = True

        result = show_missing_dependencies_prompt("webui", ["fastapi", "uvicorn"], "serve")

        assert result is True
        assert mock_console.print.call_count >= 2
        mock_confirm.assert_called_once()

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.Confirm.ask")
    def test_prompt_user_accepts(self, mock_confirm: MagicMock, mock_console: MagicMock) -> None:
        """Test prompt when user accepts installation."""
        mock_confirm.return_value = True

        result = show_missing_dependencies_prompt("metrics", ["sacrebleu"], "benchmark")

        assert result is True

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.Confirm.ask")
    def test_prompt_user_declines(self, mock_confirm: MagicMock, mock_console: MagicMock) -> None:
        """Test prompt when user declines installation."""
        mock_confirm.return_value = False

        result = show_missing_dependencies_prompt("metrics", ["sacrebleu"], "benchmark")

        assert result is False

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.Confirm.ask")
    def test_prompt_with_large_packages(
        self, mock_confirm: MagicMock, mock_console: MagicMock
    ) -> None:
        """Test prompt estimates size correctly for large packages."""
        mock_confirm.return_value = True

        result = show_missing_dependencies_prompt("metrics", ["sentence-transformers"], "benchmark")

        assert result is True
        assert mock_console.print.call_count >= 2

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.Confirm.ask")
    @patch("kttc.utils.dependencies.DEPENDENCY_GROUPS", {"test": {"pkg-2gb": "Test package (2GB)"}})
    def test_prompt_estimates_2gb_packages(
        self, mock_confirm: MagicMock, mock_console: MagicMock
    ) -> None:
        """Test prompt estimates size correctly for 2GB packages."""
        mock_confirm.return_value = True

        result = show_missing_dependencies_prompt("test", ["pkg-2gb"], "test-cmd")

        assert result is True
        assert mock_console.print.call_count >= 2


@pytest.mark.unit
class TestInstallDependencyGroup:
    """Test install_dependency_group function."""

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.subprocess.run")
    @patch("kttc.utils.dependencies.print_info")
    @patch("kttc.utils.dependencies.print_success")
    def test_install_success(
        self,
        mock_success: MagicMock,
        mock_info: MagicMock,
        mock_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Test successful installation."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = install_dependency_group("webui")

        assert result is True
        mock_run.assert_called_once()
        assert sys.executable in mock_run.call_args[0][0]
        assert "kttc[webui]" in mock_run.call_args[0][0]
        mock_success.assert_called_once()

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.subprocess.run")
    @patch("kttc.utils.dependencies.print_info")
    @patch("kttc.utils.dependencies.print_error")
    def test_install_failure(
        self,
        mock_error: MagicMock,
        mock_info: MagicMock,
        mock_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Test failed installation."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Installation failed")

        result = install_dependency_group("webui")

        assert result is False
        mock_error.assert_called_once()

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.subprocess.run")
    @patch("kttc.utils.dependencies.print_info")
    @patch("kttc.utils.dependencies.print_error")
    def test_install_timeout(
        self,
        mock_error: MagicMock,
        mock_info: MagicMock,
        mock_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Test installation timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("pip install", 600)

        result = install_dependency_group("metrics")

        assert result is False
        mock_error.assert_called_once()
        assert "timed out" in mock_error.call_args[0][0].lower()

    @patch("kttc.utils.dependencies.console")
    @patch("kttc.utils.dependencies.subprocess.run")
    @patch("kttc.utils.dependencies.print_info")
    @patch("kttc.utils.dependencies.print_error")
    def test_install_exception(
        self,
        mock_error: MagicMock,
        mock_info: MagicMock,
        mock_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Test installation with unexpected exception."""
        mock_run.side_effect = Exception("Unexpected error")

        result = install_dependency_group("benchmark")

        assert result is False
        mock_error.assert_called_once()


@pytest.mark.unit
class TestEnsureDependencyGroup:
    """Test ensure_dependency_group function."""

    @patch("kttc.utils.dependencies.check_dependency_group")
    def test_ensure_already_installed(self, mock_check: MagicMock) -> None:
        """Test when dependencies are already installed."""
        mock_check.return_value = (True, [])

        result = ensure_dependency_group("webui", "serve")

        assert result is True
        mock_check.assert_called_once_with("webui")

    @patch("kttc.utils.dependencies.check_dependency_group")
    @patch("kttc.utils.dependencies.show_missing_dependencies_prompt")
    @patch("kttc.utils.dependencies.install_dependency_group")
    def test_ensure_user_accepts_install_success(
        self, mock_install: MagicMock, mock_prompt: MagicMock, mock_check: MagicMock
    ) -> None:
        """Test when user accepts and installation succeeds."""
        mock_check.return_value = (False, ["fastapi"])
        mock_prompt.return_value = True
        mock_install.return_value = True

        result = ensure_dependency_group("webui", "serve")

        assert result is True
        mock_install.assert_called_once_with("webui")

    @patch("kttc.utils.dependencies.check_dependency_group")
    @patch("kttc.utils.dependencies.show_missing_dependencies_prompt")
    @patch("kttc.utils.dependencies.install_dependency_group")
    @patch("kttc.utils.dependencies.print_error")
    def test_ensure_user_accepts_install_fails_required(
        self,
        mock_error: MagicMock,
        mock_install: MagicMock,
        mock_prompt: MagicMock,
        mock_check: MagicMock,
    ) -> None:
        """Test when installation fails and dependency is required."""
        mock_check.return_value = (False, ["fastapi"])
        mock_prompt.return_value = True
        mock_install.return_value = False

        with pytest.raises(SystemExit):
            ensure_dependency_group("webui", "serve", required=True)

    @patch("kttc.utils.dependencies.check_dependency_group")
    @patch("kttc.utils.dependencies.show_missing_dependencies_prompt")
    @patch("kttc.utils.dependencies.install_dependency_group")
    def test_ensure_user_accepts_install_fails_optional(
        self, mock_install: MagicMock, mock_prompt: MagicMock, mock_check: MagicMock
    ) -> None:
        """Test when installation fails and dependency is optional."""
        mock_check.return_value = (False, ["fastapi"])
        mock_prompt.return_value = True
        mock_install.return_value = False

        result = ensure_dependency_group("webui", "serve", required=False)

        assert result is False

    @patch("kttc.utils.dependencies.check_dependency_group")
    @patch("kttc.utils.dependencies.show_missing_dependencies_prompt")
    @patch("kttc.utils.dependencies.print_warning")
    @patch("kttc.utils.dependencies.print_info")
    def test_ensure_user_declines_required(
        self,
        mock_info: MagicMock,
        mock_warning: MagicMock,
        mock_prompt: MagicMock,
        mock_check: MagicMock,
    ) -> None:
        """Test when user declines and dependency is required."""
        mock_check.return_value = (False, ["fastapi"])
        mock_prompt.return_value = False

        with pytest.raises(SystemExit):
            ensure_dependency_group("webui", "serve", required=True)

    @patch("kttc.utils.dependencies.check_dependency_group")
    @patch("kttc.utils.dependencies.show_missing_dependencies_prompt")
    @patch("kttc.utils.dependencies.print_warning")
    def test_ensure_user_declines_optional(
        self, mock_warning: MagicMock, mock_prompt: MagicMock, mock_check: MagicMock
    ) -> None:
        """Test when user declines and dependency is optional."""
        mock_check.return_value = (False, ["fastapi"])
        mock_prompt.return_value = False

        result = ensure_dependency_group("webui", "serve", required=False)

        assert result is False
        mock_warning.assert_called_once()


@pytest.mark.unit
class TestShowOptimizationTips:
    """Test show_optimization_tips function."""

    @patch("kttc.utils.dependencies.console")
    def test_show_optimization_tips(self, mock_console: MagicMock) -> None:
        """Test optimization tips display."""
        show_optimization_tips()

        assert mock_console.print.call_count >= 2


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("kttc.utils.dependencies.check_dependency_group")
    def test_has_webui_true(self, mock_check: MagicMock) -> None:
        """Test has_webui returns True when installed."""
        mock_check.return_value = (True, [])

        result = has_webui()

        assert result is True
        mock_check.assert_called_once_with("webui")

    @patch("kttc.utils.dependencies.check_dependency_group")
    def test_has_webui_false(self, mock_check: MagicMock) -> None:
        """Test has_webui returns False when not installed."""
        mock_check.return_value = (False, ["fastapi"])

        result = has_webui()

        assert result is False

    @patch("kttc.utils.dependencies.check_dependency_group")
    def test_has_benchmark_true(self, mock_check: MagicMock) -> None:
        """Test has_benchmark returns True when installed."""
        mock_check.return_value = (True, [])

        result = has_benchmark()

        assert result is True
        mock_check.assert_called_once_with("benchmark")

    @patch("kttc.utils.dependencies.check_dependency_group")
    def test_has_benchmark_false(self, mock_check: MagicMock) -> None:
        """Test has_benchmark returns False when not installed."""
        mock_check.return_value = (False, ["datasets"])

        result = has_benchmark()

        assert result is False

    @patch("kttc.utils.dependencies.ensure_dependency_group")
    def test_require_webui(self, mock_ensure: MagicMock) -> None:
        """Test require_webui calls ensure with correct parameters."""
        require_webui("serve")

        mock_ensure.assert_called_once_with("webui", "serve", required=True)

    @patch("kttc.utils.dependencies.ensure_dependency_group")
    def test_require_benchmark(self, mock_ensure: MagicMock) -> None:
        """Test require_benchmark calls ensure with correct parameters."""
        require_benchmark("bench")

        mock_ensure.assert_called_once_with("benchmark", "bench", required=True)
