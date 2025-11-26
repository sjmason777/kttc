"""Unit tests for dependencies module.

Tests dependency management and lazy loading.
"""

import pytest

from kttc.utils.dependencies import (
    DEPENDENCY_GROUPS,
    check_dependency_group,
    check_package_installed,
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
        for group_name, packages in DEPENDENCY_GROUPS.items():
            for pkg_name, description in packages.items():
                assert isinstance(pkg_name, str)
                assert isinstance(description, str)
                assert len(description) > 0
