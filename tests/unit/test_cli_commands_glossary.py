"""Tests for CLI commands glossary module.

This module tests the CLI commands in kttc/cli/commands/glossary.py.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from kttc.cli.commands.glossary import (
    _build_entry_row,
    _filter_by_lang_pair,
    _find_duplicate_entries,
    _find_empty_entries,
    glossary_app,
)


@pytest.mark.unit
class TestListGlossariesCommand:
    """Tests for list glossaries command."""

    def test_list_glossaries_success(self) -> None:
        """Test listing glossaries successfully."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available.return_value = [
                ("medical", Path("/path/medical.json"), 10),
                ("technical", Path("/path/technical.json"), 20),
            ]
            mock_manager.return_value = mock_instance

            result = runner.invoke(glossary_app, ["list"])

        # Assert
        assert result.exit_code == 0
        assert "Available Glossaries" in result.stdout

    def test_list_glossaries_empty(self) -> None:
        """Test listing when no glossaries exist."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available.return_value = []
            mock_manager.return_value = mock_instance

            result = runner.invoke(glossary_app, ["list"])

        # Assert
        assert result.exit_code == 0
        assert "No glossaries found" in result.stdout
        assert "kttc glossary create" in result.stdout

    def test_list_glossaries_error(self) -> None:
        """Test error handling in list command."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_manager.return_value.list_available.side_effect = Exception("Test error")

            result = runner.invoke(glossary_app, ["list"])

        # Assert
        assert result.exit_code == 1
        assert "Error" in result.stdout


@pytest.mark.unit
class TestShowGlossaryCommand:
    """Tests for show glossary command."""

    def test_show_glossary_success(self) -> None:
        """Test showing glossary successfully."""
        # Arrange
        runner = CliRunner()
        mock_glossary = Mock()
        mock_glossary.metadata = Mock(version="1.0", description="Test glossary")
        mock_entry = Mock(
            source="API",
            target="API",
            source_lang="en",
            target_lang="ru",
            domain="tech",
            notes="",
            do_not_translate=False,
            case_sensitive=False,
        )
        mock_glossary.entries = [mock_entry]

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.load_glossary.return_value = mock_glossary
            mock_manager.return_value = mock_instance

            result = runner.invoke(glossary_app, ["show", "test"])

        # Assert
        assert result.exit_code == 0
        assert "Glossary: test" in result.stdout

    def test_show_glossary_with_limit(self) -> None:
        """Test showing glossary with entry limit."""
        # Arrange
        runner = CliRunner()
        mock_glossary = Mock()
        mock_glossary.metadata = None
        mock_glossary.entries = [
            Mock(
                source=f"term{i}",
                target=f"term{i}",
                source_lang="en",
                target_lang="ru",
                domain=None,
                notes=None,
                do_not_translate=False,
                case_sensitive=False,
            )
            for i in range(100)
        ]

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.load_glossary.return_value = mock_glossary
            mock_manager.return_value = mock_instance

            result = runner.invoke(glossary_app, ["show", "test", "--limit", "10"])

        # Assert
        assert result.exit_code == 0
        assert "Showing first 10" in result.stdout

    def test_show_glossary_not_found(self) -> None:
        """Test showing non-existent glossary."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.load_glossary.side_effect = FileNotFoundError("Not found")
            mock_manager.return_value = mock_instance

            result = runner.invoke(glossary_app, ["show", "nonexistent"])

        # Assert
        assert result.exit_code == 1
        assert "Error" in result.stdout


@pytest.mark.unit
class TestCreateGlossaryCommand:
    """Tests for create glossary command."""

    def test_create_glossary_from_csv(self, tmp_path: Path) -> None:
        """Test creating glossary from CSV file."""
        # Arrange
        runner = CliRunner()
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "source,target,source_lang,target_lang\nAPI,API,en,ru\n", encoding="utf-8"
        )

        # Act
        with (
            patch("kttc.cli.commands.glossary.Glossary.from_csv") as mock_from_csv,
            patch("kttc.cli.commands.glossary.Path.cwd") as mock_cwd,
        ):
            mock_cwd.return_value = tmp_path
            mock_glossary = Mock()
            mock_glossary.entries = [Mock()]
            mock_from_csv.return_value = mock_glossary

            result = runner.invoke(glossary_app, ["create", "test", "--from-csv", str(csv_file)])

        # Assert
        assert result.exit_code == 0
        assert "Created glossary 'test'" in result.stdout

    def test_create_glossary_from_json(self, tmp_path: Path) -> None:
        """Test creating glossary from JSON file."""
        # Arrange
        runner = CliRunner()
        json_file = tmp_path / "test.json"
        json_file.write_text('{"entries": []}', encoding="utf-8")

        # Act
        with (
            patch("kttc.cli.commands.glossary.Glossary.from_json") as mock_from_json,
            patch("kttc.cli.commands.glossary.Path.cwd") as mock_cwd,
        ):
            mock_cwd.return_value = tmp_path
            mock_glossary = Mock()
            mock_glossary.entries = []
            mock_from_json.return_value = mock_glossary

            result = runner.invoke(glossary_app, ["create", "test", "--from-json", str(json_file)])

        # Assert
        assert result.exit_code == 0
        assert "Created glossary 'test'" in result.stdout

    def test_create_glossary_file_not_found(self) -> None:
        """Test creating glossary from non-existent file."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(
            glossary_app, ["create", "test", "--from-csv", "/nonexistent/file.csv"]
        )

        # Assert
        assert result.exit_code == 1
        assert "File not found" in result.stdout

    def test_create_glossary_no_source_specified(self) -> None:
        """Test creating glossary without specifying source."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(glossary_app, ["create", "test"])

        # Assert
        assert result.exit_code == 1
        assert "Must specify" in result.stdout


@pytest.mark.unit
class TestMergeGlossariesCommand:
    """Tests for merge glossaries command."""

    def test_merge_glossaries_success(self, tmp_path: Path) -> None:
        """Test merging glossaries successfully."""
        # Arrange
        runner = CliRunner()

        # Act
        with (
            patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager,
            patch("kttc.cli.commands.glossary.Path.cwd") as mock_cwd,
        ):
            mock_cwd.return_value = tmp_path
            mock_instance = Mock()
            mock_merged = Mock()
            mock_merged.entries = [Mock(), Mock()]
            mock_instance.merge_glossaries.return_value = mock_merged
            mock_manager.return_value = mock_instance

            result = runner.invoke(
                glossary_app, ["merge", "base", "medical", "--output", "combined"]
            )

        # Assert
        assert result.exit_code == 0
        assert "Merged 2 glossaries" in result.stdout

    def test_merge_glossaries_error(self) -> None:
        """Test error handling in merge command."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_manager.return_value.merge_glossaries.side_effect = Exception("Merge failed")

            result = runner.invoke(glossary_app, ["merge", "a", "b", "--output", "c"])

        # Assert
        assert result.exit_code == 1
        assert "Error" in result.stdout


@pytest.mark.unit
class TestExportGlossaryCommand:
    """Tests for export glossary command."""

    def test_export_glossary_to_csv(self, tmp_path: Path) -> None:
        """Test exporting glossary to CSV."""
        # Arrange
        runner = CliRunner()
        output_file = tmp_path / "output.csv"

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_glossary = Mock()
            mock_glossary.to_csv = Mock()
            mock_instance.load_glossary.return_value = mock_glossary
            mock_manager.return_value = mock_instance

            result = runner.invoke(
                glossary_app, ["export", "test", "--format", "csv", "--output", str(output_file)]
            )

        # Assert
        assert result.exit_code == 0
        assert "Exported 'test'" in result.stdout

    def test_export_glossary_to_json(self, tmp_path: Path) -> None:
        """Test exporting glossary to JSON."""
        # Arrange
        runner = CliRunner()
        output_file = tmp_path / "output.json"

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_glossary = Mock()
            mock_glossary.to_json = Mock()
            mock_instance.load_glossary.return_value = mock_glossary
            mock_manager.return_value = mock_instance

            result = runner.invoke(
                glossary_app, ["export", "test", "--format", "json", "--output", str(output_file)]
            )

        # Assert
        assert result.exit_code == 0
        assert "Exported 'test'" in result.stdout

    def test_export_glossary_unsupported_format(self) -> None:
        """Test exporting with unsupported format."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.glossary.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_glossary = Mock()
            mock_instance.load_glossary.return_value = mock_glossary
            mock_manager.return_value = mock_instance

            result = runner.invoke(glossary_app, ["export", "test", "--format", "xml"])

        # Assert
        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout


@pytest.mark.unit
class TestValidateGlossaryCommand:
    """Tests for validate glossary command."""

    def test_validate_glossary_json_success(self, tmp_path: Path) -> None:
        """Test validating valid JSON glossary."""
        # Arrange
        runner = CliRunner()
        json_file = tmp_path / "test.json"
        json_file.write_text('{"entries": []}', encoding="utf-8")

        # Act
        with patch("kttc.cli.commands.glossary.Glossary.from_json") as mock_from_json:
            mock_glossary = Mock()
            mock_glossary.entries = []
            mock_from_json.return_value = mock_glossary

            result = runner.invoke(glossary_app, ["validate", str(json_file)])

        # Assert
        assert result.exit_code == 0
        assert "Glossary is valid" in result.stdout

    def test_validate_glossary_with_duplicates(self, tmp_path: Path) -> None:
        """Test validating glossary with duplicate entries."""
        # Arrange
        runner = CliRunner()
        json_file = tmp_path / "test.json"
        json_file.write_text('{"entries": []}', encoding="utf-8")

        # Act
        with patch("kttc.cli.commands.glossary.Glossary.from_json") as mock_from_json:
            mock_entry = Mock(
                source="API", source_lang="en", target_lang="ru", target="API", notes=""
            )
            mock_glossary = Mock()
            mock_glossary.entries = [mock_entry, mock_entry]  # Duplicates
            mock_from_json.return_value = mock_glossary

            result = runner.invoke(glossary_app, ["validate", str(json_file)])

        # Assert
        assert result.exit_code == 0
        assert "issues" in result.stdout or "valid" in result.stdout

    def test_validate_glossary_unsupported_format(self, tmp_path: Path) -> None:
        """Test validating file with unsupported format."""
        # Arrange
        runner = CliRunner()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test", encoding="utf-8")

        # Act
        result = runner.invoke(glossary_app, ["validate", str(txt_file)])

        # Assert
        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout


@pytest.mark.unit
class TestHelperFunctions:
    """Tests for helper functions."""

    def test_filter_by_lang_pair_no_filter(self) -> None:
        """Test filtering without language pair."""
        # Arrange
        entries = [Mock(source_lang="en", target_lang="ru")]

        # Act
        with patch("kttc.cli.commands.glossary.console"):
            result = _filter_by_lang_pair(entries, None)

        # Assert
        assert result == entries

    def test_filter_by_lang_pair_with_filter(self) -> None:
        """Test filtering with language pair."""
        # Arrange
        entries = [
            Mock(source_lang="en", target_lang="ru"),
            Mock(source_lang="en", target_lang="es"),
        ]

        # Act
        with patch("kttc.cli.commands.glossary.console"):
            result = _filter_by_lang_pair(entries, "en-ru")

        # Assert
        assert len(result) == 1
        assert result[0].target_lang == "ru"

    def test_filter_by_lang_pair_invalid_format(self) -> None:
        """Test filtering with invalid language pair format."""
        import typer

        # Arrange
        entries = [Mock()]

        # Act & Assert
        with patch("kttc.cli.commands.glossary.console"), pytest.raises(typer.Exit):
            _filter_by_lang_pair(entries, "invalid")

    def test_build_entry_row_basic(self) -> None:
        """Test building entry row without markers."""
        # Arrange
        entry = Mock(
            source="API",
            target="API",
            source_lang="en",
            target_lang="ru",
            domain="tech",
            notes="Test note",
            do_not_translate=False,
            case_sensitive=False,
        )

        # Act
        result = _build_entry_row(entry)

        # Assert
        assert result == ("API", "API", "en→ru", "tech", "Test note")

    def test_build_entry_row_with_markers(self) -> None:
        """Test building entry row with DNT and CS markers."""
        # Arrange
        entry = Mock(
            source="API",
            target="API",
            source_lang="en",
            target_lang="ru",
            domain="tech",
            notes="Test",
            do_not_translate=True,
            case_sensitive=True,
        )

        # Act
        result = _build_entry_row(entry)

        # Assert
        assert "[red]DNT[/red]" in result[4]
        assert "[yellow]CS[/yellow]" in result[4]

    def test_find_duplicate_entries_no_duplicates(self) -> None:
        """Test finding duplicates when none exist."""
        # Arrange
        entries = [
            Mock(source="API", source_lang="en", target_lang="ru"),
            Mock(source="SDK", source_lang="en", target_lang="ru"),
        ]

        # Act
        issues = _find_duplicate_entries(entries)

        # Assert
        assert len(issues) == 0

    def test_find_duplicate_entries_with_duplicates(self) -> None:
        """Test finding duplicate entries."""
        # Arrange
        entries = [
            Mock(source="API", source_lang="en", target_lang="ru"),
            Mock(source="API", source_lang="en", target_lang="ru"),
        ]

        # Act
        issues = _find_duplicate_entries(entries)

        # Assert
        assert len(issues) > 0
        assert "duplicate" in issues[0].lower()

    def test_find_empty_entries_no_empty(self) -> None:
        """Test finding empty entries when none exist."""
        # Arrange
        entries = [Mock(source="API", target="API")]

        # Act
        issues = _find_empty_entries(entries)

        # Assert
        assert len(issues) == 0

    def test_find_empty_entries_with_empty_source(self) -> None:
        """Test finding entries with empty source."""
        # Arrange
        entries = [Mock(source="", target="API")]

        # Act
        issues = _find_empty_entries(entries)

        # Assert
        assert len(issues) > 0
        assert "empty source" in issues[0].lower()

    def test_find_empty_entries_with_empty_target(self) -> None:
        """Test finding entries with empty target."""
        # Arrange
        entries = [Mock(source="API", target="  ")]

        # Act
        issues = _find_empty_entries(entries)

        # Assert
        assert len(issues) > 0
        assert "empty target" in issues[0].lower()
