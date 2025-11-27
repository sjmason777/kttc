"""Tests for CLI commands terminology module.

This module tests the CLI commands in kttc/cli/commands/terminology.py.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from kttc.cli.commands.terminology import (
    _display_cases_glossary,
    _display_classifiers_glossary,
    _display_generic_glossary,
    _display_mqm_glossary,
    _search_in_dict,
    terminology_app,
)


@pytest.mark.unit
class TestListGlossariesCommand:
    """Tests for list glossaries command."""

    def test_list_glossaries_success(self) -> None:
        """Test listing glossaries successfully."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available_glossaries.return_value = {
                "en": ["mqm_core", "nlp_terms"],
                "ru": ["morphology_ru"],
            }
            mock_instance.get_metadata.return_value = Mock(total_terms=10)
            mock_instance.glossaries_dir = Path("/fake/path")
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["list"])

        # Assert
        assert result.exit_code == 0
        assert "Linguistic Reference Glossaries" in result.stdout

    def test_list_glossaries_filtered_by_language(self) -> None:
        """Test listing glossaries filtered by language."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available_glossaries.return_value = {
                "en": ["mqm_core"],
                "ru": ["morphology_ru"],
            }
            mock_instance.get_metadata.return_value = Mock(total_terms=5)
            mock_instance.glossaries_dir = Path("/fake/path")
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["list", "--lang", "en"])

        # Assert
        assert result.exit_code == 0

    def test_list_glossaries_no_glossaries_found(self) -> None:
        """Test listing when no glossaries are found."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available_glossaries.return_value = {}
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["list"])

        # Assert
        assert result.exit_code == 0
        assert "No terminology glossaries found" in result.stdout

    def test_list_glossaries_language_not_found(self) -> None:
        """Test listing with language that has no glossaries."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available_glossaries.return_value = {"en": ["mqm_core"]}
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["list", "--lang", "xx"])

        # Assert
        assert result.exit_code == 0
        assert "No glossaries found for language 'xx'" in result.stdout

    def test_list_glossaries_handles_exception(self) -> None:
        """Test error handling when exception occurs."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_manager.return_value.list_available_glossaries.side_effect = Exception(
                "Test error"
            )

            result = runner.invoke(terminology_app, ["list"])

        # Assert
        assert result.exit_code == 1
        assert "Error" in result.stdout


@pytest.mark.unit
class TestShowGlossaryCommand:
    """Tests for show glossary command."""

    def test_show_glossary_mqm_format(self) -> None:
        """Test showing MQM glossary in table format."""
        # Arrange
        runner = CliRunner()
        glossary_data = {
            "error_types": {
                "mistranslation": {"definition": "Wrong translation", "severity": "major"}
            }
        }

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.load_glossary.return_value = glossary_data
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["show", "en", "mqm_core"])

        # Assert
        assert result.exit_code == 0
        assert "Terminology Glossary" in result.stdout

    def test_show_glossary_json_format(self) -> None:
        """Test showing glossary in JSON format."""
        # Arrange
        runner = CliRunner()
        glossary_data = {"test": "data"}

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.load_glossary.return_value = glossary_data
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["show", "en", "test", "--format", "json"])

        # Assert
        assert result.exit_code == 0

    def test_show_glossary_not_found(self) -> None:
        """Test showing non-existent glossary."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.load_glossary.return_value = None
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["show", "xx", "nonexistent"])

        # Assert
        assert result.exit_code == 0
        assert "No glossary found" in result.stdout


@pytest.mark.unit
class TestSearchGlossariesCommand:
    """Tests for search glossaries command."""

    def test_search_glossaries_success(self) -> None:
        """Test searching glossaries successfully."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available_glossaries.return_value = {"en": ["mqm_core"]}
            mock_instance.search_terms.return_value = [
                {"path": "mqm_core > accuracy", "data": {"definition": "test"}}
            ]
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["search", "accuracy"])

        # Assert
        assert result.exit_code == 0
        assert "Search Results" in result.stdout

    def test_search_glossaries_no_results(self) -> None:
        """Test searching with no results."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available_glossaries.return_value = {"en": ["mqm_core"]}
            mock_instance.search_terms.return_value = []
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["search", "nonexistent"])

        # Assert
        assert result.exit_code == 0
        assert "No results found" in result.stdout

    def test_search_glossaries_filtered_by_language(self) -> None:
        """Test searching filtered by language."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.GlossaryManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.list_available_glossaries.return_value = {"en": ["mqm_core"]}
            mock_instance.search_terms.return_value = []
            mock_manager.return_value = mock_instance

            result = runner.invoke(terminology_app, ["search", "test", "--lang", "en"])

        # Assert
        assert result.exit_code == 0


@pytest.mark.unit
class TestValidateErrorCommand:
    """Tests for validate error type command."""

    def test_validate_error_type_valid(self) -> None:
        """Test validating a valid error type."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.TermValidator") as mock_validator:
            mock_instance = Mock()
            mock_instance.validate_mqm_error_type.return_value = (
                True,
                {"definition": "Test definition", "severity": "major"},
            )
            mock_validator.return_value = mock_instance

            result = runner.invoke(terminology_app, ["validate-error", "mistranslation"])

        # Assert
        assert result.exit_code == 0
        assert "Valid MQM Error Type" in result.stdout

    def test_validate_error_type_invalid(self) -> None:
        """Test validating an invalid error type."""
        # Arrange
        runner = CliRunner()

        # Act
        with patch("kttc.cli.commands.terminology.TermValidator") as mock_validator:
            mock_instance = Mock()
            mock_instance.validate_mqm_error_type.return_value = (False, None)
            mock_validator.return_value = mock_instance

            result = runner.invoke(terminology_app, ["validate-error", "invalid_error"])

        # Assert
        assert result.exit_code == 0
        assert "Invalid MQM Error Type" in result.stdout


@pytest.mark.unit
class TestListValidatorsCommand:
    """Tests for list validators command."""

    def test_list_validators(self) -> None:
        """Test listing available validators."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(terminology_app, ["validators"])

        # Assert
        assert result.exit_code == 0
        assert "Available Language Validators" in result.stdout
        assert "Russian" in result.stdout
        assert "Chinese" in result.stdout


@pytest.mark.unit
class TestDisplayHelperFunctions:
    """Tests for display helper functions."""

    def test_display_mqm_glossary(self) -> None:
        """Test MQM glossary display."""
        # Arrange
        glossary_data = {
            "error_types": {"accuracy": {"definition": "Accuracy error", "severity": "major"}}
        }

        # Act & Assert - should not raise
        with patch("kttc.cli.commands.terminology.console"):
            _display_mqm_glossary(glossary_data, limit=10)

    def test_display_cases_glossary(self) -> None:
        """Test cases glossary display."""
        # Arrange
        glossary_data = {"cases": {"nominative": {"endings": ["-", "-а"], "function": "Subject"}}}

        # Act & Assert - should not raise
        with patch("kttc.cli.commands.terminology.console"):
            _display_cases_glossary(glossary_data, "cases", "Test Cases")

    def test_display_classifiers_glossary(self) -> None:
        """Test classifiers glossary display."""
        # Arrange
        glossary_data = {
            "classifiers": {"个": {"pinyin": "gè", "category": "general", "usage": "General usage"}}
        }

        # Act & Assert - should not raise
        with patch("kttc.cli.commands.terminology.console"):
            _display_classifiers_glossary(glossary_data, limit=10)

    def test_display_generic_glossary(self) -> None:
        """Test generic glossary display."""
        # Arrange
        glossary_data = {"key1": "value1", "key2": {"nested": "data"}}

        # Act & Assert - should not raise
        with patch("kttc.cli.commands.terminology.console"):
            _display_generic_glossary(glossary_data, limit=10)


@pytest.mark.unit
class TestSearchInDict:
    """Tests for _search_in_dict helper function."""

    def test_search_in_dict_case_sensitive(self) -> None:
        """Test case-sensitive search in dictionary."""
        # Arrange
        data = {"Test": "value", "nested": {"Data": "found"}}

        # Act
        results = _search_in_dict(data, "Test", case_sensitive=True)

        # Assert
        assert len(results) > 0
        assert any("Test" in r[0] for r in results)

    def test_search_in_dict_case_insensitive(self) -> None:
        """Test case-insensitive search in dictionary."""
        # Arrange
        data = {"Test": "value", "nested": {"data": "found"}}

        # Act
        results = _search_in_dict(data, "test", case_sensitive=False)

        # Assert
        assert len(results) > 0

    def test_search_in_dict_nested(self) -> None:
        """Test search in nested dictionary."""
        # Arrange
        data = {"level1": {"level2": {"target": "found"}}}

        # Act
        results = _search_in_dict(data, "target", case_sensitive=False)

        # Assert
        assert len(results) > 0
        assert any("level1.level2.target" in r[0] for r in results)

    def test_search_in_dict_list(self) -> None:
        """Test search in list within dictionary."""
        # Arrange
        data = {"items": ["test", "data", "target"]}

        # Act
        results = _search_in_dict(data, "target", case_sensitive=False)

        # Assert
        assert len(results) > 0

    def test_search_in_dict_no_match(self) -> None:
        """Test search with no matches."""
        # Arrange
        data = {"key": "value"}

        # Act
        results = _search_in_dict(data, "nonexistent", case_sensitive=False)

        # Assert
        assert len(results) == 0
