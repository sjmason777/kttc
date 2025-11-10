"""Unit tests for ErrorParser."""

import pytest

from kttc.agents import ErrorParser
from kttc.core import ErrorSeverity


class TestErrorParser:
    """Test ErrorParser for extracting errors from LLM responses."""

    def test_parse_single_error(self) -> None:
        """Test parsing a single error block."""
        response = """
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Incorrect translation of 'hello'
SUGGESTION: Use 'hola' instead
ERROR_END
"""
        errors = ErrorParser.parse_errors(response)
        assert len(errors) == 1

        error = errors[0]
        assert error.category == "accuracy"
        assert error.subcategory == "mistranslation"
        assert error.severity == ErrorSeverity.MAJOR
        assert error.location == (0, 5)
        assert "Incorrect translation" in error.description
        assert error.suggestion == "Use 'hola' instead"

    def test_parse_multiple_errors(self) -> None:
        """Test parsing multiple error blocks."""
        response = """
Some text before errors.

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: omission
SEVERITY: critical
LOCATION: 10-15
DESCRIPTION: Missing word
ERROR_END

More text in between.

ERROR_START
CATEGORY: accuracy
SUBCATEGORY: addition
SEVERITY: minor
LOCATION: 20-25
DESCRIPTION: Extra word added
SUGGESTION: Remove it
ERROR_END
"""
        errors = ErrorParser.parse_errors(response)
        assert len(errors) == 2

        assert errors[0].subcategory == "omission"
        assert errors[0].severity == ErrorSeverity.CRITICAL
        assert errors[1].subcategory == "addition"
        assert errors[1].severity == ErrorSeverity.MINOR

    def test_parse_no_errors(self) -> None:
        """Test parsing response with no errors."""
        response = "The translation is perfect. No errors found."
        errors = ErrorParser.parse_errors(response)
        assert len(errors) == 0

    def test_parse_location_formats(self) -> None:
        """Test different location format variations."""
        test_cases = [
            ("0-5", (0, 5)),
            ("0,5", (0, 5)),
            ("[0, 5]", (0, 5)),
            ("(0, 5)", (0, 5)),
            ("0 - 5", (0, 5)),
        ]

        for location_str, expected in test_cases:
            response = f"""
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: test
SEVERITY: minor
LOCATION: {location_str}
DESCRIPTION: Test error
ERROR_END
"""
            errors = ErrorParser.parse_errors(response)
            assert len(errors) == 1
            assert errors[0].location == expected

    def test_parse_all_severity_levels(self) -> None:
        """Test parsing all severity levels."""
        severities = [
            ("neutral", ErrorSeverity.NEUTRAL),
            ("minor", ErrorSeverity.MINOR),
            ("major", ErrorSeverity.MAJOR),
            ("critical", ErrorSeverity.CRITICAL),
        ]

        for severity_str, expected_severity in severities:
            response = f"""
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: test
SEVERITY: {severity_str}
LOCATION: 0-5
DESCRIPTION: Test error
ERROR_END
"""
            errors = ErrorParser.parse_errors(response)
            assert len(errors) == 1
            assert errors[0].severity == expected_severity

    def test_parse_case_insensitive(self) -> None:
        """Test that parsing is case-insensitive."""
        response = """
error_start
CATEGORY: ACCURACY
SUBCATEGORY: Mistranslation
severity: MAJOR
location: 0-5
Description: Test error
error_end
"""
        errors = ErrorParser.parse_errors(response)
        assert len(errors) == 1
        assert errors[0].category == "ACCURACY"
        assert errors[0].severity == ErrorSeverity.MAJOR

    def test_parse_without_suggestion(self) -> None:
        """Test parsing error without optional suggestion field."""
        response = """
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Incorrect translation
ERROR_END
"""
        errors = ErrorParser.parse_errors(response)
        assert len(errors) == 1
        assert errors[0].suggestion is None

    def test_parse_missing_required_field(self) -> None:
        """Test that missing required fields are handled gracefully."""
        response = """
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
DESCRIPTION: Missing location field
ERROR_END
"""
        # Parser should skip malformed errors and return empty list
        errors = ErrorParser.parse_errors(response)
        assert len(errors) == 0

    def test_parse_invalid_severity(self) -> None:
        """Test handling of invalid severity values."""
        response = """
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: test
SEVERITY: invalid_severity
LOCATION: 0-5
DESCRIPTION: Test error
ERROR_END
"""
        errors = ErrorParser.parse_errors(response)
        # Should skip invalid error
        assert len(errors) == 0

    def test_parse_invalid_location_format(self) -> None:
        """Test handling of invalid location formats."""
        invalid_locations = ["invalid", "0", "abc-def", "-5-10"]

        for invalid_location in invalid_locations:
            response = f"""
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: test
SEVERITY: minor
LOCATION: {invalid_location}
DESCRIPTION: Test error
ERROR_END
"""
            errors = ErrorParser.parse_errors(response)
            # Should skip errors with invalid locations
            assert len(errors) == 0

    def test_parse_multiline_description(self) -> None:
        """Test parsing errors with multiline descriptions."""
        response = """
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: This is a long description
that spans multiple lines
and contains detailed information
SUGGESTION: Fix it properly
ERROR_END
"""
        errors = ErrorParser.parse_errors(response)
        assert len(errors) == 1
        # Description should only capture first line (due to MULTILINE regex)
        assert "This is a long description" in errors[0].description

    def test_location_parser_edge_cases(self) -> None:
        """Test edge cases in location parsing."""
        parser = ErrorParser()

        # Valid ranges
        assert parser._parse_location("0-100") == (0, 100)
        assert parser._parse_location("5,10") == (5, 10)

        # Invalid ranges should raise ValueError
        with pytest.raises(ValueError):
            parser._parse_location("10-5")  # end < start

        with pytest.raises(ValueError):
            parser._parse_location("-5-10")  # negative start

        with pytest.raises(ValueError):
            parser._parse_location("invalid")  # no separator

    def test_severity_parser_edge_cases(self) -> None:
        """Test edge cases in severity parsing."""
        parser = ErrorParser()

        # Valid severities (case-insensitive)
        assert parser._parse_severity("MAJOR") == ErrorSeverity.MAJOR
        assert parser._parse_severity("critical") == ErrorSeverity.CRITICAL
        assert parser._parse_severity("  minor  ") == ErrorSeverity.MINOR

        # Invalid severity should raise ValueError
        with pytest.raises(ValueError, match="Unknown severity"):
            parser._parse_severity("invalid_severity")
