# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ErrorParser glossary enrichment."""


from kttc.agents.parser import ErrorParser


class TestErrorParserGlossaryEnrichment:
    """Test ErrorParser integration with glossary-based enrichment."""

    def test_parse_errors_basic(self):
        """Test basic error parsing without enrichment."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Wrong meaning
ERROR_END"""

        errors = ErrorParser.parse_errors(response, enrich_with_glossary=False)

        assert len(errors) == 1
        assert errors[0].category == "accuracy"
        assert errors[0].subcategory == "mistranslation"
        assert errors[0].severity.value == "major"
        assert errors[0].location == (0, 10)
        assert errors[0].description == "Wrong meaning"

    def test_parse_errors_with_enrichment(self):
        """Test error parsing with glossary enrichment."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Wrong meaning
ERROR_END"""

        errors = ErrorParser.parse_errors(response, enrich_with_glossary=True, language="en")

        assert len(errors) == 1
        assert errors[0].category == "accuracy"
        assert errors[0].subcategory == "mistranslation"

        # Description may be enriched with MQM definition
        # Should at minimum contain original description
        assert "Wrong meaning" in errors[0].description

    def test_parse_multiple_errors_with_enrichment(self):
        """Test parsing multiple errors with enrichment."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Wrong meaning
ERROR_END

ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: minor
LOCATION: 15-20
DESCRIPTION: Grammar issue
ERROR_END"""

        errors = ErrorParser.parse_errors(response, enrich_with_glossary=True, language="en")

        assert len(errors) == 2
        assert errors[0].category == "accuracy"
        assert errors[1].category == "fluency"

        # Both should contain original descriptions
        assert "Wrong meaning" in errors[0].description
        assert "Grammar issue" in errors[1].description

    def test_enrichment_disabled(self):
        """Test that enrichment can be explicitly disabled."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Wrong meaning
ERROR_END"""

        errors = ErrorParser.parse_errors(response, enrich_with_glossary=False)

        assert len(errors) == 1
        # Description should be exactly as provided, no enrichment
        assert errors[0].description == "Wrong meaning"

    def test_enrichment_with_different_languages(self):
        """Test enrichment works with different language codes."""
        response = """ERROR_START
CATEGORY: fluency
SUBCATEGORY: grammar
SEVERITY: minor
LOCATION: 0-5
DESCRIPTION: Test error
ERROR_END"""

        # Test with different language codes
        for lang in ["en", "ru", "zh", "hi", "fa"]:
            errors = ErrorParser.parse_errors(response, enrich_with_glossary=True, language=lang)

            assert len(errors) == 1
            assert "Test error" in errors[0].description

    def test_enrichment_preserves_all_error_fields(self):
        """Test that enrichment preserves all original error fields."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: omission
SEVERITY: critical
LOCATION: 5-15
DESCRIPTION: Missing content
SUGGESTION: Add translated text
ERROR_END"""

        errors = ErrorParser.parse_errors(response, enrich_with_glossary=True)

        assert len(errors) == 1
        error = errors[0]

        # All fields should be preserved
        assert error.category == "accuracy"
        assert error.subcategory == "omission"
        assert error.severity.value == "critical"
        assert error.location == (5, 15)
        assert "Missing content" in error.description
        assert error.suggestion == "Add translated text"

    def test_enrichment_handles_empty_response(self):
        """Test enrichment handles empty/no error responses gracefully."""
        response = "No errors found."

        errors = ErrorParser.parse_errors(response, enrich_with_glossary=True)

        assert len(errors) == 0

    def test_enrichment_handles_invalid_error_type(self):
        """Test enrichment handles unknown/invalid error types gracefully."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: unknown_error_type_xyz
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Test error
ERROR_END"""

        # Should not raise exception, should handle gracefully
        errors = ErrorParser.parse_errors(response, enrich_with_glossary=True)

        assert len(errors) == 1
        assert errors[0].subcategory == "unknown_error_type_xyz"
        assert "Test error" in errors[0].description

    def test_enrichment_with_malformed_json_fallback(self):
        """Test that enrichment failures don't break parsing."""
        response = """ERROR_START
CATEGORY: fluency
SUBCATEGORY: spelling
SEVERITY: minor
LOCATION: 0-5
DESCRIPTION: Typo
ERROR_END"""

        # Should parse successfully even if enrichment has issues
        errors = ErrorParser.parse_errors(
            response, enrich_with_glossary=True, language="invalid_lang"
        )

        assert len(errors) == 1
        assert errors[0].category == "fluency"

    def test_parse_errors_with_all_severity_levels(self):
        """Test enrichment works with all severity levels."""
        severities = ["neutral", "minor", "major", "critical"]

        for severity in severities:
            response = f"""ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: {severity}
LOCATION: 0-10
DESCRIPTION: Test error
ERROR_END"""

            errors = ErrorParser.parse_errors(response, enrich_with_glossary=True)

            assert len(errors) == 1
            assert errors[0].severity.value == severity

    def test_enrichment_idempotent(self):
        """Test that enrichment is idempotent (doesn't double-enrich)."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Wrong meaning
ERROR_END"""

        errors1 = ErrorParser.parse_errors(response, enrich_with_glossary=True)
        errors2 = ErrorParser.parse_errors(response, enrich_with_glossary=True)

        assert len(errors1) == 1
        assert len(errors2) == 1

        # Should get same enrichment both times
        assert errors1[0].description == errors2[0].description

    def test_enrichment_with_complex_descriptions(self):
        """Test enrichment handles complex multi-line descriptions."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Wrong meaning: 'cat' translated as 'dog'. This changes the entire meaning of the sentence.
ERROR_END"""

        errors = ErrorParser.parse_errors(response, enrich_with_glossary=True)

        assert len(errors) == 1
        # Original description should be preserved
        assert "Wrong meaning" in errors[0].description
        assert "cat" in errors[0].description

    def test_parse_errors_default_enrichment_enabled(self):
        """Test that enrichment is enabled by default."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Test
ERROR_END"""

        # Call without specifying enrich_with_glossary (should default to True)
        errors = ErrorParser.parse_errors(response)

        assert len(errors) == 1
        # Should have been enriched (by default)

    def test_parse_errors_default_language_english(self):
        """Test that default language is English."""
        response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-10
DESCRIPTION: Test
ERROR_END"""

        # Call without specifying language (should default to "en")
        errors = ErrorParser.parse_errors(response, enrich_with_glossary=True)

        assert len(errors) == 1
