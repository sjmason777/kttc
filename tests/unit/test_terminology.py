"""Tests for terminology management module."""

import pytest

from kttc.terminology import GlossaryManager, TermValidator


class TestGlossaryManager:
    """Tests for GlossaryManager."""

    @pytest.fixture
    def manager(self):
        """Create a GlossaryManager instance."""
        return GlossaryManager()

    def test_initialization(self, manager):
        """Test that GlossaryManager initializes correctly."""
        assert manager.glossaries_dir.exists()
        assert manager.glossaries_dir.is_dir()

    def test_load_english_mqm_glossary(self, manager):
        """Test loading English MQM glossary."""
        glossary = manager.load_glossary("en", "mqm_core")

        assert "metadata" in glossary
        assert glossary["metadata"]["language"] == "en"
        assert glossary["metadata"]["glossary_type"] == "MQM Core"
        assert "error_dimensions" in glossary

    def test_load_english_nlp_glossary(self, manager):
        """Test loading English NLP glossary."""
        glossary = manager.load_glossary("en", "nlp_terms")

        assert "metadata" in glossary
        assert glossary["metadata"]["language"] == "en"
        assert "core_nlp_concepts" in glossary

    def test_load_russian_morphology(self, manager):
        """Test loading Russian morphology glossary."""
        glossary = manager.load_glossary("ru", "morphology_ru")

        assert "metadata" in glossary
        assert glossary["metadata"]["language"] == "ru"
        assert "grammatical_categories" in glossary

    def test_get_mqm_error_dimensions(self, manager):
        """Test getting MQM error dimensions."""
        dimensions = manager.get_mqm_error_dimensions("en")

        assert isinstance(dimensions, list)
        assert len(dimensions) > 0

        # Check for expected dimensions
        dimension_ids = [d["id"] for d in dimensions]
        assert "terminology" in dimension_ids
        assert "accuracy" in dimension_ids
        assert "linguistic_conventions" in dimension_ids

    def test_get_severity_levels(self, manager):
        """Test getting severity levels."""
        levels = manager.get_severity_levels("en")

        assert isinstance(levels, dict)
        assert "minor" in levels
        assert "major" in levels
        assert "critical" in levels

        # Check penalty multipliers
        assert levels["minor"]["penalty_multiplier"] == 1
        assert levels["major"]["penalty_multiplier"] == 5
        assert levels["critical"]["penalty_multiplier"] == 10

    def test_list_available_glossaries(self, manager):
        """Test listing available glossaries."""
        available = manager.list_available_glossaries()

        assert isinstance(available, dict)
        assert "en" in available
        assert "ru" in available

        # Check English glossaries
        en_glossaries = available["en"]
        assert "mqm_core" in en_glossaries
        assert "nlp_terms" in en_glossaries
        assert "grammar_advanced" in en_glossaries

    def test_search_terms(self, manager):
        """Test searching for terms."""
        results = manager.search_terms("accuracy", language="en")

        assert isinstance(results, list)
        assert len(results) > 0

    def test_get_metadata(self, manager):
        """Test getting glossary metadata."""
        # Load glossary first
        manager.load_glossary("en", "mqm_core")

        metadata = manager.get_metadata("en", "mqm_core")

        assert metadata is not None
        assert metadata.language == "en"
        assert metadata.glossary_type == "MQM Core"
        assert metadata.version == "1.0.0"

    def test_load_nonexistent_glossary(self, manager):
        """Test loading a glossary that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            manager.load_glossary("xx", "nonexistent")


class TestTermValidator:
    """Tests for TermValidator."""

    @pytest.fixture
    def validator(self):
        """Create a TermValidator instance."""
        return TermValidator()

    def test_initialization(self, validator):
        """Test that TermValidator initializes correctly."""
        assert validator.glossary_manager is not None

    def test_validate_mqm_error_type(self, validator):
        """Test validating MQM error types."""
        # Test valid error types
        is_valid, info = validator.validate_mqm_error_type("terminology", "en")
        assert is_valid
        assert info is not None

        is_valid, info = validator.validate_mqm_error_type("accuracy", "en")
        assert is_valid

        # Test invalid error type
        is_valid, info = validator.validate_mqm_error_type("nonexistent_error", "en")
        assert not is_valid
        assert info is None

    def test_get_severity_multiplier(self, validator):
        """Test getting severity multipliers."""
        assert validator.get_severity_multiplier("minor", "en") == 1.0
        assert validator.get_severity_multiplier("major", "en") == 5.0
        assert validator.get_severity_multiplier("critical", "en") == 10.0
        assert validator.get_severity_multiplier("neutral", "en") == 0.0

    def test_validate_terminology_consistency(self, validator):
        """Test terminology consistency validation."""
        # Consistent translations - no errors
        errors = validator.validate_terminology_consistency(
            source_terms=["API", "API"],
            target_terms=["API", "API"],
            source_lang="en",
            target_lang="ru",
        )
        assert len(errors) == 0

        # Inconsistent translations - should detect error
        errors = validator.validate_terminology_consistency(
            source_terms=["API", "API", "API"],
            target_terms=["API", "АПИ", "интерфейс"],
            source_lang="en",
            target_lang="ru",
        )
        assert len(errors) == 1
        assert errors[0]["error_type"] == "terminology_inconsistency"
        assert errors[0]["source_term"] == "API"
        assert len(errors[0]["translations"]) == 3

    def test_cache_clearing(self, validator):
        """Test cache clearing."""
        validator._terminology_cache["test"] = {"data"}
        assert len(validator._terminology_cache) > 0

        validator.clear_cache()
        assert len(validator._terminology_cache) == 0

    def test_check_approved_terminology_no_glossary(self, validator):
        """Test checking terms when glossary doesn't exist."""
        # Should return empty list when glossary doesn't exist
        errors = validator.check_approved_terminology(
            terms=["test", "word"],
            language="en",
            glossary_name="nonexistent_glossary",
        )
        assert errors == []

    def test_check_approved_terminology_with_glossary(self, validator, tmp_path):
        """Test checking terms against glossary."""
        import json

        # Create a temporary glossary
        glossary_data = {"terms": {"API": "definition", "SDK": "definition"}}
        glossary_file = tmp_path / "test_glossary.json"
        glossary_file.write_text(json.dumps(glossary_data), encoding="utf-8")

        # Patch the glossaries_dir to use our temp directory
        original_dir = validator.glossary_manager.glossaries_dir
        validator.glossary_manager.glossaries_dir = tmp_path

        try:
            # Term in glossary - no error
            errors = validator.check_approved_terminology(
                terms=["API"],
                language="en",
                glossary_name="test_glossary",
            )
            assert len(errors) == 0

            # Term not in glossary - should report error
            errors = validator.check_approved_terminology(
                terms=["unknown_term"],
                language="en",
                glossary_name="test_glossary",
            )
            assert len(errors) == 1
            assert errors[0]["error_type"] == "unapproved_terminology"
            assert errors[0]["term"] == "unknown_term"
        finally:
            validator.glossary_manager.glossaries_dir = original_dir

    def test_check_approved_terminology_case_insensitive(self, validator, tmp_path):
        """Test that terminology check is case-insensitive."""
        import json

        glossary_data = {"terms": {"API": "definition"}}
        glossary_file = tmp_path / "case_test.json"
        glossary_file.write_text(json.dumps(glossary_data), encoding="utf-8")

        original_dir = validator.glossary_manager.glossaries_dir
        validator.glossary_manager.glossaries_dir = tmp_path

        try:
            # Lowercase should match uppercase in glossary
            errors = validator.check_approved_terminology(
                terms=["api"],
                language="en",
                glossary_name="case_test",
            )
            assert len(errors) == 0
        finally:
            validator.glossary_manager.glossaries_dir = original_dir

    def test_check_approved_terminology_invalid_json(self, validator, tmp_path):
        """Test handling of invalid JSON in glossary file."""
        glossary_file = tmp_path / "invalid.json"
        glossary_file.write_text("not valid json{", encoding="utf-8")

        original_dir = validator.glossary_manager.glossaries_dir
        validator.glossary_manager.glossaries_dir = tmp_path

        try:
            # Should handle gracefully and return empty errors
            errors = validator.check_approved_terminology(
                terms=["test"],
                language="en",
                glossary_name="invalid",
            )
            assert errors == []
        finally:
            validator.glossary_manager.glossaries_dir = original_dir

    def test_detect_false_friends_no_matches(self, validator):
        """Test false friend detection with no matches."""
        errors = validator.detect_false_friends(
            source_text="Hello world",
            target_text="Привет мир",
            source_lang="en",
            target_lang="ru",
        )
        # Should return empty list or list (depending on glossary content)
        assert isinstance(errors, list)

    def test_detect_false_friends_missing_glossary(self, validator):
        """Test false friend detection when glossary is missing."""
        # Use unsupported language pair
        errors = validator.detect_false_friends(
            source_text="Hello",
            target_text="Bonjour",
            source_lang="en",
            target_lang="xx",  # Non-existent language
        )
        # Should handle gracefully
        assert isinstance(errors, list)

    def test_validate_mqm_error_type_subtype(self, validator):
        """Test validating MQM error subtypes."""
        # Test subtype validation (mistranslation is a subtype of accuracy)
        is_valid, info = validator.validate_mqm_error_type("mistranslation", "en")
        assert is_valid
        if info:
            assert "parent_dimension" in info or "id" in info

    def test_validate_mqm_error_type_invalid_language(self, validator):
        """Test MQM validation with invalid language."""
        # Should return False for invalid language (FileNotFoundError is caught)
        is_valid, info = validator.validate_mqm_error_type("accuracy", "xx")
        # Returns False because glossary doesn't exist
        assert is_valid is False
        assert info is None

    def test_get_severity_multiplier_unknown_level(self, validator):
        """Test getting multiplier for unknown severity level."""
        # Should return default 1.0
        multiplier = validator.get_severity_multiplier("unknown_severity", "en")
        assert multiplier == 1.0

    def test_get_severity_multiplier_invalid_language(self, validator):
        """Test getting multiplier with invalid language returns default."""
        # When glossary doesn't exist, should return default 1.0
        # The function catches exceptions and returns default
        multiplier = validator.get_severity_multiplier("major", "xx")
        assert multiplier == 1.0

    def test_validate_language_specific_errors_no_data(self, validator):
        """Test language-specific validation when no data available."""
        # When glossary doesn't exist, should return empty list
        # The function catches FileNotFoundError and returns empty
        errors = validator.validate_language_specific_errors(
            text="Test text",
            language="xx",  # Non-existent language
            error_types=None,
        )
        # Should return empty list
        assert errors == []

    def test_validate_language_specific_errors_with_filter(self, validator):
        """Test language-specific validation with error type filter."""
        errors = validator.validate_language_specific_errors(
            text="Test text",
            language="en",
            error_types=["specific_error_type"],
        )
        # Should return filtered results
        assert isinstance(errors, list)

    def test_validate_terminology_consistency_empty_lists(self, validator):
        """Test terminology consistency with empty input."""
        errors = validator.validate_terminology_consistency(
            source_terms=[],
            target_terms=[],
            source_lang="en",
            target_lang="ru",
        )
        assert errors == []

    def test_validate_terminology_consistency_single_occurrence(self, validator):
        """Test that single occurrences don't trigger inconsistency."""
        errors = validator.validate_terminology_consistency(
            source_terms=["API", "SDK", "UI"],
            target_terms=["API", "SDK", "UI"],
            source_lang="en",
            target_lang="ru",
        )
        assert len(errors) == 0

    def test_validate_terminology_consistency_multiple_terms(self, validator):
        """Test consistency check with multiple different terms."""
        errors = validator.validate_terminology_consistency(
            source_terms=["API", "API", "SDK", "SDK"],
            target_terms=["API", "АПИ", "SDK", "SDK"],
            source_lang="en",
            target_lang="ru",
        )
        # Only API should be flagged as inconsistent
        assert len(errors) == 1
        assert errors[0]["source_term"] == "API"

    def test_initialization_with_custom_manager(self):
        """Test initializing validator with custom glossary manager."""
        custom_manager = GlossaryManager()
        validator = TermValidator(glossary_manager=custom_manager)
        assert validator.glossary_manager is custom_manager

    def test_initialization_creates_default_manager(self):
        """Test that initialization creates default manager if none provided."""
        validator = TermValidator(glossary_manager=None)
        assert validator.glossary_manager is not None
        assert isinstance(validator.glossary_manager, GlossaryManager)


class TestGlossaryIntegration:
    """Integration tests for glossary system."""

    def test_full_workflow(self):
        """Test complete glossary usage workflow."""
        # Initialize
        manager = GlossaryManager()
        validator = TermValidator(manager)

        # Load glossaries
        en_mqm = manager.load_glossary("en", "mqm_core")
        ru_mqm = manager.load_glossary("ru", "mqm_core")

        # Check both loaded successfully
        assert en_mqm["metadata"]["language"] == "en"
        assert ru_mqm["metadata"]["language"] == "ru"

        # Validate error types
        is_valid_en, _ = validator.validate_mqm_error_type("mistranslation", "en")
        assert is_valid_en

        # Get severity
        critical_penalty = validator.get_severity_multiplier("critical", "en")
        assert critical_penalty == 10.0

        # Check terminology consistency
        errors = validator.validate_terminology_consistency(
            source_terms=["machine learning"],
            target_terms=["машинное обучение"],
            source_lang="en",
            target_lang="ru",
        )
        assert len(errors) == 0  # No inconsistency with single occurrence

    def test_multilingual_glossaries(self):
        """Test loading glossaries for multiple languages."""
        manager = GlossaryManager()

        # Load English
        en_mqm = manager.load_glossary("en", "mqm_core")
        assert en_mqm["metadata"]["language"] == "en"

        # Load Russian
        ru_mqm = manager.load_glossary("ru", "mqm_core")
        assert ru_mqm["metadata"]["language"] == "ru"

        # Verify English has error dimensions
        en_dims = manager.get_mqm_error_dimensions("en")
        # Note: Russian glossary uses different structure, just verify it loaded
        assert "error_dimensions" in ru_mqm or "metadata" in ru_mqm
        assert len(en_dims) > 0
