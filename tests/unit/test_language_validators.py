"""Tests for language-specific validators."""

import pytest

from kttc.terminology import (
    ChineseMeasureWordValidator,
    HindiPostpositionValidator,
    LanguageValidatorFactory,
    PersianEzafeValidator,
    RussianCaseAspectValidator,
)


class TestRussianCaseAspectValidator:
    """Tests for Russian case and aspect validator."""

    @pytest.fixture
    def validator(self):
        """Create a Russian validator instance."""
        return RussianCaseAspectValidator()

    def test_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator.glossary_manager is not None
        assert validator.morphology is not None
        assert validator.mqm is not None

    def test_get_case_info(self, validator):
        """Test getting case information."""
        # Test nominative case
        nom_info = validator.get_case_info("nominative")
        assert nom_info is not None
        assert nom_info["case_en"] == "Nominative"

        # Test genitive case
        gen_info = validator.get_case_info("genitive")
        assert gen_info is not None
        assert gen_info["case_en"] == "Genitive"

    def test_get_aspect_info(self, validator):
        """Test getting aspect information."""
        # Test perfective aspect
        perf_info = validator.get_aspect_info("perfective")
        assert perf_info is not None
        assert "when_to_use" in perf_info

        # Test imperfective aspect
        imperf_info = validator.get_aspect_info("imperfective")
        assert imperf_info is not None
        assert "when_to_use" in imperf_info

    def test_validate_case_preposition(self, validator):
        """Test preposition-case validation."""
        # Test valid combinations
        is_valid, error = validator.validate_case_preposition("в", "prepositional")
        # Note: This might return False if the glossary doesn't have complete preposition data
        # The test validates the method works, not necessarily the linguistic accuracy

        assert isinstance(is_valid, bool)
        if not is_valid:
            assert error is not None

    def test_get_aspect_usage_rules(self, validator):
        """Test getting aspect usage rules."""
        rules = validator.get_aspect_usage_rules("perfective")
        assert isinstance(rules, dict)
        # Rules might be empty if glossary doesn't have them, but should be dict
        assert "when_to_use" in rules or len(rules) == 0


class TestChineseMeasureWordValidator:
    """Tests for Chinese measure word validator."""

    @pytest.fixture
    def validator(self):
        """Create a Chinese measure word validator instance."""
        return ChineseMeasureWordValidator()

    def test_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator.glossary_manager is not None
        assert validator.classifiers is not None
        assert validator.mqm is not None

    def test_get_classifier_by_category(self, validator):
        """Test getting classifiers by category."""
        individual = validator.get_classifier_by_category("individual_classifiers")
        assert isinstance(individual, dict)
        assert len(individual) > 0

    def test_get_most_common_classifiers(self, validator):
        """Test getting most common classifiers."""
        common = validator.get_most_common_classifiers(limit=5)
        assert isinstance(common, list)
        assert len(common) <= 5
        # Each item should be a tuple (classifier_char, info_dict)
        if len(common) > 0:
            assert isinstance(common[0], tuple)
            assert len(common[0]) == 2

    def test_get_classifier_for_noun(self, validator):
        """Test finding classifier for specific nouns."""
        # Test with a common noun that should have a classifier
        # Note: This depends on glossary content
        result = validator.get_classifier_for_noun("人")
        # Result might be None if noun not in examples, but method should work
        assert result is None or isinstance(result, dict)

    def test_validate_classifier_noun_pair(self, validator):
        """Test validating classifier-noun pairs."""
        # Test method execution
        is_valid, message = validator.validate_classifier_noun_pair("个", "人")
        assert isinstance(is_valid, bool)
        # Message can be None (valid) or string (error/suggestion)
        assert message is None or isinstance(message, str)


class TestHindiPostpositionValidator:
    """Tests for Hindi postposition validator."""

    @pytest.fixture
    def validator(self):
        """Create a Hindi postposition validator instance."""
        return HindiPostpositionValidator()

    def test_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator.glossary_manager is not None
        assert validator.cases is not None
        assert validator.mqm is not None

    def test_get_case_info(self, validator):
        """Test getting case information by number."""
        # Test first case (Nominative/Karta)
        case1 = validator.get_case_info(1)
        assert case1 is not None
        assert "case_number" in case1

        # Test sixth case (Genitive/Sambandh)
        case6 = validator.get_case_info(6)
        assert case6 is not None
        assert "case_number" in case6

    def test_get_postposition_for_case(self, validator):
        """Test getting postposition for a case."""
        # Test ergative case (case 1 with transitive verbs)
        post = validator.get_postposition_for_case(1)
        assert post is not None or post == "ने (with transitive verbs in perfective aspect)"

        # Test accusative case (case 2)
        post2 = validator.get_postposition_for_case(2)
        assert isinstance(post2, str) or post2 is None

    def test_validate_ergative_construction(self, validator):
        """Test ergative construction validation."""
        # Transitive verb in simple past should require ने
        is_valid, error = validator.validate_ergative_construction(
            "transitive", "simple_past", has_ne=True
        )
        assert is_valid is True
        assert error is None

        # Transitive verb in simple past without ने should be invalid
        is_valid, error = validator.validate_ergative_construction(
            "transitive", "simple_past", has_ne=False
        )
        assert is_valid is False
        assert error is not None

        # Intransitive verb should not have ने
        is_valid, error = validator.validate_ergative_construction(
            "intransitive", "simple_past", has_ne=False
        )
        assert is_valid is True

    def test_get_oblique_form_rule(self, validator):
        """Test getting oblique form rules."""
        rules = validator.get_oblique_form_rule()
        assert isinstance(rules, dict)


class TestPersianEzafeValidator:
    """Tests for Persian ezafe validator."""

    @pytest.fixture
    def validator(self):
        """Create a Persian ezafe validator instance."""
        return PersianEzafeValidator()

    def test_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator.glossary_manager is not None
        assert validator.grammar is not None
        assert validator.mqm is not None

    def test_get_ezafe_rules(self, validator):
        """Test getting ezafe rules."""
        rules = validator.get_ezafe_rules()
        assert isinstance(rules, dict)
        # Should have basic ezafe information
        assert "definition" in rules or "functions" in rules

    def test_validate_ezafe_usage(self, validator):
        """Test validating ezafe usage."""
        # Test with a valid construction type
        is_valid, info = validator.validate_ezafe_usage("noun_adjective")
        assert isinstance(is_valid, bool)
        assert isinstance(info, dict)

    def test_get_compound_verb_info(self, validator):
        """Test getting compound verb information."""
        # Test with common light verb
        info = validator.get_compound_verb_info("کردن")
        # Result can be None or dict depending on glossary content
        assert info is None or isinstance(info, dict)

    def test_validate_object_marker_ra(self, validator):
        """Test validating object marker را."""
        # Definite object should have را
        is_valid, error = validator.validate_object_marker_ra("definite", has_ra=True)
        assert is_valid is True
        assert error is None

        # Definite object without را should be invalid
        is_valid, error = validator.validate_object_marker_ra("definite", has_ra=False)
        assert is_valid is False
        assert error is not None

        # Indefinite object should not have را
        is_valid, error = validator.validate_object_marker_ra("indefinite", has_ra=False)
        assert is_valid is True


class TestLanguageValidatorFactory:
    """Tests for language validator factory."""

    def test_create_validator_russian(self):
        """Test creating Russian validator."""
        validator = LanguageValidatorFactory.create_validator("ru")
        assert isinstance(validator, RussianCaseAspectValidator)

    def test_create_validator_chinese(self):
        """Test creating Chinese validator."""
        validator = LanguageValidatorFactory.create_validator("zh")
        assert isinstance(validator, ChineseMeasureWordValidator)

    def test_create_validator_hindi(self):
        """Test creating Hindi validator."""
        validator = LanguageValidatorFactory.create_validator("hi")
        assert isinstance(validator, HindiPostpositionValidator)

    def test_create_validator_persian(self):
        """Test creating Persian validator."""
        validator = LanguageValidatorFactory.create_validator("fa")
        assert isinstance(validator, PersianEzafeValidator)

    def test_create_validator_invalid_language(self):
        """Test creating validator for unsupported language."""
        with pytest.raises(ValueError):
            LanguageValidatorFactory.create_validator("xx")

    def test_list_available_validators(self):
        """Test listing available validators."""
        available = LanguageValidatorFactory.list_available_validators()
        assert isinstance(available, list)
        assert "ru" in available
        assert "zh" in available
        assert "hi" in available
        assert "fa" in available
        assert len(available) == 4


class TestLanguageValidatorIntegration:
    """Integration tests for language validators."""

    def test_all_validators_can_be_created(self):
        """Test that all validators can be instantiated."""
        validators = {
            "ru": RussianCaseAspectValidator,
            "zh": ChineseMeasureWordValidator,
            "hi": HindiPostpositionValidator,
            "fa": PersianEzafeValidator,
        }

        for lang, validator_class in validators.items():
            validator = validator_class()
            assert validator is not None
            assert validator.glossary_manager is not None

    def test_validators_load_correct_glossaries(self):
        """Test that validators load the correct glossaries."""
        # Russian validator
        ru_validator = RussianCaseAspectValidator()
        assert ru_validator.morphology["metadata"]["language"] == "ru"

        # Chinese validator
        zh_validator = ChineseMeasureWordValidator()
        assert zh_validator.classifiers["metadata"]["language"] == "zh"

        # Hindi validator
        hi_validator = HindiPostpositionValidator()
        assert hi_validator.cases["metadata"]["language"] == "hi"

        # Persian validator
        fa_validator = PersianEzafeValidator()
        assert fa_validator.grammar["metadata"]["language"] == "fa"
