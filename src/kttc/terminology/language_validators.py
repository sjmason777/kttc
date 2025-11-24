"""
Language-specific validators using glossaries.

This module provides validators for language-specific grammatical features
that leverage the terminology glossaries for validation rules.
"""

from typing import Any, cast

from .glossary_manager import GlossaryManager


class RussianCaseAspectValidator:
    """Validator for Russian case and aspect usage."""

    def __init__(self, glossary_manager: GlossaryManager | None = None):
        """
        Initialize Russian validator.

        Args:
            glossary_manager: GlossaryManager instance. If None, creates a new one.
        """
        self.glossary_manager = glossary_manager or GlossaryManager()
        self.morphology = self.glossary_manager.load_glossary("ru", "morphology_ru")
        self.mqm = self.glossary_manager.load_glossary("ru", "mqm_core")

    def get_case_info(self, case_name: str) -> dict[str, Any] | None:
        """
        Get information about a Russian case.

        Args:
            case_name: Name of the case (nominative, genitive, etc.)

        Returns:
            Dictionary with case information or None if not found
        """
        cases = (
            self.morphology.get("grammatical_categories", {})
            .get("case", {})
            .get("russian_cases", {})
        )

        # Look up case directly by key (case name is the key in the dict)
        case_info = cases.get(case_name.lower())
        if case_info:
            # Add the case name in English for consistency
            return {"case_en": case_name.capitalize(), **case_info}
        return None

    def get_aspect_info(self, aspect_name: str) -> dict[str, Any] | None:
        """
        Get information about a Russian aspect.

        Args:
            aspect_name: Name of the aspect (perfective, imperfective)

        Returns:
            Dictionary with aspect information or None if not found
        """
        aspects = (
            self.morphology.get("grammatical_categories", {})
            .get("aspect", {})
            .get("russian_aspects", {})
        )

        if aspect_name.lower() in ["perfective", "совершенный"]:
            aspect_info = aspects.get("perfective")
            if aspect_info:
                # Map "characteristics" to "when_to_use" for consistency
                return {
                    "when_to_use": aspect_info.get("characteristics", []),
                    "examples": aspect_info.get("examples", ""),
                    **aspect_info,
                }
        elif aspect_name.lower() in ["imperfective", "несовершенный"]:
            aspect_info = aspects.get("imperfective")
            if aspect_info:
                # Map "characteristics" to "when_to_use" for consistency
                return {
                    "when_to_use": aspect_info.get("characteristics", []),
                    "examples": aspect_info.get("examples", ""),
                    **aspect_info,
                }

        return None

    def validate_case_preposition(self, preposition: str, case: str) -> tuple[bool, str | None]:
        """
        Validate if a preposition requires a specific case.

        Args:
            preposition: Russian preposition
            case: Case name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        case_info = self.get_case_info(case)
        if not case_info:
            return False, f"Unknown case: {case}"

        # Get common prepositions for this case from glossary
        common_preps = case_info.get("common_prepositions", [])

        if preposition.lower() in [p.lower() for p in common_preps]:
            return True, None

        return False, f"Preposition '{preposition}' typically does not take {case} case"

    def get_aspect_usage_rules(self, aspect: str) -> dict[str, Any]:
        """
        Get usage rules for a specific aspect.

        Args:
            aspect: Aspect name (perfective or imperfective)

        Returns:
            Dictionary with usage rules
        """
        aspect_info = self.get_aspect_info(aspect)
        if not aspect_info:
            return {}

        return {
            "when_to_use": aspect_info.get("when_to_use", []),
            "indicators": aspect_info.get("indicators", []),
            "examples": aspect_info.get("examples", []),
        }


class ChineseMeasureWordValidator:
    """Validator for Chinese measure words (classifiers)."""

    def __init__(self, glossary_manager: GlossaryManager | None = None):
        """
        Initialize Chinese measure word validator.

        Args:
            glossary_manager: GlossaryManager instance. If None, creates a new one.
        """
        self.glossary_manager = glossary_manager or GlossaryManager()
        self.classifiers = self.glossary_manager.load_glossary("zh", "classifiers_zh")
        self.mqm = self.glossary_manager.load_glossary("zh", "mqm_core")

    def get_classifier_for_noun(self, noun: str) -> dict[str, Any] | None:
        """
        Find appropriate classifier(s) for a given noun.

        Args:
            noun: Chinese noun

        Returns:
            Dictionary with classifier information or None
        """
        # Search through all classifier categories
        for category in [
            "individual_classifiers",
            "collective_classifiers",
            "container_classifiers",
            "measurement_classifiers",
            "temporal_classifiers",
            "verbal_classifiers",
        ]:
            classifiers_dict = self.classifiers.get(category, {}).get("classifiers", {})

            # classifiers is a dict, not a list
            for clf_char, clf_info in classifiers_dict.items():
                examples = clf_info.get("examples", [])
                # Check if noun appears in any example
                for example in examples:
                    if noun in example:
                        return {
                            "classifier": clf_char,
                            "pinyin": clf_info.get("pinyin"),
                            "category": category,
                            "usage": clf_info.get("usage"),
                            "examples": examples,
                        }

        return None

    def validate_classifier_noun_pair(self, classifier: str, noun: str) -> tuple[bool, str | None]:
        """
        Validate if a classifier is appropriate for a noun.

        Args:
            classifier: Chinese classifier
            noun: Chinese noun

        Returns:
            Tuple of (is_valid, suggestion_or_error)
        """
        correct_clf = self.get_classifier_for_noun(noun)

        if not correct_clf:
            return True, f"No specific classifier rule found for '{noun}'"

        if correct_clf["classifier"] == classifier:
            return True, None

        return (
            False,
            f"For '{noun}', consider using '{correct_clf['classifier']}' ({correct_clf['pinyin']}) instead of '{classifier}'",
        )

    def get_classifier_by_category(self, category: str) -> dict[str, Any]:
        """
        Get all classifiers in a specific category.

        Args:
            category: Category name (e.g., "individual_classifiers")

        Returns:
            Dictionary of classifiers
        """
        result: dict[str, Any] = self.classifiers.get(category, {}).get("classifiers", {})
        return result

    def get_most_common_classifiers(self, limit: int = 10) -> list[tuple[str, dict[str, Any]]]:
        """
        Get the most common classifiers.

        Args:
            limit: Maximum number of classifiers to return

        Returns:
            List of tuples (classifier_character, classifier_info)
        """
        # 个 is the most common, followed by others
        common = []
        individual = self.get_classifier_by_category("individual_classifiers")

        # Get first 'limit' classifiers from individual category
        for i, (clf_char, clf_info) in enumerate(individual.items()):
            if i >= limit:
                break
            common.append((clf_char, clf_info))

        return common


class HindiPostpositionValidator:
    """Validator for Hindi postpositions and case markers."""

    def __init__(self, glossary_manager: GlossaryManager | None = None):
        """
        Initialize Hindi postposition validator.

        Args:
            glossary_manager: GlossaryManager instance. If None, creates a new one.
        """
        self.glossary_manager = glossary_manager or GlossaryManager()
        self.cases = self.glossary_manager.load_glossary("hi", "cases_hi")
        self.mqm = self.glossary_manager.load_glossary("hi", "mqm_core")

    def get_case_info(self, case_number: int) -> dict[str, Any] | None:
        """
        Get information about a Hindi case by number (1-8).

        Args:
            case_number: Case number (1-8)

        Returns:
            Dictionary with case information or None if not found
        """
        cases = self.cases.get("eight_cases", {})

        case_key = (
            f"{case_number}_"
            + [
                "कर्ता_कारक",
                "कर्म_कारक",
                "करण_कारक",
                "संप्रदान_कारक",
                "अपादान_कारक",
                "संबंध_कारक",
                "अधिकरण_कारक",
                "संबोधन_कारक",
            ][case_number - 1]
        )

        result: dict[str, Any] | None = cases.get(case_key)
        return result

    def get_postposition_for_case(self, case_number: int) -> str | None:
        """
        Get the main postposition for a case.

        Args:
            case_number: Case number (1-8)

        Returns:
            Postposition string or None
        """
        case_info = self.get_case_info(case_number)
        if case_info:
            return case_info.get("postposition")
        return None

    def validate_ergative_construction(
        self, verb_type: str, tense: str, has_ne: bool
    ) -> tuple[bool, str | None]:
        """
        Validate ergative 'ने' usage in Hindi.

        Args:
            verb_type: "transitive" or "intransitive"
            tense: Tense name
            has_ne: Whether 'ने' is used

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Ergative 'ने' is required for transitive verbs in perfective past
        perfective_tenses = ["simple_past", "perfect", "pluperfect"]

        if verb_type == "transitive" and tense in perfective_tenses:
            if not has_ne:
                return False, "Ergative marker 'ने' required for transitive verbs in perfective past"
            return True, None
        elif verb_type == "intransitive":
            if has_ne:
                return False, "Ergative marker 'ने' should not be used with intransitive verbs"
            return True, None
        else:
            if has_ne and tense not in perfective_tenses:
                return False, f"Ergative marker 'ने' not typically used in {tense}"
            return True, None

    def get_oblique_form_rule(self) -> dict[str, Any]:
        """
        Get rules for oblique form usage.

        Returns:
            Dictionary with oblique form rules
        """
        result: dict[str, Any] = self.cases.get("case_system_overview", {}).get("तिर्यक_रूप", {})
        return result


class PersianEzafeValidator:
    """Validator for Persian ezafe construction."""

    def __init__(self, glossary_manager: GlossaryManager | None = None):
        """
        Initialize Persian ezafe validator.

        Args:
            glossary_manager: GlossaryManager instance. If None, creates a new one.
        """
        self.glossary_manager = glossary_manager or GlossaryManager()
        self.grammar = self.glossary_manager.load_glossary("fa", "grammar_fa")
        self.mqm = self.glossary_manager.load_glossary("fa", "mqm_core")

    def get_ezafe_rules(self) -> dict[str, Any]:
        """
        Get ezafe construction rules.

        Returns:
            Dictionary with ezafe rules
        """
        result: dict[str, Any] = self.grammar.get("ezafe_construction", {}).get("اضافه", {})
        return result

    def validate_ezafe_usage(self, construction_type: str) -> tuple[bool, dict[str, Any]]:
        """
        Validate ezafe usage for a construction type.

        Args:
            construction_type: Type of construction (noun_adjective, noun_noun, etc.)

        Returns:
            Tuple of (is_valid, usage_info)
        """
        ezafe_info = self.get_ezafe_rules()
        functions = ezafe_info.get("functions", [])
        examples = ezafe_info.get("examples", {})

        if construction_type in examples:
            return True, {
                "rule": functions,
                "examples": examples[construction_type],
                "pronunciation": ezafe_info.get("pronunciation"),
            }

        return False, {"error": f"Unknown construction type: {construction_type}"}

    def get_compound_verb_info(self, light_verb: str) -> dict[str, Any] | None:
        """
        Get information about compound verbs with a specific light verb.

        Args:
            light_verb: Persian light verb (کردن, شدن, etc.)

        Returns:
            Dictionary with compound verb information
        """
        compound_verbs: dict[str, Any] = self.grammar.get("compound_verbs", {}).get("فعل_مرکب", {})
        light_verbs = compound_verbs.get("light_verbs", [])

        for lv in light_verbs:
            if light_verb in str(lv):
                result: dict[str, Any] = compound_verbs
                return result

        examples = compound_verbs.get("examples", {})

        # Map light verb to its key
        light_verb_map = {"کردن": "با_کردن", "شدن": "با_شدن", "زدن": "با_زدن", "دادن": "با_دادن"}

        key = light_verb_map.get(light_verb)
        if key and key in examples:
            return {
                "light_verb": light_verb,
                "examples": examples[key],
                "structure": compound_verbs.get("structure"),
            }

        return None

    def validate_object_marker_ra(self, object_type: str, has_ra: bool) -> tuple[bool, str | None]:
        """
        Validate object marker 'را' usage.

        Args:
            object_type: Type of object (definite, indefinite, proper_noun, pronoun)
            has_ra: Whether 'را' is present

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Rules from glossary
        should_have_ra = object_type in ["definite", "proper_noun", "pronoun"]

        if should_have_ra and not has_ra:
            return False, f"Object marker 'را' required for {object_type} objects"
        elif not should_have_ra and has_ra:
            return False, f"Object marker 'را' not typically used with {object_type} objects"

        return True, None


class LanguageValidatorFactory:
    """Factory for creating language-specific validators."""

    @staticmethod
    def create_validator(
        language: str, glossary_manager: GlossaryManager | None = None
    ) -> (
        RussianCaseAspectValidator
        | ChineseMeasureWordValidator
        | HindiPostpositionValidator
        | PersianEzafeValidator
    ):
        """
        Create a language-specific validator.

        Args:
            language: Language code (ru, zh, hi, fa)
            glossary_manager: Optional GlossaryManager instance

        Returns:
            Language-specific validator instance

        Raises:
            ValueError: If language is not supported
        """
        validators = {
            "ru": RussianCaseAspectValidator,
            "zh": ChineseMeasureWordValidator,
            "hi": HindiPostpositionValidator,
            "fa": PersianEzafeValidator,
        }

        if language not in validators:
            raise ValueError(f"No validator available for language: {language}")

        return cast(
            RussianCaseAspectValidator
            | ChineseMeasureWordValidator
            | HindiPostpositionValidator
            | PersianEzafeValidator,
            validators[language](glossary_manager),
        )

    @staticmethod
    def list_available_validators() -> list[str]:
        """
        List all available language validators.

        Returns:
            List of language codes with validators
        """
        return ["ru", "zh", "hi", "fa"]
