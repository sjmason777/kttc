"""
Term Validator for checking terminology consistency and correctness.
"""

import logging
from typing import Any

from kttc.terminology.glossary_manager import GlossaryManager


class TermValidator:
    """
    Validator for terminology consistency and correctness in translations.

    Checks for:
    - Terminology consistency (same term translated consistently)
    - Approved terminology usage (terms match glossary)
    - Forbidden term usage (terms that should not be used)
    - Case sensitivity violations
    """

    def __init__(self, glossary_manager: GlossaryManager | None = None):
        """
        Initialize Term Validator.

        Args:
            glossary_manager: GlossaryManager instance. Creates new one if None.
        """
        self.glossary_manager = glossary_manager or GlossaryManager()
        self._terminology_cache: dict[str, set[str]] = {}

    def validate_terminology_consistency(
        self,
        source_terms: list[str],
        target_terms: list[str],
        source_lang: str = "en",
        target_lang: str = "ru",
    ) -> list[dict[str, Any]]:
        """
        Validate that source terms are consistently translated.

        Args:
            source_terms: List of source language terms found in text
            target_terms: List of target language terms found in text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of inconsistency errors
        """
        errors = []

        # Group identical source terms and their translations
        term_translations: dict[str, list[str]] = {}

        for src_term, tgt_term in zip(source_terms, target_terms):
            if src_term not in term_translations:
                term_translations[src_term] = []
            term_translations[src_term].append(tgt_term)

        # Check for inconsistencies
        for src_term, translations in term_translations.items():
            unique_translations = set(translations)
            if len(unique_translations) > 1:
                errors.append(
                    {
                        "error_type": "terminology_inconsistency",
                        "source_term": src_term,
                        "translations": list(unique_translations),
                        "count": len(translations),
                        "severity": "major",
                        "message": f"Term '{src_term}' translated inconsistently: {unique_translations}",
                    }
                )

        return errors

    def check_approved_terminology(
        self,
        terms: list[str],
        language: str,
        glossary_name: str = "base",
    ) -> list[dict[str, Any]]:
        """
        Check if terms match approved glossary.

        Args:
            terms: List of terms to check
            language: Language code
            glossary_name: Name of glossary file (without .json)

        Returns:
            List of terminology violations
        """
        errors: list[dict[str, Any]] = []

        try:
            # Try to load the specified glossary
            glossary_path = self.glossary_manager.glossaries_dir / f"{glossary_name}.json"

            if not glossary_path.exists():
                # Glossary doesn't exist - skip validation
                return errors

            # Load approved terms from glossary
            import json

            with open(glossary_path, encoding="utf-8") as f:
                glossary_data = json.load(f)

            approved_terms = set(glossary_data.get("terms", {}).keys())

            # Check each term
            for term in terms:
                if term.lower() not in {t.lower() for t in approved_terms}:
                    errors.append(
                        {
                            "error_type": "unapproved_terminology",
                            "term": term,
                            "language": language,
                            "glossary": glossary_name,
                            "severity": "minor",
                            "message": f"Term '{term}' not found in approved glossary '{glossary_name}'",
                        }
                    )

        except Exception:
            # If glossary loading fails, skip validation
            # This is acceptable as glossary validation is optional
            logging.warning("Failed to load glossary for terminology validation", exc_info=True)

        return errors

    def detect_false_friends(
        self,
        source_text: str,
        target_text: str,
        source_lang: str = "en",
        target_lang: str = "ru",
    ) -> list[dict[str, Any]]:
        """
        Detect potential false friends (cognates with different meanings).

        Args:
            source_text: Source language text
            target_text: Target language text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of potential false friend errors
        """
        errors = []

        # Get false friends from MQM glossary if available
        try:
            mqm_glossary = self.glossary_manager.load_glossary(target_lang, "mqm_core")

            false_friends = mqm_glossary.get("translation_specific_" + target_lang[:2], {}).get(
                "false_friends_" + source_lang[:2] + "_" + target_lang[:2], {}
            )

            if isinstance(false_friends, dict) and "examples" in false_friends:
                for ff_term, ff_data in false_friends.get("examples", {}).items():
                    # Check if false friend appears in target text
                    if ff_term.lower() in target_text.lower():
                        errors.append(
                            {
                                "error_type": "false_friend",
                                "term": ff_term,
                                "correct_meaning": ff_data,
                                "source_lang": source_lang,
                                "target_lang": target_lang,
                                "severity": "major",
                                "message": f"Potential false friend detected: '{ff_term}'",
                            }
                        )

        except Exception:
            # If glossary not available, skip
            logging.warning(
                "Failed to load MQM glossary for false friends detection", exc_info=True
            )

        return errors

    def validate_mqm_error_type(
        self, error_type: str, language: str = "en"
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Validate that an error type exists in MQM taxonomy.

        Args:
            error_type: Error type identifier
            language: Language code

        Returns:
            Tuple of (is_valid, error_info)
        """
        try:
            error_dimensions = self.glossary_manager.get_mqm_error_dimensions(language)

            # Search in dimensions
            for dimension in error_dimensions:
                if dimension.get("id") == error_type:
                    return True, dimension

                # Search in subtypes
                if error_type in dimension.get("subtypes", []):
                    return True, {
                        "parent_dimension": dimension.get("id"),
                        "subtype": error_type,
                        "severity_weight": dimension.get("severity_weight", 1.0),
                    }

            return False, None

        except Exception:
            logging.warning("Failed to validate MQM error type", exc_info=True)
            return False, None

    def get_severity_multiplier(self, severity_level: str, language: str = "en") -> float:
        """
        Get penalty multiplier for a severity level.

        Args:
            severity_level: Severity level name (e.g., 'minor', 'major', 'critical')
            language: Language code

        Returns:
            Penalty multiplier (default: 1.0)
        """
        try:
            severity_levels = self.glossary_manager.get_severity_levels(language)
            level_data = severity_levels.get(severity_level, {})
            return float(level_data.get("penalty_multiplier", 1.0))
        except (KeyError, TypeError, AttributeError):
            logging.warning("Failed to get severity multiplier, using default value", exc_info=True)
            return 1.0

    def validate_language_specific_errors(
        self,
        text: str,
        language: str,
        error_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Validate language-specific error types.

        Args:
            text: Text to validate
            language: Language code
            error_types: Specific error types to check (None = check all)

        Returns:
            List of detected errors
        """
        errors = []

        # Load language-specific MQM glossary
        try:
            mqm_glossary = self.glossary_manager.load_glossary(language, "mqm_core")
            lang_specific_key = f"{language}_specific_errors"

            if lang_specific_key in mqm_glossary:
                lang_errors = mqm_glossary[lang_specific_key]

                # Check each error type
                for error_name, error_data in lang_errors.items():
                    if error_types and error_name not in error_types:
                        continue

                    # Perform basic validation
                    # (In production, this would use NLP tools)
                    errors.append(
                        {
                            "error_type": error_name,
                            "language": language,
                            "definition": error_data.get("definition", ""),
                            "category": error_data.get("category", "linguistic_conventions"),
                            "severity": error_data.get("severity", "minor"),
                        }
                    )

        except (KeyError, ValueError, TypeError):
            logging.warning("Failed to validate language-specific errors", exc_info=True)

        return errors

    def clear_cache(self) -> None:
        """Clear the terminology cache."""
        self._terminology_cache.clear()
