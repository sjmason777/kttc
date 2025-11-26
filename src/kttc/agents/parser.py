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

"""Parser for extracting error annotations from LLM responses.

Parses structured error blocks in the format:
ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Incorrect translation
SUGGESTION: Use correct term
ERROR_END

Enriches errors with MQM definitions from terminology glossaries.
"""

import logging
import re

from kttc.core import ErrorAnnotation, ErrorSeverity
from kttc.terminology import GlossaryManager, TermValidator

logger = logging.getLogger(__name__)


class ErrorParser:
    """Parser for extracting error annotations from LLM text responses.

    Supports optional glossary enrichment to add MQM definitions and
    multi-lingual error descriptions to parsed errors.
    """

    ERROR_BLOCK_PATTERN = re.compile(r"ERROR_START\s*(.*?)\s*ERROR_END", re.DOTALL | re.IGNORECASE)

    FIELD_PATTERN = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)

    @classmethod
    def parse_errors(
        cls, llm_response: str, enrich_with_glossary: bool = True, language: str = "en"
    ) -> list[ErrorAnnotation]:
        """Parse all error blocks from LLM response.

        Args:
            llm_response: Raw text response from LLM
            enrich_with_glossary: Whether to enrich errors with MQM glossary data (default: True)
            language: Language code for glossary lookup (default: "en")

        Returns:
            List of parsed error annotations, optionally enriched with glossary data

        Raises:
            AgentParsingError: If parsing fails

        Example:
            >>> response = "ERROR_START\\nCATEGORY: accuracy\\n...\\nERROR_END"
            >>> errors = ErrorParser.parse_errors(response)
            >>> print(errors[0].category)
            accuracy
            >>> # Enriched errors include MQM definitions
            >>> errors_enriched = ErrorParser.parse_errors(response, enrich_with_glossary=True)
        """
        errors = []
        error_blocks = cls.ERROR_BLOCK_PATTERN.findall(llm_response)

        if not error_blocks:
            # No errors found - this is valid (translation might be perfect)
            return []

        for block in error_blocks:
            try:
                error = cls._parse_error_block(block)
                errors.append(error)
            except (ValueError, KeyError):
                # Log warning but continue parsing other errors
                logger.warning(f"Failed to parse error block: {block[:100]}...")
                continue

        # Enrich errors with glossary data if requested
        if enrich_with_glossary and errors:
            errors = cls._enrich_with_glossary(errors, language)

        return errors

    @classmethod
    def _parse_error_block(cls, block: str) -> ErrorAnnotation:
        """Parse a single error block into ErrorAnnotation.

        Args:
            block: Text content between ERROR_START and ERROR_END

        Returns:
            Parsed error annotation

        Raises:
            ValueError: If required fields are missing or invalid
            KeyError: If field parsing fails
        """
        fields: dict[str, str] = {}

        # Extract all field: value pairs
        for match in cls.FIELD_PATTERN.finditer(block):
            field_name = match.group(1).lower().strip()
            field_value = match.group(2).strip()
            fields[field_name] = field_value

        # Validate required fields
        required_fields = ["category", "subcategory", "severity", "location", "description"]
        missing_fields = [f for f in required_fields if f not in fields]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Parse location (e.g., "0-5" or "0, 5")
        location = cls._parse_location(fields["location"])

        # Parse severity
        severity = cls._parse_severity(fields["severity"])

        # Create error annotation
        return ErrorAnnotation(
            category=fields["category"],
            subcategory=fields["subcategory"],
            severity=severity,
            location=location,
            description=fields["description"],
            suggestion=fields.get("suggestion"),  # Optional field
        )

    @staticmethod
    def _parse_location(location_str: str) -> tuple[int, int]:
        """Parse location string into (start, end) tuple.

        Args:
            location_str: Location string like "0-5", "0,5", "[0, 5]"

        Returns:
            Tuple of (start_char, end_char)

        Raises:
            ValueError: If location format is invalid
        """
        # Remove brackets and whitespace
        location_str = location_str.strip("[]()").replace(" ", "")

        # Try different separators
        for separator in ["-", ",", ":"]:
            if separator in location_str:
                parts = location_str.split(separator)
                if len(parts) == 2:
                    try:
                        start = int(parts[0])
                        end = int(parts[1])
                        if start < 0 or end < start:
                            raise ValueError(f"Invalid location range: {start}-{end}")
                        return (start, end)
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Invalid location format: {location_str}") from e

        raise ValueError(f"Invalid location format: {location_str}")

    @staticmethod
    def _parse_severity(severity_str: str) -> ErrorSeverity:
        """Parse severity string into ErrorSeverity enum.

        Args:
            severity_str: Severity string like "major", "critical", etc.

        Returns:
            ErrorSeverity enum value

        Raises:
            ValueError: If severity is not recognized
        """
        severity_lower = severity_str.lower().strip()

        severity_map = {
            "neutral": ErrorSeverity.NEUTRAL,
            "minor": ErrorSeverity.MINOR,
            "major": ErrorSeverity.MAJOR,
            "critical": ErrorSeverity.CRITICAL,
        }

        if severity_lower not in severity_map:
            raise ValueError(
                f"Unknown severity: {severity_str}. "
                f"Must be one of: {', '.join(severity_map.keys())}"
            )

        return severity_map[severity_lower]

    @classmethod
    def _enrich_with_glossary(
        cls, errors: list[ErrorAnnotation], language: str = "en"
    ) -> list[ErrorAnnotation]:
        """Enrich error annotations with MQM glossary definitions.

        Args:
            errors: List of parsed error annotations
            language: Language code for glossary lookup (default: "en")

        Returns:
            List of errors enriched with glossary data (definitions, examples)

        Example:
            >>> errors = [ErrorAnnotation(...)]
            >>> enriched = ErrorParser._enrich_with_glossary(errors, "en")
            >>> # enriched[0].description now includes MQM definition
        """
        try:
            # Initialize glossary manager and term validator
            glossary_manager = GlossaryManager()
            term_validator = TermValidator()

            # Load MQM glossary for the target language
            mqm_glossary = glossary_manager.load_glossary(language, "mqm_core")

            if not mqm_glossary:
                logger.debug(
                    f"No MQM glossary found for language '{language}', skipping enrichment"
                )
                return errors

            # Enrich each error with glossary information
            for error in errors:
                # Validate and get MQM error type information
                is_valid, mqm_info = term_validator.validate_mqm_error_type(
                    error.subcategory, language
                )

                if is_valid and mqm_info:
                    # Add MQM definition to description if available
                    if (
                        "definition" in mqm_info
                        and mqm_info["definition"]
                        and "[MQM:" not in error.description
                    ):
                        error.description = f"{error.description} [MQM: {mqm_info['definition']}]"

                    # Log successful enrichment
                    logger.debug(
                        f"Enriched error '{error.subcategory}' with MQM definition from glossary"
                    )
                else:
                    # Log unknown error types (might be custom or language-specific)
                    logger.debug(
                        f"No MQM definition found for error type '{error.subcategory}' "
                        f"in language '{language}'"
                    )

            logger.info(
                f"Enriched {len(errors)} errors with MQM glossary data (language: {language})"
            )

        except Exception as e:
            # Log error but don't fail - enrichment is optional
            logger.warning(f"Failed to enrich errors with glossary data: {e}")

        return errors
