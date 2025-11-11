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
"""

import re

from kttc.core import ErrorAnnotation, ErrorSeverity


class ErrorParser:
    """Parser for extracting error annotations from LLM text responses."""

    ERROR_BLOCK_PATTERN = re.compile(r"ERROR_START\s*(.*?)\s*ERROR_END", re.DOTALL | re.IGNORECASE)

    FIELD_PATTERN = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)

    @classmethod
    def parse_errors(cls, llm_response: str) -> list[ErrorAnnotation]:
        """Parse all error blocks from LLM response.

        Args:
            llm_response: Raw text response from LLM

        Returns:
            List of parsed error annotations

        Raises:
            AgentParsingError: If parsing fails

        Example:
            >>> response = "ERROR_START\\nCATEGORY: accuracy\\n...\\nERROR_END"
            >>> errors = ErrorParser.parse_errors(response)
            >>> print(errors[0].category)
            accuracy
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
                # In production, you'd want proper logging here
                continue

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
