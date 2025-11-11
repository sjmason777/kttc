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

"""Hallucination Detection Agent.

Detects hallucinated content in translations including:
- Factual errors (information not in source)
- Entity preservation issues
- Extreme length deviations
- Semantic inconsistencies

Based on NAACL 2025 research on hallucination mitigation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider

from .base import AgentEvaluationError, AgentParsingError, BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HallucinationAgent(BaseAgent):
    """Agent for detecting hallucinated content in translations.

    Checks for:
    - Entity preservation (names, numbers, dates)
    - Factual consistency between source and translation
    - Suspicious length ratios
    - Information additions not present in source

    Example:
        >>> agent = HallucinationAgent(llm_provider)
        >>> errors = await agent.evaluate(task)
        >>> critical_hallucinations = [e for e in errors
        ...     if e.severity == ErrorSeverity.CRITICAL]
    """

    # Thresholds based on research
    LENGTH_RATIO_MIN = 0.4  # Too short may indicate omissions
    LENGTH_RATIO_MAX = 2.0  # Too long may indicate additions/hallucinations
    FACTUAL_CONSISTENCY_THRESHOLD = 0.80  # Below this = likely hallucination

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """Initialize hallucination agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature (default: 0.1 for deterministic)
            max_tokens: Maximum tokens in response
        """
        super().__init__(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def category(self) -> str:
        """Error category for this agent."""
        return "accuracy"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate translation for hallucinations.

        Args:
            task: Translation task to evaluate

        Returns:
            List of error annotations for detected hallucinations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        errors: list[ErrorAnnotation] = []

        try:
            # Check 1: Entity preservation
            entity_errors = await self._check_entity_preservation(task)
            errors.extend(entity_errors)

            # Check 2: Length ratio
            length_errors = self._check_length_ratio(task)
            errors.extend(length_errors)

            # Check 3: LLM-based factual consistency
            consistency_errors = await self._check_factual_consistency(task)
            errors.extend(consistency_errors)

            logger.info(f"HallucinationAgent found {len(errors)} issues")
            return errors

        except Exception as e:
            logger.error(f"Hallucination agent evaluation failed: {e}")
            raise AgentEvaluationError(f"Hallucination detection failed: {e}") from e

    async def _check_entity_preservation(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Check if named entities are preserved in translation.

        Args:
            task: Translation task

        Returns:
            List of errors for missing or hallucinated entities
        """
        prompt = f"""You are an expert at detecting factual consistency in translations.

Your task: Extract named entities (names, numbers, dates, locations) from source and translation,
then check if all source entities are preserved correctly.

SOURCE ({task.source_lang}): {task.source_text}
TRANSLATION ({task.target_lang}): {task.translation}

Instructions:
1. Extract all named entities from SOURCE (names, numbers, dates, locations, organizations)
2. Check if each entity appears in TRANSLATION (allow for translation/transliteration)
3. Identify any entities in TRANSLATION that are NOT in SOURCE (hallucinations)

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "hallucination_entity_missing|hallucination_entity_added|hallucination_entity_modified",
      "severity": "critical|major|minor",
      "description": "Clear explanation of the issue",
      "entity_source": "entity from source (if applicable)",
      "entity_translation": "entity from translation"
    }}
  ]
}}

Rules:
- Missing critical entities (names, amounts) = CRITICAL
- Modified entities = MAJOR
- Missing contextual entities = MINOR
- If all entities preserved correctly, return empty errors array

Output only valid JSON, no explanation."""

        try:
            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Parse response
            response_data = self._parse_json_response(response)
            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                # Estimate location (middle of translation as approximation)
                location_start = len(task.translation) // 2
                location_end = location_start + 10

                errors.append(
                    ErrorAnnotation(
                        category="accuracy",
                        subcategory=error_dict.get("subcategory", "hallucination_entity"),
                        severity=ErrorSeverity(error_dict.get("severity", "major")),
                        location=(location_start, location_end),
                        description=error_dict.get("description", "Entity preservation issue"),
                        suggestion=None,
                    )
                )

            return errors

        except Exception as e:
            logger.warning(f"Entity preservation check failed: {e}")
            return []

    def _check_length_ratio(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Check if translation length is suspiciously different from source.

        Extreme deviations may indicate hallucinations or omissions.

        Args:
            task: Translation task

        Returns:
            List of errors for suspicious length ratios
        """
        source_len = len(task.source_text)
        translation_len = len(task.translation)

        if source_len == 0:
            return []

        ratio = translation_len / source_len

        errors = []

        if ratio > self.LENGTH_RATIO_MAX:
            errors.append(
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="hallucination_length_excessive",
                    severity=ErrorSeverity.MAJOR,
                    location=(0, len(task.translation)),
                    description=(
                        f"Translation is {ratio:.1f}x longer than source "
                        f"({translation_len} vs {source_len} chars). "
                        "This may indicate added/hallucinated content."
                    ),
                    suggestion="Review for unnecessary additions or hallucinated information",
                )
            )
        elif ratio < self.LENGTH_RATIO_MIN:
            errors.append(
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="hallucination_length_insufficient",
                    severity=ErrorSeverity.MAJOR,
                    location=(0, len(task.translation)),
                    description=(
                        f"Translation is {ratio:.1f}x shorter than source "
                        f"({translation_len} vs {source_len} chars). "
                        "This may indicate omitted content."
                    ),
                    suggestion="Review for missing information from source",
                )
            )

        return errors

    async def _check_factual_consistency(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Check factual consistency between source and translation using LLM.

        Args:
            task: Translation task

        Returns:
            List of errors for factual inconsistencies
        """
        prompt = f"""You are an expert at detecting factual hallucinations in translations.

Your task: Identify any information in the translation that is NOT supported by the source text.

SOURCE ({task.source_lang}): {task.source_text}
TRANSLATION ({task.target_lang}): {task.translation}

Instructions:
1. Read the source text carefully
2. Read the translation
3. Identify any claims, facts, or information in the translation that:
   - Are NOT present in the source
   - Contradict the source
   - Add extra context not implied by source
   - Modify factual details (numbers, names, relationships)

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "hallucination_factual|hallucination_addition|hallucination_contradiction",
      "severity": "critical|major|minor",
      "description": "What information was hallucinated and why it's incorrect",
      "hallucinated_content": "The specific text that was hallucinated"
    }}
  ]
}}

Severity guidelines:
- CRITICAL: Completely fabricated facts, wrong numbers/names, contradictions
- MAJOR: Significant additions not in source
- MINOR: Minor embellishments or implied information

If translation is factually consistent with source, return empty errors array.

Output only valid JSON, no explanation."""

        try:
            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Parse response
            response_data = self._parse_json_response(response)
            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                hallucinated_text = error_dict.get("hallucinated_content", "")

                # Try to find location of hallucinated content
                location = self._find_text_location(task.translation, hallucinated_text)

                errors.append(
                    ErrorAnnotation(
                        category="accuracy",
                        subcategory=error_dict.get("subcategory", "hallucination_factual"),
                        severity=ErrorSeverity(error_dict.get("severity", "major")),
                        location=location,
                        description=error_dict.get(
                            "description", "Factual consistency issue detected"
                        ),
                        suggestion="Remove or correct hallucinated information",
                    )
                )

            return errors

        except Exception as e:
            logger.warning(f"Factual consistency check failed: {e}")
            return []

    def _find_text_location(self, text: str, search_text: str) -> tuple[int, int]:
        """Find approximate location of search_text in text.

        Args:
            text: Text to search in
            search_text: Text to search for

        Returns:
            Tuple of (start, end) character positions
        """
        if not search_text:
            return (0, min(20, len(text)))

        # Try exact match
        index = text.find(search_text)
        if index != -1:
            return (index, index + len(search_text))

        # Try partial match (first few words)
        words = search_text.split()[:3]
        for word in words:
            index = text.find(word)
            if index != -1:
                return (index, index + len(word))

        # Fallback to middle of text
        mid = len(text) // 2
        return (mid, mid + min(20, len(text) - mid))

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON dictionary

        Raises:
            AgentParsingError: If parsing fails
        """
        try:
            # Try direct JSON parsing
            return cast(dict[str, Any], json.loads(response))
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(1)))
                except json.JSONDecodeError as e:
                    raise AgentParsingError(f"Failed to parse JSON from code block: {e}") from e

            # Try to find JSON object in response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(0)))
                except json.JSONDecodeError as e:
                    raise AgentParsingError(f"Failed to parse extracted JSON: {e}") from e

            raise AgentParsingError(f"No valid JSON found in response: {response[:200]}")
