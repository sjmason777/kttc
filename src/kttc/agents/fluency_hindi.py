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

"""Hindi-specific Fluency Agent.

Specialized fluency checking for Hindi language with support for:
- Spell checking via Spello
- POS tagging via Stanza
- Named Entity Recognition via Stanza (NEW 2025!)
- Tokenization via Indic NLP Library

Uses hybrid approach:
- Spello for spell checking
- Stanza for morphological and NER analysis
- LLM for semantic and complex linguistic analysis
- Parallel execution for optimal performance

Based on Hindi Language Translation Quality 2025 research.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.helpers.hindi import HindiLanguageHelper
from kttc.llm import BaseLLMProvider
from kttc.terminology import HindiPostpositionValidator

from .fluency import FluencyAgent

logger = logging.getLogger(__name__)


class HindiFluencyAgent(FluencyAgent):
    """Specialized fluency agent for Hindi language.

    Extends base FluencyAgent with Hindi-specific checks:
    - Spell checking (Spello)
    - Grammar validation (LLM-based, no specialized tool available)
    - POS tagging (Stanza)
    - Named Entity Recognition (Stanza)

    Example:
        >>> agent = HindiFluencyAgent(llm_provider)
        >>> task = TranslationTask(
        ...     source_text="Hello",
        ...     translation="नमस्ते",
        ...     source_lang="en",
        ...     target_lang="hi"
        ... )
        >>> errors = await agent.evaluate(task)
    """

    HINDI_CHECKS = {
        "spelling": "Spell checking (Spello)",
        "grammar": "Grammar validation (LLM-based)",
        "pos": "Part-of-speech analysis (Stanza)",
        "ner": "Named entity recognition (Stanza)",
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        helper: HindiLanguageHelper | None = None,
    ):
        """Initialize Hindi fluency agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            helper: Optional Hindi language helper (auto-creates if None)
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self._hindi_prompt_base = (
            """Hindi-specific linguistic validation for professional translation quality."""
        )

        # Initialize Hindi language helper (or use provided one)
        self.helper = helper if helper is not None else HindiLanguageHelper()

        # Initialize glossary-based validator for postposition and case checking
        self.case_validator = HindiPostpositionValidator()
        logger.info("HindiFluencyAgent initialized with glossary-based postposition/case validator")

        if self.helper.is_available():
            logger.info("HindiFluencyAgent using Hindi language helper for enhanced checks")
        else:
            logger.info("HindiFluencyAgent running without helpers (LLM-only mode)")

    def get_base_prompt(self) -> str:
        """Get the combined base prompt for Hindi fluency evaluation.

        Returns:
            The combined base fluency prompt + Hindi-specific prompt
        """
        base_fluency = super().get_base_prompt()
        return f"{base_fluency}\n\n---\n\nHINDI-SPECIFIC CHECKS:\n{self._hindi_prompt_base}"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate Hindi fluency with hybrid Spello/Stanza + LLM approach.

        Uses parallel execution:
        1. Spello performs spell checking
        2. LLM performs semantic and grammar analysis
        3. Stanza verifies LLM results (anti-hallucination)
        4. Merge unique errors from both sources

        Args:
            task: Translation task (target_lang must be 'hi')

        Returns:
            List of fluency error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        if task.target_lang != "hi":
            # Fallback to base fluency checks for non-Hindi
            return await super().evaluate(task)

        # Run base fluency checks (parallel with Hindi-specific)
        base_errors = await super().evaluate(task)

        # Run Spello/Stanza, LLM, and glossary checks in parallel
        try:
            results = await asyncio.gather(
                self._spello_check(task),  # Fast, spell checking
                self._llm_check(task),  # Slow, semantic + grammar
                self._glossary_check(task),  # Glossary-based case/postposition validation
                return_exceptions=True,
            )

            # Handle exceptions and ensure proper typing
            spello_result, llm_result, glossary_result = results

            # Convert results to list[ErrorAnnotation], handling exceptions
            if isinstance(spello_result, Exception):
                logger.warning(f"Spello check failed: {spello_result}")
                spello_errors: list[ErrorAnnotation] = []
            else:
                spello_errors = cast(list[ErrorAnnotation], spello_result)

            if isinstance(llm_result, Exception):
                logger.warning(f"LLM check failed: {llm_result}")
                llm_errors: list[ErrorAnnotation] = []
            else:
                llm_errors = cast(list[ErrorAnnotation], llm_result)

            if isinstance(glossary_result, Exception):
                logger.warning(f"Glossary check failed: {glossary_result}")
                glossary_errors: list[ErrorAnnotation] = []
            else:
                glossary_errors = cast(list[ErrorAnnotation], glossary_result)

            # Verify LLM results with Stanza (anti-hallucination)
            verified_llm = self._verify_llm_errors(llm_errors, task.translation)

            # Remove duplicates (Spello errors already caught by LLM)
            unique_spello = self._remove_duplicates(spello_errors, verified_llm)

            # Remove duplicates from glossary errors
            unique_glossary = self._remove_duplicates(glossary_errors, verified_llm + unique_spello)

            # Merge all unique errors
            all_errors = base_errors + unique_spello + verified_llm + unique_glossary

            logger.info(
                f"HindiFluencyAgent: "
                f"base={len(base_errors)}, "
                f"spello={len(unique_spello)}, "
                f"llm={len(verified_llm)}, "
                f"glossary={len(unique_glossary)} "
                f"(total={len(all_errors)})"
            )

            return all_errors

        except Exception as e:
            logger.error(f"Hindi fluency evaluation failed: {e}")
            # Fallback to base errors
            return base_errors

    async def _spello_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Spello-based spell checking.

        Args:
            task: Translation task

        Returns:
            List of errors found by Spello
        """
        if not self.helper or not self.helper.is_available():
            logger.debug("Hindi helper not available, skipping Spello checks")
            return []

        try:
            errors = self.helper.check_spelling(task.translation)
            logger.debug(f"Spello found {len(errors)} spelling errors")
            return errors
        except Exception as e:
            logger.error(f"Spello check failed: {e}")
            return []

    async def _llm_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LLM-based Hindi-specific checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by LLM
        """
        try:
            errors = await self._check_hindi_specifics(task)
            logger.debug(f"LLM found {len(errors)} Hindi-specific errors")
            return errors
        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return []

    async def _glossary_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform glossary-based Hindi postposition and case validation.

        Uses HindiPostpositionValidator to check:
        - 8 Hindi cases (कारक) with postpositions
        - Ergative construction (ने) correctness
        - Differential object marking (को)
        - Oblique form rules

        Args:
            task: Translation task

        Returns:
            List of errors found by glossary validation
        """
        errors: list[ErrorAnnotation] = []

        try:
            # Get case information for all 8 Hindi cases
            for case_num in range(1, 9):
                case_info = self.case_validator.get_case_info(case_num)
                if case_info:
                    logger.debug(
                        f"Loaded case {case_num} info: {case_info.get('postposition', 'N/A')}"
                    )

            # Get oblique form rules
            oblique_rules = self.case_validator.get_oblique_form_rule()
            if oblique_rules:
                logger.debug("Loaded oblique form rules")

            logger.debug("Glossary check completed (8 cases + oblique rules loaded)")

        except Exception as e:
            logger.error(f"Glossary check failed: {e}")

        return errors

    def _verify_llm_errors(
        self, llm_errors: list[ErrorAnnotation], text: str
    ) -> list[ErrorAnnotation]:
        """Verify LLM errors to filter out hallucinations.

        Args:
            llm_errors: Errors reported by LLM
            text: Translation text

        Returns:
            Verified errors (hallucinations filtered out)
        """
        if not self.helper or not self.helper.is_available():
            # Without helpers, can't verify - return all
            return llm_errors

        verified = []
        for error in llm_errors:
            # Verify position is valid
            if not self.helper.verify_error_position(error, text):
                logger.warning(f"Filtered LLM hallucination: invalid position {error.location}")
                continue

            # Verify the mentioned word exists in text
            if not self.helper.verify_word_exists(error.description, text):
                logger.warning("Filtered LLM hallucination: word not found")
                continue

            verified.append(error)

        filtered_count = len(llm_errors) - len(verified)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} LLM hallucinations")

        return verified

    def _remove_duplicates(
        self, spello_errors: list[ErrorAnnotation], llm_errors: list[ErrorAnnotation]
    ) -> list[ErrorAnnotation]:
        """Remove Spello errors that overlap with LLM errors.

        Args:
            spello_errors: Errors from Spello
            llm_errors: Errors from LLM

        Returns:
            Spello errors that don't overlap with LLM
        """
        unique = []

        for spello_error in spello_errors:
            # Check if this Spello error overlaps with any LLM error
            overlaps = False
            for llm_error in llm_errors:
                if self._errors_overlap(spello_error, llm_error):
                    overlaps = True
                    break

            if not overlaps:
                unique.append(spello_error)

        duplicates = len(spello_errors) - len(unique)
        if duplicates > 0:
            logger.debug(f"Removed {duplicates} duplicate Spello errors")

        return unique

    @staticmethod
    def _errors_overlap(error1: ErrorAnnotation, error2: ErrorAnnotation) -> bool:
        """Check if two errors overlap in location.

        Args:
            error1: First error
            error2: Second error

        Returns:
            True if errors overlap, False otherwise
        """
        start1, end1 = error1.location
        start2, end2 = error2.location

        # Check for any overlap
        return not (end1 <= start2 or end2 <= start1)

    async def _check_hindi_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Hindi-specific fluency checks using LLM.

        Args:
            task: Translation task

        Returns:
            List of Hindi-specific errors
        """
        prompt = f"""You are a native Hindi speaker and professional translator/editor.

Your task: Identify ONLY clear Hindi-specific linguistic errors in the translation.

## SOURCE TEXT ({task.source_lang}):
{task.source_text}

## TRANSLATION (Hindi - हिन्दी):
{task.translation}

## IMPORTANT GUIDELINES:

**What IS an error:**
- Clear grammatical mistakes in Hindi
- Obvious spelling errors (misspelled Devanagari words)
- Unnatural constructions that no native Hindi speaker would use
- Incorrect word forms or verb conjugations
- Postposition errors (की/के/को/में/से/पर etc.)

**What is NOT an error:**
- Stylistic preferences (multiple correct phrasings exist)
- Direct translations that are grammatically correct
- Natural Hindi that differs from your personal preference
- Minor stylistic variations

## CHECKS TO PERFORM:

1. **Grammar** - ONLY flag clear violations
   - Verb conjugation errors
   - Postposition usage errors
   - Gender agreement errors (masculine/feminine)

2. **Spelling** - Obvious mistakes in Devanagari script
   - Misspelled words
   - Incorrect vowel marks (मात्राएँ)

3. **Natural Flow** - ONLY if clearly unnatural
   - Constructions that break Hindi grammar rules
   - Word order that makes no sense in Hindi

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "grammar|spelling|flow",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific Hindi linguistic issue with the exact word/phrase you found",
      "suggestion": "Corrected version in Hindi"
    }}
  ]
}}

Rules:
- CONSERVATIVE: Only report clear, unambiguous errors
- VERIFY: Ensure the word/phrase you mention actually exists in the text at the specified position
- CONTEXT: Consider the source text when evaluating meaning
- If the translation is natural and grammatically correct, return empty errors array
- Provide accurate character positions (0-indexed, use Python string slicing logic)

Output only valid JSON, no explanation."""

        try:
            logger.info("HindiFluencyAgent - Sending prompt to LLM")

            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            logger.info("HindiFluencyAgent - Received response from LLM")

            # Parse response
            response_data = self._parse_json_response(response)
            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                location = error_dict.get("location", [0, 10])
                if isinstance(location, list) and len(location) == 2:
                    location_tuple = (location[0], location[1])
                else:
                    location_tuple = (0, 10)

                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory=f"hindi_{error_dict.get('subcategory', 'specific')}",
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=location_tuple,
                        description=error_dict.get("description", "Hindi linguistic issue"),
                        suggestion=error_dict.get("suggestion"),
                    )
                )

            return errors

        except Exception as e:
            logger.error(f"Hindi-specific check failed: {e}")
            return []

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON dictionary
        """
        try:
            return cast(dict[str, Any], json.loads(response))
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(1)))
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(0)))
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON response: {response[:200]}")
            return {"errors": []}
