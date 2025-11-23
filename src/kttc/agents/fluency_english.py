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

"""English-specific Fluency Agent.

Specialized fluency checking for English language with support for:
- Grammar checking via LanguageTool
- Subject-verb agreement
- Article usage (a/an/the)
- Spelling and punctuation
- Tense consistency

Uses hybrid approach:
- LanguageTool helper for deterministic grammar/spelling checks
- LLM for semantic and complex linguistic analysis
- Parallel execution for optimal performance

Based on English Language Translation Quality 2025 best practices.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.helpers.english import EnglishLanguageHelper
from kttc.llm import BaseLLMProvider
from kttc.terminology.english_traps import EnglishTrapsValidator

from .fluency import FluencyAgent

logger = logging.getLogger(__name__)


class EnglishFluencyAgent(FluencyAgent):
    """Specialized fluency agent for English language.

    Extends base FluencyAgent with English-specific checks:
    - Grammar (subject-verb agreement, articles, prepositions)
    - Spelling and punctuation
    - Tense consistency
    - Register/formality consistency

    Example:
        >>> agent = EnglishFluencyAgent(llm_provider)
        >>> task = TranslationTask(
        ...     source_text="Привет",
        ...     translation="Hello",
        ...     source_lang="ru",
        ...     target_lang="en"
        ... )
        >>> errors = await agent.evaluate(task)
    """

    ENGLISH_CHECKS = {
        "grammar": "Grammar validation (articles, prepositions, agreement)",
        "spelling": "Spelling and punctuation",
        "tense": "Tense consistency",
        "register": "Formality register consistency",
        "homophones": "Homophone confusion detection (their/there/they're, etc.)",
        "phrasal_verbs": "Phrasal verb identification (cannot translate literally)",
        "heteronyms": "Heteronym context validation (lead, read, etc.)",
        "adjective_order": "Adjective order validation (opinion-size-age-color...)",
        "prepositions": "Preposition collocation checking (depend on, interested in)",
        "idioms": "Idiom detection (piece of cake, break a leg, etc.)",
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        helper: EnglishLanguageHelper | None = None,
        traps_validator: EnglishTrapsValidator | None = None,
    ):
        """Initialize English fluency agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            helper: Optional English language helper for LanguageTool checks (auto-creates if None)
            traps_validator: Optional English traps validator (auto-creates if None)
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self._english_prompt_base = (
            """English-specific linguistic validation for professional translation quality."""
        )

        # Initialize LanguageTool helper (or use provided one)
        self.helper = helper if helper is not None else EnglishLanguageHelper()

        # Initialize English traps validator (homophones, phrasal verbs, idioms, etc.)
        if traps_validator is not None:
            self.traps_validator = traps_validator
        else:
            self.traps_validator = EnglishTrapsValidator()

        if self.helper.is_available():
            logger.info("EnglishFluencyAgent using LanguageTool helper for enhanced checks")
        else:
            logger.info("EnglishFluencyAgent running without LanguageTool (LLM-only mode)")

        if self.traps_validator.is_available():
            logger.info("EnglishFluencyAgent using EnglishTrapsValidator for trap detection")
        else:
            logger.info("EnglishFluencyAgent running without trap glossaries")

    def get_base_prompt(self) -> str:
        """Get the combined base prompt for English fluency evaluation.

        Returns:
            The combined base fluency prompt + English-specific prompt
        """
        base_fluency = super().get_base_prompt()
        return f"{base_fluency}\n\n---\n\nENGLISH-SPECIFIC CHECKS:\n{self._english_prompt_base}"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate English fluency with hybrid LanguageTool + LLM + Traps approach.

        Uses parallel execution:
        1. LanguageTool performs deterministic grammar/spelling checks
        2. LLM performs semantic and complex linguistic analysis
        3. EnglishTrapsValidator detects homophones, phrasal verbs, idioms, etc.
        4. LanguageTool verifies LLM results (anti-hallucination)
        5. Merge unique errors from all sources

        Args:
            task: Translation task (target_lang must be 'en')

        Returns:
            List of fluency error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        if task.target_lang != "en":
            # Fallback to base fluency checks for non-English
            return await super().evaluate(task)

        # Run base fluency checks (parallel with English-specific)
        base_errors = await super().evaluate(task)

        # Run LanguageTool, LLM, and Traps checks in parallel
        try:
            results = await asyncio.gather(
                self._languagetool_check(task),  # Fast, deterministic
                self._llm_check(task),  # Slow, semantic
                self._traps_check(task),  # Fast, glossary-based
                return_exceptions=True,
            )

            # Handle exceptions and ensure proper typing
            lt_result, llm_result, traps_result = results

            # Convert results to list[ErrorAnnotation], handling exceptions
            if isinstance(lt_result, Exception):
                logger.warning(f"LanguageTool check failed: {lt_result}")
                lt_errors: list[ErrorAnnotation] = []
            else:
                lt_errors = cast(list[ErrorAnnotation], lt_result)

            if isinstance(llm_result, Exception):
                logger.warning(f"LLM check failed: {llm_result}")
                llm_errors: list[ErrorAnnotation] = []
            else:
                llm_errors = cast(list[ErrorAnnotation], llm_result)

            if isinstance(traps_result, Exception):
                logger.warning(f"Traps check failed: {traps_result}")
                traps_errors: list[ErrorAnnotation] = []
            else:
                traps_errors = cast(list[ErrorAnnotation], traps_result)

            # Verify LLM results with LanguageTool (anti-hallucination)
            verified_llm = self._verify_llm_errors(llm_errors, task.translation)

            # Remove duplicates (LanguageTool errors already caught by LLM)
            unique_lt = self._remove_duplicates(lt_errors, verified_llm)

            # Merge all unique errors
            all_errors = base_errors + unique_lt + verified_llm + traps_errors

            logger.info(
                f"EnglishFluencyAgent: "
                f"base={len(base_errors)}, "
                f"languagetool={len(unique_lt)}, "
                f"llm={len(verified_llm)}, "
                f"traps={len(traps_errors)} "
                f"(total={len(all_errors)})"
            )

            return all_errors

        except Exception as e:
            logger.error(f"English fluency evaluation failed: {e}")
            # Fallback to base errors
            return base_errors

    async def _languagetool_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LanguageTool-based grammar checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by LanguageTool
        """
        if not self.helper or not self.helper.is_available():
            logger.debug("LanguageTool helper not available, skipping checks")
            return []

        try:
            errors = self.helper.check_grammar(task.translation)
            logger.debug(f"LanguageTool found {len(errors)} grammar errors")
            return errors
        except Exception as e:
            logger.error(f"LanguageTool check failed: {e}")
            return []

    async def _llm_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LLM-based English-specific checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by LLM
        """
        try:
            errors = await self._check_english_specifics(task)
            logger.debug(f"LLM found {len(errors)} English-specific errors")
            return errors
        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return []

    async def _traps_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform English traps checks (homophones, phrasal verbs, idioms, etc.).

        Uses glossary-based detection for:
        - Homophone errors (their/there/they're, your/you're, etc.)
        - Phrasal verbs (cannot be translated literally)
        - Heteronyms (same spelling, different pronunciation)
        - Adjective order violations
        - Preposition errors
        - Idioms (cannot be translated literally)

        Args:
            task: Translation task

        Returns:
            List of errors found by EnglishTrapsValidator
        """
        if not self.traps_validator or not self.traps_validator.is_available():
            logger.debug("EnglishTrapsValidator not available, skipping traps checks")
            return []

        try:
            errors: list[ErrorAnnotation] = []
            text = task.translation

            # Check homophones (high priority - common errors)
            homophone_errors = self.traps_validator.find_homophones_in_text(text)
            for h in homophone_errors:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="english_homophone_error",
                        severity=ErrorSeverity.MAJOR,
                        location=h.get("position", (0, 10)),
                        description=h.get("suggestion", "Possible homophone error"),
                        suggestion=(
                            f"Use '{h.get('correct_word', '')}' "
                            f"instead of '{h.get('wrong_word', '')}'"
                        ),
                    )
                )

            # Check phrasal verbs (critical for translation)
            phrasal_verbs = self.traps_validator.find_phrasal_verbs_in_text(text)
            for pv in phrasal_verbs:
                meanings = pv.get("meanings", [])
                meaning_str = "; ".join([m.get("meaning", "") for m in meanings[:2]])
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="english_phrasal_verb",
                        severity=ErrorSeverity.MINOR,  # Info level - not an error per se
                        location=(0, min(50, len(text))),
                        description=(
                            f"Phrasal verb '{pv.get('phrasal_verb', '')}' "
                            "- DO NOT translate literally"
                        ),
                        suggestion=f"Meanings: {meaning_str}",
                    )
                )

            # Check adjective order
            adj_errors = self.traps_validator.check_adjective_order(text)
            for adj in adj_errors:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="english_adjective_order",
                        severity=ErrorSeverity.MAJOR,
                        location=adj.get("position", (0, 10)),
                        description=f"Adjective order violation: {adj.get('violation', '')}",
                        suggestion=f"Correct order: {adj.get('correct_order', '')}",
                    )
                )

            # Check preposition errors
            prep_errors = self.traps_validator.find_preposition_errors(text)
            for prep in prep_errors:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="english_preposition_error",
                        severity=ErrorSeverity.MAJOR,
                        location=prep.get("position", (0, 10)),
                        description=f"Preposition error: '{prep.get('found_text', '')}'",
                        suggestion=f"Use: {prep.get('correction', '')}",
                    )
                )

            # Check idioms (informational - for translation context)
            idioms = self.traps_validator.find_idioms_in_text(text)
            for idiom in idioms:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="english_idiom",
                        severity=ErrorSeverity.MINOR,  # Info level
                        location=(0, min(50, len(text))),
                        description=(
                            f"Idiom detected: '{idiom.get('idiom', '')}' "
                            "- DO NOT translate literally"
                        ),
                        suggestion=f"Meaning: {idiom.get('meaning', '')}",
                    )
                )

            logger.debug(f"EnglishTrapsValidator found {len(errors)} issues")
            return errors

        except Exception as e:
            logger.error(f"Traps check failed: {e}")
            return []

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
            # Without LanguageTool, can't verify - return all
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
        self, lt_errors: list[ErrorAnnotation], llm_errors: list[ErrorAnnotation]
    ) -> list[ErrorAnnotation]:
        """Remove LanguageTool errors that overlap with LLM errors.

        Args:
            lt_errors: Errors from LanguageTool
            llm_errors: Errors from LLM

        Returns:
            LanguageTool errors that don't overlap with LLM
        """
        unique = []

        for lt_error in lt_errors:
            # Check if this LanguageTool error overlaps with any LLM error
            overlaps = False
            for llm_error in llm_errors:
                if self._errors_overlap(lt_error, llm_error):
                    overlaps = True
                    break

            if not overlaps:
                unique.append(lt_error)

        duplicates = len(lt_errors) - len(unique)
        if duplicates > 0:
            logger.debug(f"Removed {duplicates} duplicate LanguageTool errors")

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

    async def _check_english_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform English-specific fluency checks.

        Args:
            task: Translation task

        Returns:
            List of English-specific errors
        """
        prompt = f"""You are a native English speaker and professional translator/editor.

Your task: Identify ONLY clear English-specific linguistic errors in the translation.

## SOURCE TEXT ({task.source_lang}):
{task.source_text}

## TRANSLATION (English):
{task.translation}

## IMPORTANT GUIDELINES:

**What IS an error:**
- Clear grammatical mistakes (subject-verb agreement, article errors)
- Obvious spelling or punctuation errors
- Unnatural constructions that no native English speaker would use
- Incorrect tense that changes meaning from source

**What is NOT an error:**
- Stylistic preferences (multiple correct phrasings exist)
- Direct translations that are grammatically correct
- Natural English that differs from your personal preference
- Minor stylistic variations

## CHECKS TO PERFORM:

1. **Grammar** - ONLY flag clear violations
   - Subject-verb agreement errors
   - Article usage (a/an/the) errors
   - Preposition errors

2. **Spelling & Punctuation** - Obvious mistakes only
   - Misspelled words
   - Incorrect punctuation that affects meaning

3. **Tense Consistency** - Consider source text context
   - Check if tense matches the source meaning
   - Inconsistent tense usage within translation

4. **Register** - ONLY if clearly inconsistent
   - Mixing formal and informal language inappropriately

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "grammar|spelling|tense|register",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific English linguistic issue with the exact word/phrase you found",
      "suggestion": "Corrected version in English"
    }}
  ]
}}

Rules:
- CONSERVATIVE: Only report clear, unambiguous errors
- VERIFY: Ensure the word/phrase you mention actually exists in the text at the specified position
- CONTEXT: Consider the source text when evaluating tense and meaning
- If the translation is natural and grammatically correct, return empty errors array
- Provide accurate character positions (0-indexed, use Python string slicing logic)

Output only valid JSON, no explanation."""

        try:
            logger.info("EnglishFluencyAgent - Sending prompt to LLM")

            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            logger.info("EnglishFluencyAgent - Received response from LLM")

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
                        subcategory=f"english_{error_dict.get('subcategory', 'specific')}",
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=location_tuple,
                        description=error_dict.get("description", "English linguistic issue"),
                        suggestion=error_dict.get("suggestion"),
                    )
                )

            return errors

        except Exception as e:
            logger.error(f"English-specific check failed: {e}")
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
