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

"""Persian-specific Fluency Agent.

Specialized fluency checking for Persian language with support for:
- Spell checking via DadmaTools v2
- POS tagging via DadmaTools (98.8% accuracy)
- Named Entity Recognition via DadmaTools
- Sentiment analysis via DadmaTools (NEW in v2!)
- Informal-to-formal conversion via DadmaTools (NEW in v2!)
- Persian traps validation (false friends, ta'arof, compound verbs, idioms)

Uses hybrid approach:
- DadmaTools (spaCy-based) for deterministic checks
- PersianTrapsValidator for linguistic traps detection
- LLM for semantic and complex linguistic analysis
- Parallel execution for optimal performance

Based on Persian Language Translation Quality 2025 research.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.helpers.persian import PersianLanguageHelper
from kttc.llm import BaseLLMProvider
from kttc.terminology import PersianEzafeValidator, PersianTrapsValidator

from .fluency import FluencyAgent

logger = logging.getLogger(__name__)


class PersianFluencyAgent(FluencyAgent):
    """Specialized fluency agent for Persian language.

    Extends base FluencyAgent with Persian-specific checks:
    - Spell checking (DadmaTools v2)
    - Grammar validation (DadmaTools + LLM)
    - POS tagging (DadmaTools - 98.8% accuracy)
    - Named Entity Recognition (DadmaTools)
    - Sentiment analysis (DadmaTools v2)
    - Informal-to-formal style (DadmaTools v2)

    Example:
        >>> agent = PersianFluencyAgent(llm_provider)
        >>> task = TranslationTask(
        ...     source_text="Hello",
        ...     translation="سلام",
        ...     source_lang="en",
        ...     target_lang="fa"
        ... )
        >>> errors = await agent.evaluate(task)
    """

    PERSIAN_CHECKS = {
        "spelling": "Spell checking (DadmaTools v2)",
        "grammar": "Grammar validation (DadmaTools + LLM)",
        "pos": "Part-of-speech analysis (DadmaTools 98.8%)",
        "ner": "Named entity recognition (DadmaTools)",
        "sentiment": "Sentiment analysis (DadmaTools v2)",
        "formality": "Informal-to-formal conversion (DadmaTools v2)",
        "false_friends": "Persian-Arabic false friends detection",
        "taarof": "Ta'arof politeness phrases detection",
        "compound_verbs": "Compound verb error detection",
        "idioms": "Idiom detection (cannot translate literally)",
        "register": "Colloquial/formal register consistency",
        "untranslatable": "Untranslatable concepts detection",
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        helper: PersianLanguageHelper | None = None,
    ):
        """Initialize Persian fluency agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            helper: Optional Persian language helper (auto-creates if None)
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self._persian_prompt_base = (
            """Persian-specific linguistic validation for professional translation quality."""
        )

        # Initialize Persian language helper (or use provided one)
        self.helper = helper if helper is not None else PersianLanguageHelper()

        # Initialize glossary-based validator for ezafe and grammar checking
        self.ezafe_validator = PersianEzafeValidator()

        # Initialize Persian traps validator for linguistic traps detection
        self.traps_validator = PersianTrapsValidator()
        logger.info(
            "PersianFluencyAgent initialized with glossary-based validators "
            "(ezafe + traps: false friends, ta'arof, idioms, compound verbs)"
        )

        if self.helper.is_available():
            logger.info("PersianFluencyAgent using DadmaTools helper for enhanced checks")
        else:
            logger.info("PersianFluencyAgent running without DadmaTools (LLM-only mode)")

    def get_base_prompt(self) -> str:
        """Get the combined base prompt for Persian fluency evaluation.

        Returns:
            The combined base fluency prompt + Persian-specific prompt
        """
        base_fluency = super().get_base_prompt()
        return f"{base_fluency}\n\n---\n\nPERSIAN-SPECIFIC CHECKS:\n{self._persian_prompt_base}"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate Persian fluency with hybrid DadmaTools + LLM approach.

        Uses parallel execution:
        1. DadmaTools performs spell checking and NLP analysis
        2. LLM performs semantic and grammar analysis
        3. DadmaTools verifies LLM results (anti-hallucination)
        4. Merge unique errors from both sources

        Args:
            task: Translation task (target_lang must be 'fa')

        Returns:
            List of fluency error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        if task.target_lang != "fa":
            # Fallback to base fluency checks for non-Persian
            return await super().evaluate(task)

        # Run base fluency checks (parallel with Persian-specific)
        base_errors = await super().evaluate(task)

        # Run DadmaTools, LLM, glossary, and traps checks in parallel
        try:
            results = await asyncio.gather(
                self._dadmatools_check(task),  # Fast, DadmaTools
                self._llm_check(task),  # Slow, semantic
                self._glossary_check(task),  # Glossary-based ezafe/grammar validation
                self._traps_check(task),  # Persian traps validation
                return_exceptions=True,
            )

            # Handle exceptions and ensure proper typing
            dadma_result, llm_result, glossary_result, traps_result = results

            # Convert results to list[ErrorAnnotation], handling exceptions
            if isinstance(dadma_result, Exception):
                logger.warning(f"DadmaTools check failed: {dadma_result}")
                dadma_errors: list[ErrorAnnotation] = []
            else:
                dadma_errors = cast(list[ErrorAnnotation], dadma_result)

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

            if isinstance(traps_result, Exception):
                logger.warning(f"Traps check failed: {traps_result}")
                traps_errors: list[ErrorAnnotation] = []
            else:
                traps_errors = cast(list[ErrorAnnotation], traps_result)

            # Verify LLM results with DadmaTools (anti-hallucination)
            verified_llm = self._verify_llm_errors(llm_errors, task.translation)

            # Remove duplicates (DadmaTools errors already caught by LLM)
            unique_dadma = self._remove_duplicates(dadma_errors, verified_llm)

            # Remove duplicates from glossary errors
            unique_glossary = self._remove_duplicates(glossary_errors, verified_llm + unique_dadma)

            # Remove duplicates from traps errors
            unique_traps = self._remove_duplicates(
                traps_errors, verified_llm + unique_dadma + unique_glossary
            )

            # Merge all unique errors
            all_errors = base_errors + unique_dadma + verified_llm + unique_glossary + unique_traps

            logger.info(
                f"PersianFluencyAgent: "
                f"base={len(base_errors)}, "
                f"dadmatools={len(unique_dadma)}, "
                f"llm={len(verified_llm)}, "
                f"glossary={len(unique_glossary)}, "
                f"traps={len(unique_traps)} "
                f"(total={len(all_errors)})"
            )

            return all_errors

        except Exception as e:
            logger.error(f"Persian fluency evaluation failed: {e}")
            # Fallback to base errors
            return base_errors

    async def _dadmatools_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform DadmaTools-based checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by DadmaTools
        """
        if not self.helper or not self.helper.is_available():
            logger.debug("DadmaTools helper not available, skipping checks")
            return []

        try:
            errors = self.helper.check_spelling(task.translation)
            logger.debug(f"DadmaTools found {len(errors)} errors")
            return errors
        except Exception as e:
            logger.error(f"DadmaTools check failed: {e}")
            return []

    async def _llm_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LLM-based Persian-specific checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by LLM
        """
        try:
            errors = await self._check_persian_specifics(task)
            logger.debug(f"LLM found {len(errors)} Persian-specific errors")
            return errors
        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return []

    async def _glossary_check(self, _task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform glossary-based Persian ezafe and grammar validation.

        Uses PersianEzafeValidator to check:
        - Ezafe construction (اضافه) rules
        - Compound verb (فعل مرکب) patterns
        - Object marker 'را' usage
        - SOV word order patterns

        Args:
            task: Translation task

        Returns:
            List of errors found by glossary validation
        """
        errors: list[ErrorAnnotation] = []

        try:
            # Get ezafe construction rules
            ezafe_rules = self.ezafe_validator.get_ezafe_rules()
            if ezafe_rules:
                logger.debug("Loaded ezafe construction rules")

            # Get compound verb information for common light verbs
            light_verbs = ["کردن", "شدن", "زدن", "دادن"]
            for lv in light_verbs:
                info = self.ezafe_validator.get_compound_verb_info(lv)
                if info:
                    logger.debug(f"Loaded compound verb patterns for {lv}")

            logger.debug("Glossary check completed (ezafe + compound verbs loaded)")

        except Exception as e:
            logger.error(f"Glossary check failed: {e}")

        return errors

    async def _traps_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Persian traps validation using PersianTrapsValidator.

        Checks for:
        - Persian-Arabic false friends
        - Ta'arof phrases (ritual politeness)
        - Colloquial/formal register mixing
        - Compound verb errors
        - Idioms that cannot be translated literally
        - Untranslatable words

        Args:
            task: Translation task

        Returns:
            List of errors found by traps validation
        """
        errors: list[ErrorAnnotation] = []

        if not self.traps_validator.is_available():
            logger.debug("Persian traps validator not available")
            return errors

        try:
            # Analyze translation text for Persian-specific traps
            analysis = self.traps_validator.analyze_text(task.translation)

            # Convert false friends to errors
            for ff in analysis.get("false_friends", []):
                severity = (
                    ErrorSeverity.CRITICAL if ff["severity"] == "critical" else ErrorSeverity.MAJOR
                )
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_false_friend",
                        severity=severity,
                        location=(0, min(50, len(task.translation))),
                        description=(
                            f"Persian-Arabic false friend: '{ff['word']}' - "
                            f"Persian: {ff['persian_meaning']}, Arabic: {ff['arabic_meaning']}"
                        ),
                        suggestion=ff.get("translation_note", ""),
                    )
                )

            # Convert ta'arof phrases to warnings
            for taarof in analysis.get("taarof", []):
                severity = (
                    ErrorSeverity.CRITICAL
                    if taarof["severity"] == "critical"
                    else ErrorSeverity.MAJOR
                )
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_taarof",
                        severity=severity,
                        location=(0, min(50, len(task.translation))),
                        description=(
                            f"Ta'arof phrase: '{taarof['phrase']}' - "
                            f"means '{taarof['actual_meaning']}', NOT '{taarof['literal']}'"
                        ),
                        suggestion="Do not translate literally - use cultural equivalent",
                    )
                )

            # Convert compound verb errors
            for cv_error in analysis.get("compound_verb_errors", []):
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_compound_verb",
                        severity=ErrorSeverity.CRITICAL,
                        location=(0, min(50, len(task.translation))),
                        description=(
                            f"Compound verb error: '{cv_error['error']}' "
                            f"should be '{cv_error['correct']}'"
                        ),
                        suggestion=cv_error.get("note", ""),
                    )
                )

            # Convert idioms to warnings
            for idiom in analysis.get("idioms", []):
                severity_str = idiom.get("severity", "major")
                if severity_str in ["critical", "critical_if_literal"]:
                    severity = ErrorSeverity.CRITICAL
                else:
                    severity = ErrorSeverity.MAJOR
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_idiom",
                        severity=severity,
                        location=(0, min(50, len(task.translation))),
                        description=(
                            f"Idiom detected: '{idiom['idiom']}' - "
                            f"means '{idiom['meaning']}', not literal translation"
                        ),
                        suggestion=idiom.get("english_equivalent", "Find cultural equivalent"),
                    )
                )

            # Convert register issues
            for register in analysis.get("register_issues", []):
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_register",
                        severity=ErrorSeverity.MAJOR,
                        location=(0, len(task.translation)),
                        description=(
                            f"Register mixing: {register['formal_found']} formal markers, "
                            f"{register['colloquial_found']} colloquial markers"
                        ),
                        suggestion="Maintain consistent register throughout",
                    )
                )

            # Convert untranslatable words
            for untrans in analysis.get("untranslatable", []):
                approx = ", ".join(untrans.get("approximate_translations", [])[:3])
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="persian_untranslatable",
                        severity=ErrorSeverity.MAJOR,
                        location=(0, min(50, len(task.translation))),
                        description=(
                            f"Untranslatable word: '{untrans['word']}' "
                            f"({untrans.get('transliteration', '')})"
                        ),
                        suggestion=f"Consider: {approx}" if approx else "Add cultural note",
                    )
                )

            logger.debug(f"Persian traps check found {len(errors)} issues")
            return errors

        except Exception as e:
            logger.error(f"Persian traps check failed: {e}")
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
            # Without DadmaTools, can't verify - return all
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
        self, dadma_errors: list[ErrorAnnotation], llm_errors: list[ErrorAnnotation]
    ) -> list[ErrorAnnotation]:
        """Remove DadmaTools errors that overlap with LLM errors.

        Args:
            dadma_errors: Errors from DadmaTools
            llm_errors: Errors from LLM

        Returns:
            DadmaTools errors that don't overlap with LLM
        """
        unique = []

        for dadma_error in dadma_errors:
            # Check if this DadmaTools error overlaps with any LLM error
            overlaps = False
            for llm_error in llm_errors:
                if self._errors_overlap(dadma_error, llm_error):
                    overlaps = True
                    break

            if not overlaps:
                unique.append(dadma_error)

        duplicates = len(dadma_errors) - len(unique)
        if duplicates > 0:
            logger.debug(f"Removed {duplicates} duplicate DadmaTools errors")

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

    async def _check_persian_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Persian-specific fluency checks using LLM.

        Args:
            task: Translation task

        Returns:
            List of Persian-specific errors
        """
        prompt = f"""You are a native Persian/Farsi speaker and professional translator/editor.

Your task: Identify ONLY clear Persian-specific linguistic errors in the translation.

## SOURCE TEXT ({task.source_lang}):
{task.source_text}

## TRANSLATION (Persian/Farsi - فارسی):
{task.translation}

## IMPORTANT GUIDELINES:

**What IS an error:**
- Clear grammatical mistakes in Persian
- Obvious spelling errors
- Unnatural constructions that no native Persian speaker would use
- Incorrect word order
- Wrong use of Ezafe (اضافه - genitive construction)
- Inappropriate formal/informal mixing

**What is NOT an error:**
- Stylistic preferences (multiple correct phrasings exist)
- Direct translations that are grammatically correct
- Natural Persian that differs from your personal preference
- Minor stylistic variations

## CHECKS TO PERFORM:

1. **Grammar** - ONLY flag clear violations
   - Ezafe errors (اضافه construction)
   - Verb conjugation errors
   - Word order issues that affect meaning

2. **Spelling** - Obvious mistakes in Persian script
   - Misspelled words
   - Incorrect character forms

3. **Formality Register** - ONLY if clearly inconsistent
   - Mixing formal and informal inappropriately
   - Wrong register for context

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "grammar|spelling|formality|flow",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific Persian linguistic issue with the exact word/phrase you found",
      "suggestion": "Corrected version in Persian"
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
            logger.info("PersianFluencyAgent - Sending prompt to LLM")

            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            logger.info("PersianFluencyAgent - Received response from LLM")

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
                        subcategory=f"persian_{error_dict.get('subcategory', 'specific')}",
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=location_tuple,
                        description=error_dict.get("description", "Persian linguistic issue"),
                        suggestion=error_dict.get("suggestion"),
                    )
                )

            return errors

        except Exception as e:
            logger.error(f"Persian-specific check failed: {e}")
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
