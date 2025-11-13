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

"""Russian-specific Fluency Agent.

Specialized fluency checking for Russian language with support for:
- Case agreement (падежное согласование)
- Aspect usage (совершенный/несовершенный вид)
- Word order validation
- Particle usage (же, ли, бы)
- Register/formality checking

Uses hybrid approach:
- MAWO NLP helper (mawo-pymorphy3 + mawo-razdel) for deterministic grammar checks
- LLM for semantic and complex linguistic analysis
- Parallel execution for optimal performance

Based on Russian Language Translation Quality 2025 research.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.helpers.russian import RussianLanguageHelper
from kttc.llm import BaseLLMProvider

from .fluency import FluencyAgent

logger = logging.getLogger(__name__)


class RussianFluencyAgent(FluencyAgent):
    """Specialized fluency agent for Russian language.

    Extends base FluencyAgent with Russian-specific checks:
    - Case agreement (6 cases: Nominative, Genitive, Dative, Accusative, Instrumental, Prepositional)
    - Verb aspect (perfective/imperfective)
    - Particle usage
    - Register consistency (ты/вы)

    Example:
        >>> agent = RussianFluencyAgent(llm_provider)
        >>> task = TranslationTask(
        ...     source_text="Hello",
        ...     translation="Привет",
        ...     source_lang="en",
        ...     target_lang="ru"
        ... )
        >>> errors = await agent.evaluate(task)
    """

    RUSSIAN_CHECKS = {
        "case_agreement": "Case agreement validation (падежное согласование)",
        "aspect_usage": "Verb aspect (perfective/imperfective) correctness",
        "word_order": "Natural word order for Russian",
        "particle_usage": "Particle correctness (же, ли, бы, etc.)",
        "register": "Formality register (ты/вы consistency)",
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        helper: RussianLanguageHelper | None = None,
    ):
        """Initialize Russian fluency agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            helper: Optional Russian language helper for NLP checks (auto-creates if None)
        """
        super().__init__(llm_provider, temperature, max_tokens)
        # Store Russian-specific prompt template (same as in _check_russian_specifics)
        self._russian_prompt_base = (
            """Russian-specific linguistic validation for professional translation quality."""
        )

        # Initialize NLP helper (or use provided one)
        self.helper = helper if helper is not None else RussianLanguageHelper()

        if self.helper.is_available():
            logger.info("RussianFluencyAgent using MAWO NLP helper for enhanced checks")
        else:
            logger.info("RussianFluencyAgent running without MAWO NLP (LLM-only mode)")

    def get_base_prompt(self) -> str:
        """Get the combined base prompt for Russian fluency evaluation.

        Returns:
            The combined base fluency prompt + Russian-specific prompt
        """
        base_fluency = super().get_base_prompt()
        return f"{base_fluency}\n\n---\n\nRUSSIAN-SPECIFIC CHECKS:\n{self._russian_prompt_base}"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate Russian fluency with hybrid NLP + LLM approach.

        Uses parallel execution:
        1. NLP helper performs deterministic grammar checks
        2. LLM performs semantic and complex linguistic analysis
        3. NLP verifies LLM results (anti-hallucination)
        4. Merge unique errors from both sources

        Args:
            task: Translation task (target_lang must be 'ru')

        Returns:
            List of fluency error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        if task.target_lang != "ru":
            # Fallback to base fluency checks for non-Russian
            return await super().evaluate(task)

        # Run base fluency checks (parallel with Russian-specific)
        base_errors = await super().evaluate(task)

        # Run NLP, LLM, and entity checks in parallel
        try:
            results = await asyncio.gather(
                self._nlp_check(task),  # Fast, deterministic
                self._llm_check(task),  # Slow, semantic
                self._entity_check(task),  # NER-based entity preservation
                return_exceptions=True,
            )

            # Handle exceptions and ensure proper typing
            nlp_result, llm_result, entity_result = results

            # Convert results to list[ErrorAnnotation], handling exceptions
            if isinstance(nlp_result, Exception):
                logger.warning(f"NLP check failed: {nlp_result}")
                nlp_errors: list[ErrorAnnotation] = []
            else:
                nlp_errors = cast(list[ErrorAnnotation], nlp_result)

            if isinstance(llm_result, Exception):
                logger.warning(f"LLM check failed: {llm_result}")
                llm_errors: list[ErrorAnnotation] = []
            else:
                llm_errors = cast(list[ErrorAnnotation], llm_result)

            if isinstance(entity_result, Exception):
                logger.warning(f"Entity check failed: {entity_result}")
                entity_errors: list[ErrorAnnotation] = []
            else:
                entity_errors = cast(list[ErrorAnnotation], entity_result)

            # Verify LLM results with NLP (anti-hallucination)
            verified_llm = self._verify_llm_errors(llm_errors, task.translation)

            # Remove duplicates (NLP errors already caught by LLM)
            unique_nlp = self._remove_duplicates(nlp_errors, verified_llm)

            # Merge all unique errors
            all_errors = base_errors + unique_nlp + verified_llm + entity_errors

            logger.info(
                f"RussianFluencyAgent: "
                f"base={len(base_errors)}, "
                f"nlp={len(unique_nlp)}, "
                f"llm={len(verified_llm)}, "
                f"entity={len(entity_errors)} "
                f"(total={len(all_errors)})"
            )

            return all_errors

        except Exception as e:
            logger.error(f"Russian fluency evaluation failed: {e}")
            # Fallback to base errors
            return base_errors

    async def _nlp_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform NLP-based grammar checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by NLP
        """
        if not self.helper or not self.helper.is_available():
            logger.debug("NLP helper not available, skipping NLP checks")
            return []

        try:
            errors = self.helper.check_grammar(task.translation)
            logger.debug(f"NLP found {len(errors)} grammar errors")
            return errors
        except Exception as e:
            logger.error(f"NLP check failed: {e}")
            return []

    async def _llm_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform LLM-based Russian-specific checks.

        Args:
            task: Translation task

        Returns:
            List of errors found by LLM
        """
        try:
            errors = await self._check_russian_specifics(task)
            logger.debug(f"LLM found {len(errors)} Russian-specific errors")
            return errors
        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return []

    async def _entity_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform NER-based entity preservation checks.

        Args:
            task: Translation task

        Returns:
            List of errors for missing/mismatched entities
        """
        if not self.helper or not self.helper.is_available():
            logger.debug("Helper not available, skipping entity checks")
            return []

        try:
            errors = self.helper.check_entity_preservation(task.source_text, task.translation)
            logger.debug(f"Entity check found {len(errors)} preservation issues")
            return errors
        except Exception as e:
            logger.error(f"Entity check failed: {e}")
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
            # Without NLP, can't verify - return all
            return llm_errors

        verified = []
        for error in llm_errors:
            # Verify position is valid
            if not self.helper.verify_error_position(error, text):
                logger.warning(f"Filtered LLM hallucination: invalid position {error.location}")
                continue

            verified.append(error)

        filtered_count = len(llm_errors) - len(verified)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} LLM hallucinations")

        return verified

    def _remove_duplicates(
        self, nlp_errors: list[ErrorAnnotation], llm_errors: list[ErrorAnnotation]
    ) -> list[ErrorAnnotation]:
        """Remove NLP errors that overlap with LLM errors.

        Args:
            nlp_errors: Errors from NLP
            llm_errors: Errors from LLM

        Returns:
            NLP errors that don't overlap with LLM
        """
        unique = []

        for nlp_error in nlp_errors:
            # Check if this NLP error overlaps with any LLM error
            overlaps = False
            for llm_error in llm_errors:
                if self._errors_overlap(nlp_error, llm_error):
                    overlaps = True
                    break

            if not overlaps:
                unique.append(nlp_error)

        duplicates = len(nlp_errors) - len(unique)
        if duplicates > 0:
            logger.debug(f"Removed {duplicates} duplicate NLP errors")

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

    async def _check_russian_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Russian-specific fluency checks with morphological context.

        Args:
            task: Translation task

        Returns:
            List of Russian-specific errors
        """
        # Get morphological enrichment data if helper is available
        morphology_section = ""
        if self.helper and self.helper.is_available():
            enrichment = self.helper.get_enrichment_data(task.translation)

            if enrichment.get("has_morphology"):
                morphology_section = "\n## MORPHOLOGICAL ANALYSIS (for context):\n"

                # Add verb aspect information
                if enrichment.get("verb_aspects"):
                    morphology_section += "\n**Verbs in translation:**\n"
                    for verb, info in enrichment["verb_aspects"].items():
                        morphology_section += f"- '{verb}': {info['aspect_name']} aspect\n"

                # Add adjective-noun pair information
                if enrichment.get("adjective_noun_pairs"):
                    morphology_section += "\n**Adjective-Noun pairs:**\n"
                    for pair in enrichment["adjective_noun_pairs"]:
                        adj = pair["adjective"]
                        noun = pair["noun"]
                        status = (
                            "✓ agreement OK"
                            if pair["agreement"] == "correct"
                            else "⚠ CHECK agreement"
                        )
                        morphology_section += (
                            f"- '{adj['word']}' ({adj['gender']}, {adj['case']}) + "
                            f"'{noun['word']}' ({noun['gender']}, {noun['case']}) - {status}\n"
                        )

                morphology_section += (
                    "\nUse this morphological context to make informed decisions.\n"
                )

        prompt = f"""You are a native Russian speaker and professional translator.

Your task: Identify ONLY clear Russian-specific linguistic errors in the translation.

## SOURCE TEXT (English):
{task.source_text}
{morphology_section}
## TRANSLATION (Russian):
{task.translation}

## IMPORTANT GUIDELINES:

**What IS an error:**
- Clear grammatical mistakes (case agreement violations)
- Obvious aspectual mistakes that change meaning
- Unnatural constructions that no native speaker would use
- Incorrect particle usage that affects meaning

**What is NOT an error:**
- Stylistic preferences (multiple correct word orders exist)
- Direct translations that are grammatically correct
- Natural Russian that differs from your personal preference
- Minor stylistic variations

## CHECKS TO PERFORM:

1. **Case Agreement (Падежное согласование)** - ONLY flag clear violations
   - Noun-adjective gender/case mismatch
   - Numeral-noun agreement errors

2. **Verb Aspect (Вид глагола)** - Consider source text context
   - Check if aspect matches the source tense/context
   - Perfective for completed/single actions
   - Imperfective for ongoing/repeated actions
   - Remember: both aspects may be acceptable depending on interpretation

3. **Register Consistency** - ONLY if clearly inconsistent
   - Mixing formal (вы) and informal (ты) inappropriately

4. **Particle Usage** - ONLY clear mistakes
   - Incorrect particle that changes/breaks meaning

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "case_agreement|aspect_usage|particle_usage|register",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific Russian linguistic issue with the exact word/phrase you found",
      "suggestion": "Corrected version in Russian"
    }}
  ]
}}

Rules:
- CONSERVATIVE: Only report clear, unambiguous errors
- VERIFY: Ensure the word/phrase you mention actually exists in the text at the specified position
- CONTEXT: Consider the source text when evaluating aspect and tense
- If the translation is natural and grammatically correct, return empty errors array
- Provide accurate character positions (0-indexed, use Python string slicing logic)

Output only valid JSON, no explanation."""

        try:
            # Log metadata only (not the actual content for security)
            logger.info("RussianFluencyAgent - Sending prompt to LLM")
            logger.debug(
                f"Prompt metadata: length={len(prompt)} chars, "
                f"has_morphology={len(morphology_section) > 0}, "
                f"source_length={len(task.source_text)} chars, "
                f"translation_length={len(task.translation)} chars"
            )

            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Log metadata only (not the actual response for security)
            logger.info("RussianFluencyAgent - Received response from LLM")
            logger.debug(f"Response metadata: length={len(response)} chars")

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
                        subcategory=f"russian_{error_dict.get('subcategory', 'specific')}",
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=location_tuple,
                        description=error_dict.get("description", "Russian linguistic issue"),
                        suggestion=error_dict.get("suggestion"),
                    )
                )

            return errors

        except Exception as e:
            logger.error(f"Russian-specific check failed: {e}")
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
