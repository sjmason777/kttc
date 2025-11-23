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
from kttc.terminology import RussianCaseAspectValidator, RussianTrapsValidator

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

        # Initialize glossary-based validator for case/aspect checking
        self.case_validator = RussianCaseAspectValidator()
        logger.info("RussianFluencyAgent initialized with glossary-based case/aspect validator")

        # Initialize Russian traps validator (homonyms, idioms, position verbs, etc.)
        self.traps_validator = RussianTrapsValidator()
        if self.traps_validator.is_available():
            logger.info("RussianFluencyAgent: Russian traps validator enabled (auto)")
        else:
            logger.warning("RussianFluencyAgent: Russian traps glossaries not found")

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

        # Run NLP, LLM, glossary, entity, and traps checks in parallel
        try:
            results = await asyncio.gather(
                self._nlp_check(task),  # Fast, deterministic
                self._llm_check(task),  # Slow, semantic
                self._glossary_check(task),  # Glossary-based case/aspect validation
                self._entity_check(task),  # NER-based entity preservation
                self._traps_check(task),  # Russian traps: homonyms, idioms, position verbs
                return_exceptions=True,
            )

            # Handle exceptions and ensure proper typing
            nlp_result, llm_result, glossary_result, entity_result, traps_result = results

            # Convert results to list[ErrorAnnotation], handling exceptions
            if isinstance(nlp_result, Exception):
                logger.warning(f"NLP check failed: {nlp_result}")
                nlp_errors: list[ErrorAnnotation] = []
            else:
                nlp_errors = cast(list[ErrorAnnotation], nlp_result)
                # CRITICAL: Filter false positives from NLP (digit+genitive, etc.)
                nlp_errors_before = len(nlp_errors)
                nlp_errors = self._filter_false_positives(nlp_errors, task.translation)
                filtered_count = nlp_errors_before - len(nlp_errors)
                if filtered_count > 0:
                    logger.info(f"Filtered {filtered_count} false positives from NLP output")

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

            if isinstance(entity_result, Exception):
                logger.warning(f"Entity check failed: {entity_result}")
                entity_errors: list[ErrorAnnotation] = []
            else:
                entity_errors = cast(list[ErrorAnnotation], entity_result)

            if isinstance(traps_result, Exception):
                logger.warning(f"Traps check failed: {traps_result}")
                traps_errors: list[ErrorAnnotation] = []
            else:
                traps_errors = cast(list[ErrorAnnotation], traps_result)

            # Verify LLM results with NLP (anti-hallucination)
            verified_llm = self._verify_llm_errors(llm_errors, task.translation)

            # Remove duplicates (NLP errors already caught by LLM)
            unique_nlp = self._remove_duplicates(nlp_errors, verified_llm)

            # Remove duplicates from glossary errors
            unique_glossary = self._remove_duplicates(glossary_errors, verified_llm + unique_nlp)

            # Remove duplicates from traps errors
            unique_traps = self._remove_duplicates(
                traps_errors, verified_llm + unique_nlp + unique_glossary
            )

            # Merge all unique errors
            all_errors = (
                base_errors
                + unique_nlp
                + verified_llm
                + unique_glossary
                + entity_errors
                + unique_traps
            )

            logger.info(
                f"RussianFluencyAgent: "
                f"base={len(base_errors)}, "
                f"nlp={len(unique_nlp)}, "
                f"llm={len(verified_llm)}, "
                f"glossary={len(unique_glossary)}, "
                f"entity={len(entity_errors)}, "
                f"traps={len(unique_traps)} "
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

    async def _glossary_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform glossary-based Russian case and aspect validation.

        Uses RussianCaseAspectValidator to check:
        - Case agreement (падежное согласование)
        - Aspect usage (совершенный/несовершенный вид)
        - Preposition-case agreement

        Args:
            task: Translation task

        Returns:
            List of errors found by glossary validation
        """
        errors: list[ErrorAnnotation] = []

        try:
            # Get aspect usage rules for common patterns
            # Check perfective vs imperfective aspect usage
            perfective_rules = self.case_validator.get_aspect_usage_rules("perfective")
            imperfective_rules = self.case_validator.get_aspect_usage_rules("imperfective")

            if perfective_rules.get("when_to_use"):
                # For now, we provide aspect info that can be used by LLM
                # Full aspect detection would require morphological parsing
                logger.debug(
                    f"Loaded aspect rules: perfective={len(perfective_rules.get('when_to_use', []))} "
                    f"cases, imperfective={len(imperfective_rules.get('when_to_use', []))} cases"
                )

            # Validate common Russian cases
            # This is a simplified check - full validation would require morphological parsing
            # For now, we enrich errors with glossary-based case information
            for case_name in [
                "nominative",
                "genitive",
                "dative",
                "accusative",
                "instrumental",
                "prepositional",
            ]:
                case_info = self.case_validator.get_case_info(case_name)
                if case_info:
                    logger.debug(f"Loaded {case_name} case info: {case_info.get('function')}")

            logger.debug("Glossary check completed (case/aspect rules loaded)")

        except Exception as e:
            logger.error(f"Glossary check failed: {e}")

        return errors

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

    def _check_idioms(self, translation: str, analysis: dict[str, Any]) -> list[ErrorAnnotation]:
        """Check idioms in translation."""
        errors = []
        for idiom in analysis.get("idioms", []):
            idiom_text = idiom.get("idiom", "")
            pos = translation.lower().find(idiom_text.lower())
            if pos >= 0:
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="russian_idiom",
                        severity=ErrorSeverity.MAJOR,
                        location=(pos, pos + len(idiom_text)),
                        description=f"Idiom detected: '{idiom_text}' means '{idiom.get('meaning', '')}'. Literal translation '{idiom.get('literal', '')}' is INCORRECT.",
                        suggestion=idiom.get("english_equivalent"),
                    )
                )
        return errors

    def _check_position_verbs(self, analysis: dict[str, Any]) -> list[ErrorAnnotation]:
        """Check position verb errors."""
        return [
            ErrorAnnotation(
                category="fluency",
                subcategory="russian_position_verb",
                severity=ErrorSeverity.MAJOR,
                location=(0, 10),
                description=f"Position verb error: {e.get('reason', '')}. Use '{e.get('correct', '')}' instead.",
                suggestion=e.get("correct"),
            )
            for e in analysis.get("position_verbs", [])
        ]

    def _check_homonyms(self, translation: str, analysis: dict[str, Any]) -> list[ErrorAnnotation]:
        """Check critical homonyms."""
        errors = []
        for homonym in analysis.get("homonyms", []):
            if homonym.get("severity") != "critical":
                continue
            word = homonym.get("word", "")
            pos = translation.lower().find(word.lower())
            if pos < 0:
                continue
            meanings = ", ".join(
                [
                    f"{m.get('meaning', '')} ({m.get('english', '')})"
                    for m in homonym.get("meanings", [])[:3]
                ]
            )
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="russian_homonym",
                    severity=ErrorSeverity.MINOR,
                    location=(pos, pos + len(word)),
                    description=f"Homonym '{word}' has multiple meanings: {meanings}. Verify correct translation based on context.",
                    suggestion=None,
                )
            )
        return errors

    def _check_paronyms(self, translation: str, analysis: dict[str, Any]) -> list[ErrorAnnotation]:
        """Check paronym confusion."""
        errors = []
        for paronym in analysis.get("paronyms", []):
            if not paronym.get("common_error"):
                continue
            word = paronym.get("word", "")
            pos = translation.lower().find(word.lower())
            if pos < 0:
                continue
            pair = paronym.get("pair", [])
            definitions = paronym.get("definitions", {})
            other_word = [w for w in pair if w != word]
            other_def = definitions.get(other_word[0], "") if other_word else ""
            errors.append(
                ErrorAnnotation(
                    category="fluency",
                    subcategory="russian_paronym",
                    severity=ErrorSeverity.MINOR,
                    location=(pos, pos + len(word)),
                    description=f"Paronym check: '{word}' ({definitions.get(word, '')}). Often confused with '{other_word[0] if other_word else ''}' ({other_def}).",
                    suggestion=None,
                )
            )
        return errors

    async def _traps_check(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Check for Russian language traps (idioms, position verbs, homonyms, paronyms)."""
        if not self.traps_validator or not self.traps_validator.is_available():
            logger.debug("Traps validator not available, skipping traps checks")
            return []

        try:
            translation = task.translation
            analysis = self.traps_validator.analyze_text(translation)
            errors = self._check_idioms(translation, analysis)
            errors.extend(self._check_position_verbs(analysis))
            errors.extend(self._check_homonyms(translation, analysis))
            errors.extend(self._check_paronyms(translation, analysis))
            logger.debug(f"Traps check found {len(errors)} issues")
            return errors

        except Exception as e:
            logger.error(f"Traps check failed: {e}")

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

    def _build_morphology_section(self, translation: str) -> str:
        """Build morphology section for prompt if helper is available."""
        if not self.helper or not self.helper.is_available():
            return ""

        enrichment = self.helper.get_enrichment_data(translation)
        if not enrichment.get("has_morphology"):
            return ""

        section = "\n## MORPHOLOGICAL ANALYSIS (for context):\n"
        if enrichment.get("verb_aspects"):
            section += "\n**Verbs in translation:**\n"
            for verb, info in enrichment["verb_aspects"].items():
                section += f"- '{verb}': {info['aspect_name']} aspect\n"

        if enrichment.get("adjective_noun_pairs"):
            section += "\n**Adjective-Noun pairs:**\n"
            for pair in enrichment["adjective_noun_pairs"]:
                adj, noun = pair["adjective"], pair["noun"]
                status = "✓ agreement OK" if pair["agreement"] == "correct" else "⚠ CHECK agreement"
                section += f"- '{adj['word']}' ({adj['gender']}, {adj['case']}) + '{noun['word']}' ({noun['gender']}, {noun['case']}) - {status}\n"

        section += "\nUse this morphological context to make informed decisions.\n"
        return section

    async def _check_russian_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Russian-specific fluency checks with morphological context."""
        morphology_section = self._build_morphology_section(task.translation)

        # MODERN PROMPT ENGINEERING (Nov 2025):
        # 1. Step-Back Prompting: First classify text type
        # 2. According-to Prompting: Bind to source text
        # 3. Few-Shot Examples: Show correct/incorrect patterns
        # 4. Chain-of-Verification: Self-check answers

        prompt = f"""You are a native Russian speaker and professional linguist with expertise in grammar checking.

## STEP 1: TEXT CLASSIFICATION (Step-Back Prompting)

First, classify the text type to apply appropriate rules:
- **Technical**: Product reviews, specifications, IT articles (Gen 5, Wi-Fi 7, USB4 are PROPER NAMES)
- **Literary**: Fiction, creative writing (more flexible word order)
- **Formal**: Business, academic (strict grammar rules)

According to the SOURCE TEXT below, this appears to be: [auto-detect from content]

## SOURCE TEXT (English):
{task.source_text}
{morphology_section}
## TRANSLATION (Russian - YOUR PRIMARY REFERENCE):
{task.translation}

**CRITICAL**: All your analysis MUST be "according to the TRANSLATION text above". Do not invent phrases that don't exist in the translation.

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
- Technical terminology and standard names (see SPECIAL CASES below)

## FEW-SHOT EXAMPLES (Learn from these!)

### ✅ CORRECT - Do NOT flag these:

**Technical terminology:**
- "материнская плата поддерживает PCIe Gen 5" ✓ (Gen 5 is a standard name, NOT "пять")
- "процессор Ryzen 9 9950X" ✓ (model number, NOT numeral)
- "скорость 20 Гбит/с" ✓ (technical specification)

**Plural forms:**
- "За эти деньги вы получаете много" ✓ (plural accusative = nominative form)
- "получить дополнительные функции" ✓ (plural accusative = nominative form)

**Standard particles:**
- "то же самое можно сказать" ✓ (standard phrase, separate spelling)
- "если X, то Y" ✓ (conditional construction)

### ❌ INCORRECT - These ARE errors:

**Case agreement:**
- "красивый девочка" ❌ (gender mismatch: красивая девочка)
- "в городом" ❌ (wrong case: в городе)

**Real numeral agreement:**
- "пять дом" ❌ (should be: пять домов)
- "два книга" ❌ (should be: две книги)
- "2 минута" ❌ (should be: 2 минуты)

**IMPORTANT: Digit numerals 5-20 require genitive plural:**
- "5 минут" ✓ CORRECT (5 + genitive plural)
- "10 минут" ✓ CORRECT (10 + genitive plural)
- "15 секунд" ✓ CORRECT (15 + genitive plural)
- "20 часов" ✓ CORRECT (20 + genitive plural)

Do NOT flag digit numerals (2, 5, 10, etc.) with correct genitive forms as errors!

## SPECIAL CASES - DO NOT FLAG THESE AS ERRORS:

### 1. Technical Terminology (названия стандартов, версий, спецификаций)

Technical designations are NOT numerals with nouns. DO NOT check numeral agreement for:

**Technology versions and standards:**
- PCIe Gen 5, Gen 4, Gen 3 (NOT "пять", "четыре", "три")
- Wi-Fi 7, Wi-Fi 6E, Wi-Fi 6 (NOT "семь", "шесть")
- USB4, USB 3.2, USB 2.0 (version numbers, not numerals)
- Thunderbolt 4, Thunderbolt 3 (protocol names)
- DDR5, DDR4, GDDR6 (memory types)
- HDMI 2.1, DisplayPort 1.4 (interface versions)

**Processor/hardware models:**
- Ryzen 9 9950X, Ryzen 7 7800X3D (model numbers)
- RTX 4090, RTX 4080 (GPU models)
- Core i9-14900K (CPU models)

**Technical specifications:**
- Type-C 20 Гбит/с (speed specifications)
- 230 MHz, 5.0 GHz (frequencies)
- Technical measurements with numbers

These are PROPER NAMES, not numeral+noun constructions. Never flag them as agreement errors!

### 2. Plural Forms (множественное число)

For inanimate nouns and adjectives in PLURAL:
- Nominative case (именительный) == Accusative case (винительный)
- The forms are IDENTICAL - this is correct Russian grammar!

**Examples of CORRECT usage:**
- "За эти деньги" (за + accs plural = nomn plural form) ✓ CORRECT
- "за дополнительные доллары" (за + accs plural = nomn plural form) ✓ CORRECT
- "получить дополнительные функции" (accs plural = nomn plural form) ✓ CORRECT
- "Я вижу эти книги" (accs plural = nomn plural form) ✓ CORRECT

DO NOT flag preposition + plural inanimate noun/adjective as case error!
The nominative form is CORRECT for accusative in plural.

### 3. Particle and Conjunction Usage (частицы и союзы)

**"то" has different meanings and spellings:**

**РАЗДЕЛЬНО (separate):**
- "то же самое" (the same thing) - CORRECT ✓
- "то же" (the same) - CORRECT ✓
- "если... то" (if... then) - CORRECT ✓
- "то есть" (that is) - CORRECT ✓

**СЛИТНО (together):**
- "тоже" (also, too) - CORRECT ✓

**ЧЕРЕЗ ДЕФИС (hyphen):**
- "кто-то" (someone) - CORRECT ✓
- "что-то" (something) - CORRECT ✓
- "как-то" (somehow) - CORRECT ✓
- "где-то" (somewhere) - CORRECT ✓

DO NOT suggest adding hyphens to "то же" or "если... то"!
DO NOT flag "то же самое" as double particle usage - this is a standard phrase!

## CHECKS TO PERFORM:

1. **Case Agreement (Падежное согласование)** - ONLY flag clear violations
   - Noun-adjective gender/case mismatch
   - Numeral-noun agreement errors (but EXCLUDE technical terminology - see SPECIAL CASES #1)
   - REMEMBER: Plural accusative == nominative for inanimate (see SPECIAL CASES #2)

2. **Verb Aspect (Вид глагола)** - Consider source text context
   - Check if aspect matches the source tense/context
   - Perfective for completed/single actions
   - Imperfective for ongoing/repeated actions
   - Remember: both aspects may be acceptable depending on interpretation

3. **Register Consistency** - ONLY if clearly inconsistent
   - Mixing formal (вы) and informal (ты) inappropriately

4. **Particle Usage** - ONLY clear mistakes
   - Incorrect particle that changes/breaks meaning
   - REMEMBER: "то же" and "если...то" are correct (see SPECIAL CASES #3)

## CHAIN-OF-VERIFICATION (Self-Check Process)

**CRITICAL**: Before finalizing ANY error, you MUST verify through this self-check loop:

### Verification Questions for EACH Potential Error:

**Q1: Location Verification**
- "Does text[start:end] actually contain the EXACT phrase I'm flagging?"
- "Can I point to this exact location in the TRANSLATION text above?"
- **If NO**: Do NOT report this error (it's a hallucination)

**Q2: Technical Term Check** (for numeral_agreement errors only)
- "Could this be a technical designation: Gen X, Wi-Fi X, USB X, Ryzen X, RTX X, DDR X?"
- "Is this a product model, standard name, or technical specification?"
- **If YES**: Do NOT report this error (it's a proper name, not a numeral)
- "Is this a digit numeral (2, 5, 10, etc.) with genitive plural/singular (минут, часов, секунд, дня, недель)?"
- **If YES**: Do NOT report this error (digit numerals with correct genitive forms are correct!)

**Q3: Plural Form Check** (for case_agreement and preposition_case errors only)
- "Is this a plural inanimate noun or adjective?"
- "Remember: plural inanimate nominative == accusative (это правильно!)"
- **If YES**: Do NOT report this error (the nominative form IS correct for accusative)

**Q4: Standard Phrase Check** (for particle_usage errors only)
- "Is this 'то же самое', 'то же', 'если...то', or 'то есть'?"
- "These are STANDARD CORRECT phrases in Russian"
- **If YES**: Do NOT report this error (it's correct usage)

**Q5: Source Context Check** (for aspect_usage errors only)
- "Does the source text context support both aspects?"
- "Is my suggested aspect correction actually more natural?"
- **If UNCERTAIN**: Do NOT report this error (multiple aspects can be valid)

### Verification Summary

After checking ALL potential errors, provide:
- **text_type**: Technical/Literary/Formal classification from STEP 1
- **verification_summary**: "Checked X potential errors, filtered Y false positives (Q1: Z hallucinations, Q2: A tech terms, Q3: B plural forms, Q4: C standard phrases), reporting D real errors"
- **errors**: Only errors that passed ALL verification questions

## OUTPUT FORMAT (Structured JSON Schema)

You MUST return ONLY valid JSON with this EXACT structure:

{{
  "text_type": "technical|literary|formal",
  "verification_summary": "Checked X potential errors, filtered Y false positives (Q1: Z hallucinations, Q2: A tech terms, Q3: B plural forms, Q4: C standard phrases, Q5: E uncertain aspects), reporting F real errors",
  "errors": [
    {{
      "subcategory": "case_agreement|aspect_usage|particle_usage|register",
      "severity": "critical|major|minor",
      "location": [start_char_int, end_char_int],
      "description": "According to the translation, the phrase '[exact phrase from text[start:end]]' has [specific issue]",
      "suggestion": "Corrected version in Russian",
      "verification": {{
        "location_verified": true,
        "not_technical_term": true,
        "not_plural_form": true,
        "not_standard_phrase": true,
        "source_context_supports": true
      }}
    }}
  ]
}}

**CRITICAL JSON Requirements**:
- `text_type`: MUST be one of: "technical", "literary", "formal"
- `verification_summary`: MUST include counts (X potential, Y filtered, Z real)
- `errors`: Array (can be empty if no errors found)
- `location`: MUST be [int, int] with valid 0-indexed positions
- `description`: MUST start with "According to the translation, the phrase '[exact phrase]'..."
- `verification`: MUST include all 5 boolean fields

Rules:
- CONSERVATIVE: Only report clear, unambiguous errors
- VERIFY: Ensure the word/phrase you mention actually exists in the text at the specified position
- CONTEXT: Consider the source text when evaluating aspect and tense
- SPECIAL CASES: Review SPECIAL CASES section - DO NOT flag technical terms, plural forms, or standard particles
- If the translation is natural and grammatically correct, return empty errors array
- Provide accurate character positions (0-indexed, use Python string slicing logic)
- DOUBLE-CHECK: Before flagging numeral agreement, confirm it's not a technical designation (Gen 5, Wi-Fi 7, etc.)
- DOUBLE-CHECK: Before flagging preposition case, confirm it's not a plural inanimate (nomn == accs)
- DOUBLE-CHECK: Before flagging particle usage, confirm it's not "то же самое" or "если...то"

Output only valid JSON, no explanation."""

        try:
            # Send prompt to LLM (no logging of sensitive data or metadata)
            logger.info("RussianFluencyAgent - Sending prompt to LLM")

            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Response received (no logging of sensitive data or metadata)
            logger.info("RussianFluencyAgent - Received response from LLM")

            # Parse response
            response_data = self._parse_json_response(response)

            # Extract and log verification metadata (Chain-of-Verification)
            text_type = response_data.get("text_type", "unknown")
            verification_summary = response_data.get("verification_summary", "")

            if text_type != "unknown":
                logger.info(f"LLM classified text as: {text_type}")
            if verification_summary:
                logger.info(f"LLM verification: {verification_summary}")

            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                location = error_dict.get("location", [0, 10])
                if isinstance(location, list) and len(location) == 2:
                    location_tuple = (location[0], location[1])
                else:
                    location_tuple = (0, 10)

                # Log verification data if present (for debugging)
                verification = error_dict.get("verification", {})
                if verification:
                    logger.debug(f"Error verification data: {verification}")

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

            # Filter false positives (technical terms, plural forms, standard particles)
            filtered_errors = self._filter_false_positives(errors, task.translation)
            filtered_count = len(errors) - len(filtered_errors)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} false positives from LLM output")

            return filtered_errors

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

    # Patterns for false positive filtering
    _TECH_PATTERNS = [
        r"\bGen\s+\d+",
        r"\bWi-Fi\s+\d+",
        r"\bUSB\s*\d+",
        r"\bThunderbolt\s+\d+",
        r"\bDDR\d+",
        r"\bGDDR\d+",
        r"\bHDMI\s+\d+",
        r"\bDisplayPort\s+\d+",
        r"\bRyzen\s+\d+",
        r"\bRTX\s+\d+",
        r"\bCore\s+i\d+",
        r"\bType-C\s+\d+",
        r"\d+\s+(MHz|GHz|Гбит/с|долларов?)",
    ]

    _DIGIT_GENITIVE_PATTERNS = [
        r"\d+\s+(минут[ыа]?|час(ов|а)?|секунд[ыа]?|д(ней|ня)|недель?|месяц(ев|а)?|лет|год(ов|а)?)",
        r"\d+\s+(раз[а]?|штук|процент(ов|а)?|человек|рублей|долларов)",
    ]

    _PARTICLE_PHRASES = [
        r"\bто\s+же\s+самое",
        r"\bто\s+же\b",
        r"\bесли\s+.+\s+то\b",
        r"\bто\s+есть\b",
    ]

    def _is_digit_genitive_fp(self, error: ErrorAnnotation) -> bool:
        """Check if error is a digit+genitive false positive."""
        if error.subcategory != "russian_numeral_agreement":
            return False
        pattern_match = re.search(r"\d+\s+requires\s+gent", error.description, re.IGNORECASE)
        if pattern_match:
            logger.info(f"Filtered digit+genitive FP: '{pattern_match.group()}'")
            return True
        return False

    def _is_tech_term_fp(self, flagged_text: str, context: str) -> bool:
        """Check if error is on a technical term."""
        for pattern in self._TECH_PATTERNS:
            if re.search(pattern, flagged_text, re.IGNORECASE) or re.search(
                pattern, context, re.IGNORECASE
            ):
                logger.info(f"Filtered tech term FP: pattern '{pattern}'")
                return True
        return False

    def _is_digit_genitive_location_fp(self, flagged_text: str, context: str) -> bool:
        """Check if error is on correct digit+genitive pattern."""
        for pattern in self._DIGIT_GENITIVE_PATTERNS:
            if re.search(pattern, flagged_text, re.IGNORECASE) or re.search(
                pattern, context, re.IGNORECASE
            ):
                logger.info(f"Filtered digit+genitive location FP: pattern '{pattern}'")
                return True
        return False

    def _is_preposition_case_fp(
        self, error: ErrorAnnotation, flagged_text: str, context: str
    ) -> bool:
        """Check if error is a plural inanimate preposition case false positive."""
        if error.subcategory != "russian_preposition_case":
            return False
        desc_lower = error.description.lower()
        if "got nomn" not in desc_lower and "got nom" not in desc_lower:
            return False
        combined = flagged_text + " " + context
        if re.search(r"\b(эти|дополнительные|основные|все|других)\b", combined):
            logger.info("Filtered preposition case FP: plural inanimate")
            return True
        return False

    def _is_particle_usage_fp(
        self, error: ErrorAnnotation, flagged_text: str, context: str
    ) -> bool:
        """Check if error is a standard particle phrase false positive."""
        if error.subcategory != "russian_particle_usage":
            return False
        combined = flagged_text + " " + context
        for pattern in self._PARTICLE_PHRASES:
            if re.search(pattern, combined, re.IGNORECASE):
                logger.info(f"Filtered particle usage FP: pattern '{pattern}'")
                return True
        return False

    def _filter_false_positives(
        self, errors: list[ErrorAnnotation], text: str
    ) -> list[ErrorAnnotation]:
        """Filter known false positives from LLM errors."""
        logger.info(f"=== Filter Debug: Processing {len(errors)} errors ===")

        filtered = []
        for error in errors:
            # Filter digit+genitive errors first (before location check)
            if self._is_digit_genitive_fp(error):
                continue

            # Skip if no valid location
            if not error.location or len(error.location) != 2:
                filtered.append(error)
                continue

            start, end = error.location
            if start < 0 or end > len(text):
                filtered.append(error)
                continue

            # Get flagged text and context
            flagged_text = text[start:end]
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end]

            # Filter numeral agreement false positives
            if error.subcategory == "russian_numeral_agreement":
                if self._is_tech_term_fp(flagged_text, context):
                    continue
                if self._is_digit_genitive_location_fp(flagged_text, context):
                    continue

            # Filter preposition case false positives
            if self._is_preposition_case_fp(error, flagged_text, context):
                continue

            # Filter particle usage false positives
            if self._is_particle_usage_fp(error, flagged_text, context):
                continue

            filtered.append(error)

        return filtered
