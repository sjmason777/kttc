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

"""False Positive Filter for reducing LLM hallucinations in error detection.

This module implements multiple filtering strategies based on 2025 research:
- Glossary-based filtering (do_not_translate, untranslatable terms)
- Self-contradiction detection
- Cultural adaptation awareness
- IT terminology whitelist

References:
- HiMATE: Hierarchical Multi-Agent Framework (2025)
- MQM-APE: Error Annotation with Post-Editing (2024)
- DCSQE: Distribution-Controlled Synthesis for QE (2025)
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from kttc.terminology import GlossaryManager

if TYPE_CHECKING:
    from kttc.core import ErrorAnnotation, TranslationTask
    from kttc.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class GlossaryBasedFilter:
    """Filter false positives using glossary data.

    Loads do_not_translate terms, untranslatable cultural terms,
    and IT terminology to filter out errors that are not real issues.
    """

    def __init__(self, glossary_manager: GlossaryManager | None = None):
        """Initialize the glossary-based filter.

        Args:
            glossary_manager: GlossaryManager instance. Created if not provided.
        """
        self.glossary_manager = glossary_manager or GlossaryManager()
        self._do_not_translate: set[str] | None = None
        self._it_terms: set[str] | None = None
        self._untranslatable_cache: dict[str, dict[str, Any]] = {}

    @property
    def do_not_translate_terms(self) -> set[str]:
        """Lazily load do_not_translate terms."""
        if self._do_not_translate is None:
            self._do_not_translate = self.glossary_manager.get_do_not_translate_terms()
            logger.debug(f"Loaded {len(self._do_not_translate)} do_not_translate terms")
        return self._do_not_translate

    @property
    def it_terms(self) -> set[str]:
        """Lazily load IT terminology."""
        if self._it_terms is None:
            self._it_terms = self.glossary_manager.get_it_terminology("en")
            logger.debug(f"Loaded {len(self._it_terms)} IT terminology terms")
        return self._it_terms

    def get_untranslatable_terms(self, language: str) -> dict[str, dict[str, Any]]:
        """Get untranslatable terms for a language (cached)."""
        if language not in self._untranslatable_cache:
            self._untranslatable_cache[language] = self.glossary_manager.get_untranslatable_terms(
                language
            )
        return self._untranslatable_cache[language]

    def is_false_positive(
        self,
        error: ErrorAnnotation,
        source_lang: str = "en",
        target_lang: str = "ru",
    ) -> bool:
        """Check if an error is likely a false positive.

        Args:
            error: Error annotation to check
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            True if error appears to be a false positive
        """
        # Check do_not_translate terms
        if self._is_do_not_translate_fp(error):
            return True

        # Check IT terminology
        if self._is_it_term_fp(error):
            return True

        # Check untranslatable cultural terms
        if self._is_untranslatable_fp(error, target_lang):
            return True

        # Check cultural adaptation
        if self._is_cultural_adaptation_fp(error, target_lang):
            return True

        return False

    def _is_do_not_translate_fp(self, error: ErrorAnnotation) -> bool:
        """Check if error is about a do_not_translate term."""
        if error.category not in ("terminology", "accuracy"):
            return False

        if error.subcategory not in ("untranslated", "inconsistency", "misuse"):
            return False

        description_lower = error.description.lower()

        # Check if any do_not_translate term is mentioned
        for term in self.do_not_translate_terms:
            if self._term_in_description(term, description_lower):
                if self._is_untranslated_complaint(description_lower):
                    logger.debug(f"Filtered do_not_translate FP: {term}")
                    return True

        return False

    def _is_it_term_fp(self, error: ErrorAnnotation) -> bool:
        """Check if error is about IT terminology."""
        if error.category not in ("terminology", "accuracy"):
            return False

        description_lower = error.description.lower()

        # Check if any IT term is mentioned
        for term in self.it_terms:
            if len(term) > 2 and self._term_in_description(term, description_lower):
                if self._is_untranslated_complaint(description_lower):
                    logger.debug(f"Filtered IT term FP: {term}")
                    return True

        return False

    def _is_untranslatable_fp(self, error: ErrorAnnotation, target_lang: str) -> bool:
        """Check if error is about a culturally untranslatable term."""
        if error.category != "accuracy":
            return False

        untranslatable = self.get_untranslatable_terms(target_lang)
        if not untranslatable:
            return False

        description_lower = error.description.lower()

        for term, _metadata in untranslatable.items():
            if term in description_lower:
                # Check if error is complaining about non-literal translation
                non_literal_indicators = [
                    "not literal",
                    "different meaning",
                    "semantic",
                    "meaning change",
                    "adds meaning",
                    "loses meaning",
                ]
                if any(ind in description_lower for ind in non_literal_indicators):
                    logger.debug(f"Filtered untranslatable term FP: {term}")
                    return True

        return False

    def _is_cultural_adaptation_fp(self, error: ErrorAnnotation, target_lang: str) -> bool:
        """Check if error is about valid cultural adaptation."""
        if error.category != "accuracy":
            return False

        cultural_adaptations = self.glossary_manager.get_cultural_adaptation_terms(target_lang)
        if not cultural_adaptations:
            return False

        description_lower = error.description.lower()

        for source_term, target_term in cultural_adaptations.items():
            # Check if error mentions both source and target of adaptation
            if source_term.lower() in description_lower:
                if target_term.lower() in description_lower:
                    # Error is about known cultural adaptation
                    logger.debug(f"Filtered cultural adaptation FP: {source_term} -> {target_term}")
                    return True

        return False

    @staticmethod
    def _term_in_description(term: str, description_lower: str) -> bool:
        """Check if a term appears in the description."""
        patterns = [
            f'"{term}"',
            f"'{term}'",
            f" {term} ",
            f" {term}.",
            f" {term},",
            f"[{term}]",
            f"({term})",
            f" {term}\n",
        ]
        return any(pattern in description_lower for pattern in patterns)

    @staticmethod
    def _is_untranslated_complaint(description_lower: str) -> bool:
        """Check if description complains about untranslated term."""
        indicators = [
            "untranslated",
            "not translated",
            "left as",
            "kept as",
            "left in english",
            "kept in english",
            "inconsisten",  # covers inconsistent/inconsistency
            "should be translated",
            "without translation",
        ]
        return any(ind in description_lower for ind in indicators)


class SelfReflectionFilter:
    """Filter false positives using self-reflection (HiMATE-inspired).

    Generates corrections and filters errors where the correction
    doesn't meaningfully differ from the original.
    """

    # Minimum similarity threshold to consider texts as "same"
    SIMILARITY_THRESHOLD = 0.92

    def __init__(self, llm_provider: BaseLLMProvider | None = None):
        """Initialize the self-reflection filter.

        Args:
            llm_provider: LLM provider for generating corrections.
                          If None, only text-similarity checks are performed.
        """
        self.llm_provider = llm_provider

    async def verify_error(
        self,
        error: ErrorAnnotation,
        task: TranslationTask,
    ) -> bool:
        """Verify if an error is real by attempting to correct it.

        Args:
            error: Error to verify
            task: Translation task context

        Returns:
            True if error appears to be real, False if likely false positive
        """
        if not self.llm_provider:
            # Without LLM, can't do self-reflection
            return True

        # Generate correction
        correction_prompt = self._build_correction_prompt(error, task)

        try:
            corrected = await self.llm_provider.complete(
                prompt=correction_prompt,
                max_tokens=500,
                temperature=0.1,
            )

            # Extract corrected text
            corrected_text = self._extract_correction(corrected)

            if not corrected_text:
                return True  # Couldn't extract correction, assume error is real

            # Compare with original
            similarity = self._text_similarity(task.translation, corrected_text)

            if similarity >= self.SIMILARITY_THRESHOLD:
                # Correction is essentially the same as original
                # This suggests the "error" didn't need fixing
                logger.debug(
                    f"Self-reflection: filtered FP (similarity={similarity:.2f}): "
                    f"{error.description[:50]}..."
                )
                return False

            return True  # Error appears to be real

        except Exception as e:
            logger.warning(f"Self-reflection failed: {e}")
            return True  # On error, assume error is real (conservative)

    def _build_correction_prompt(self, error: ErrorAnnotation, task: TranslationTask) -> str:
        """Build prompt for generating a correction."""
        return f"""Fix this specific translation error. Return ONLY the corrected translation, nothing else.

Source ({task.source_lang}): {task.source_text}
Translation ({task.target_lang}): {task.translation}

Error found: {error.description}
Error location: characters {error.location[0]}-{error.location[1]}

Corrected translation:"""

    @staticmethod
    def _extract_correction(response: str) -> str | None:
        """Extract corrected text from LLM response."""
        # Clean up response
        corrected = response.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "corrected translation:",
            "correction:",
            "fixed:",
            "here is the corrected translation:",
        ]
        for prefix in prefixes_to_remove:
            if corrected.lower().startswith(prefix):
                corrected = corrected[len(prefix) :].strip()

        return corrected if corrected else None

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


class ConfidenceRouter:
    """Route errors based on confidence for debate verification.

    Low-confidence errors are sent to debate for cross-verification.
    """

    # Default confidence threshold (60th percentile as per HiMATE)
    DEFAULT_THRESHOLD = 0.6

    def __init__(self, confidence_threshold: float | None = None):
        """Initialize the confidence router.

        Args:
            confidence_threshold: Threshold for routing to debate.
                                  Errors below this go to debate.
        """
        self.confidence_threshold = confidence_threshold or self.DEFAULT_THRESHOLD

    def route_errors(
        self, errors: list[ErrorAnnotation]
    ) -> tuple[list[ErrorAnnotation], list[ErrorAnnotation]]:
        """Route errors into high-confidence and low-confidence groups.

        Args:
            errors: List of errors to route

        Returns:
            Tuple of (high_confidence_errors, low_confidence_errors)
        """
        high_confidence: list[ErrorAnnotation] = []
        low_confidence: list[ErrorAnnotation] = []

        for error in errors:
            # Use confidence if available, otherwise check description indicators
            confidence = error.confidence if error.confidence is not None else 0.5

            # Additional heuristics for confidence estimation
            if confidence == 0.5:
                confidence = self._estimate_confidence(error)

            if confidence >= self.confidence_threshold:
                high_confidence.append(error)
            else:
                low_confidence.append(error)

        logger.debug(
            f"Routed errors: {len(high_confidence)} high-confidence, "
            f"{len(low_confidence)} low-confidence (threshold={self.confidence_threshold})"
        )

        return high_confidence, low_confidence

    @staticmethod
    def _estimate_confidence(error: ErrorAnnotation) -> float:
        """Estimate confidence based on error characteristics."""
        confidence = 0.5
        description_lower = error.description.lower()

        # Lower confidence indicators
        low_confidence_phrases = [
            "might be",
            "could be",
            "possibly",
            "perhaps",
            "may not",
            "minor",
            "subjective",
            "preference",
            "stylistic",
            "however",
            "though",
            "but",
            "acceptable",
        ]
        for phrase in low_confidence_phrases:
            if phrase in description_lower:
                confidence -= 0.1

        # Higher confidence indicators
        high_confidence_phrases = [
            "clearly",
            "definitely",
            "incorrect",
            "wrong",
            "error",
            "mistake",
            "missing",
            "omitted",
            "added",
            "critical",
            "major",
        ]
        for phrase in high_confidence_phrases:
            if phrase in description_lower:
                confidence += 0.1

        # Severity affects confidence
        if error.severity.value == "critical":
            confidence += 0.2
        elif error.severity.value == "major":
            confidence += 0.1
        elif error.severity.value == "neutral":
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))


@lru_cache(maxsize=1)
def get_glossary_filter() -> GlossaryBasedFilter:
    """Get cached glossary-based filter instance."""
    return GlossaryBasedFilter()
