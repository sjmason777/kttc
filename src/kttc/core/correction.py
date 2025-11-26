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

"""Automatic Error Correction for Post-Editing.

Provides automatic correction of detected translation errors:
- Light PE: Fix critical and major errors only
- Full PE: Fix all errors
- Re-evaluation after correction

Based on AI Post-Editing 2025 research (40% faster, 60% cost reduction).
"""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class AutoCorrector:
    """Automatic error correction for light post-editing.

    Applies corrections to translations based on detected errors:
    - Uses error suggestions if available
    - Generates corrections via LLM if no suggestion
    - Supports light (critical/major only) and full (all errors) modes

    Example:
        >>> corrector = AutoCorrector(llm_provider)
        >>> corrected = await corrector.auto_correct(
        ...     task=task,
        ...     errors=errors,
        ...     correction_level="light"
        ... )
        >>> print(f"Original: {task.translation}")
        >>> print(f"Corrected: {corrected}")
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        """Initialize auto-corrector.

        Args:
            llm_provider: LLM provider for generating corrections
        """
        self.llm_provider = llm_provider

    async def auto_correct(
        self,
        task: TranslationTask,
        errors: list[ErrorAnnotation],
        correction_level: str = "light",  # "light" or "full"
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """Automatically correct translation based on detected errors.

        Args:
            task: Original translation task
            errors: List of detected errors
            correction_level: "light" (critical/major only) or "full" (all errors)
            temperature: LLM temperature for correction generation
            max_tokens: Max tokens for LLM response

        Returns:
            Corrected translation text

        Example:
            >>> corrector = AutoCorrector(provider)
            >>> errors = [
            ...     ErrorAnnotation(
            ...         category="accuracy",
            ...         subcategory="mistranslation",
            ...         severity=ErrorSeverity.CRITICAL,
            ...         location=(0, 5),
            ...         description="Wrong word",
            ...         suggestion="Correct word"
            ...     )
            ... ]
            >>> corrected = await corrector.auto_correct(task, errors, "light")
        """
        if not errors:
            logger.info("No errors to correct")
            return task.translation

        # Filter errors by correction level
        if correction_level == "light":
            # Only fix critical and major errors
            errors_to_fix = [
                e for e in errors if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.MAJOR]
            ]
            logger.info(
                f"Light PE: Correcting {len(errors_to_fix)} critical/major errors "
                f"(skipping {len(errors) - len(errors_to_fix)} minor errors)"
            )
        else:
            # Fix all errors
            errors_to_fix = errors
            logger.info(f"Full PE: Correcting all {len(errors_to_fix)} errors")

        if not errors_to_fix:
            logger.info("No errors to fix after filtering")
            return task.translation

        # Use LLM to generate corrected translation
        # This is more robust than applying individual corrections
        corrected = await self._generate_corrected_translation(
            task, errors_to_fix, temperature, max_tokens
        )

        logger.info(f"Generated corrected translation ({len(corrected)} chars)")
        return corrected

    async def _generate_corrected_translation(
        self,
        task: TranslationTask,
        errors: list[ErrorAnnotation],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate corrected translation using LLM.

        Instead of applying corrections sequentially, ask LLM to fix all errors at once.
        This produces more natural and coherent corrections.

        Args:
            task: Translation task
            errors: Errors to fix
            temperature: LLM temperature
            max_tokens: Max tokens

        Returns:
            Corrected translation
        """
        # Build error descriptions
        error_descriptions = []
        for i, error in enumerate(errors, 1):
            desc = f"{i}. [{error.severity.value.upper()}] {error.category}/{error.subcategory}"
            desc += f"\n   Location: characters {error.location[0]}-{error.location[1]}"
            desc += f"\n   Issue: {error.description}"
            if error.suggestion:
                desc += f"\n   Suggested fix: {error.suggestion}"
            error_descriptions.append(desc)

        errors_text = "\n\n".join(error_descriptions)

        prompt = f"""You are a professional translator and post-editor.

Your task: Fix the detected errors in the translation while keeping the meaning accurate and text natural.

## SOURCE TEXT ({task.source_lang}):
{task.source_text}

## CURRENT TRANSLATION ({task.target_lang}):
{task.translation}

## ERRORS TO FIX:
{errors_text}

## INSTRUCTIONS:
1. Fix ALL the errors listed above
2. Keep all other parts of the translation unchanged
3. Ensure the corrected translation is fluent and natural
4. Preserve the meaning of the source text
5. Use suggested fixes when provided

## OUTPUT:
Provide ONLY the corrected translation in {task.target_lang}, without any explanation or markdown formatting.
"""

        try:
            corrected = await self.llm_provider.complete(
                prompt, temperature=temperature, max_tokens=max_tokens
            )

            # Clean up response (remove markdown, extra whitespace)
            corrected = corrected.strip()

            # Remove markdown code blocks if present
            if corrected.startswith("```") and corrected.endswith("```"):
                lines = corrected.split("\n")
                corrected = "\n".join(lines[1:-1])

            return corrected.strip()

        except Exception as e:
            logger.error(f"Failed to generate corrected translation: {e}")
            # Fallback: return original translation
            return task.translation

    async def correct_and_reevaluate(
        self,
        task: TranslationTask,
        errors: list[ErrorAnnotation],
        orchestrator: Any,  # AgentOrchestrator type
        correction_level: str = "light",
        max_iterations: int = 2,
    ) -> tuple[str, list[Any]]:
        """Correct translation and re-evaluate iteratively.

        Args:
            task: Translation task
            errors: Initial errors
            orchestrator: QA orchestrator for re-evaluation
            correction_level: "light" or "full"
            max_iterations: Maximum correction iterations

        Returns:
            Tuple of (final_translation, list_of_reports)

        Example:
            >>> corrector = AutoCorrector(provider)
            >>> final, reports = await corrector.correct_and_reevaluate(
            ...     task, errors, orchestrator, correction_level="light"
            ... )
            >>> print(f"MQM improvement: {reports[0].mqm_score} -> {reports[-1].mqm_score}")
        """
        current_translation = task.translation
        current_errors = errors
        reports = []

        for iteration in range(max_iterations):
            logger.info(f"Correction iteration {iteration + 1}/{max_iterations}")

            # Apply corrections
            corrected = await self.auto_correct(task, current_errors, correction_level)

            # Check if anything changed
            if corrected == current_translation:
                logger.info("No changes made, stopping iteration")
                break

            # Re-evaluate corrected translation
            corrected_task = TranslationTask(
                source_text=task.source_text,
                translation=corrected,
                source_lang=task.source_lang,
                target_lang=task.target_lang,
                context=task.context,
            )

            report = await orchestrator.evaluate(corrected_task)
            reports.append(report)

            logger.info(
                f"Iteration {iteration + 1}: MQM {report.mqm_score:.1f}, "
                f"Errors: {report.error_count}"
            )

            # Check if we've reached acceptable quality
            if report.status == "pass" or report.error_count == 0:
                logger.info("Quality threshold reached, stopping iteration")
                return corrected, reports

            # Prepare for next iteration
            current_translation = corrected
            current_errors = report.errors

            # Check if improvement stagnated
            if len(reports) >= 2:
                prev_score = reports[-2].mqm_score
                curr_score = reports[-1].mqm_score
                if abs(curr_score - prev_score) < 1.0:
                    logger.info("Improvement stagnated, stopping iteration")
                    break

        return current_translation, reports

    def get_correction_summary(
        self, original: str, corrected: str, errors: list[ErrorAnnotation]
    ) -> dict[str, Any]:
        """Get summary of corrections applied.

        Args:
            original: Original translation
            corrected: Corrected translation
            errors: List of errors that were fixed

        Returns:
            Dictionary with correction statistics

        Example:
            >>> summary = corrector.get_correction_summary(original, corrected, errors)
            >>> print(f"Fixed {summary['errors_fixed']} errors")
        """
        return {
            "errors_fixed": len(errors),
            "error_breakdown": {
                "critical": sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL),
                "major": sum(1 for e in errors if e.severity == ErrorSeverity.MAJOR),
                "minor": sum(1 for e in errors if e.severity == ErrorSeverity.MINOR),
            },
            "categories": list({e.category for e in errors}),
            "original_length": len(original),
            "corrected_length": len(corrected),
            "length_change": len(corrected) - len(original),
        }
