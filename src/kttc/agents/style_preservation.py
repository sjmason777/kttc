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

"""Style Preservation Agent for literary translation evaluation.

Evaluates how well stylistic features are preserved in translation,
with special attention to intentional deviations from standard language.

Based on:
- LiTransProQA (Zhang et al., 2025) - authorial voice evaluation
- MAS-LitEval (Kim et al., 2025) - stylistic consistency scoring
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider, LLMError

from .base import AgentEvaluationError, BaseAgent

if TYPE_CHECKING:
    from kttc.style import StyleProfile

logger = logging.getLogger(__name__)


# Prompt template for style preservation evaluation
STYLE_PRESERVATION_PROMPT = """You are an expert literary translation evaluator specializing in authorial voice and style preservation.

Your task is to evaluate whether the translation preserves the stylistic features of the source text.

IMPORTANT: Some source texts intentionally deviate from standard grammar/style for artistic effect. These deviations should be PRESERVED in translation, not "corrected".

SOURCE TEXT ({source_lang}):
{source_text}

TRANSLATION ({target_lang}):
{translation}

{style_context}

EVALUATION QUESTIONS (answer each with YES/NO/PARTIALLY and brief explanation):

1. Does the translation preserve the author's distinctive tone and voice?
2. Are literary devices (repetition, unusual syntax, rhythm) maintained?
3. If the source has intentional "errors" or unusual constructions, are they recreated in the translation?
4. Does the translation feel like it was written by the same author?
5. Is the overall rhythm and pacing of the prose preserved?
{additional_questions}

After answering, provide a STYLE PRESERVATION SCORE from 0-100.

Then list any STYLE ERRORS found (errors are cases where translator unnecessarily normalized or lost stylistic features):

Format each error as:
ERROR: [category] | [subcategory] | [severity: minor/major/critical] | [description]

Example:
ERROR: style_preservation | voice_loss | major | The source's deliberate redundancy "live a life" was normalized to just "live", losing the Platanov-style pleonasm.

If no style errors found, write: NO_STYLE_ERRORS_FOUND"""


class StylePreservationAgent(BaseAgent):
    """Agent for evaluating style preservation in literary translation.

    Uses LLM to assess whether stylistic features from the source
    are maintained in the translation, with special attention to
    intentional artistic deviations.

    Example:
        >>> provider = OpenAIProvider(api_key="...")
        >>> agent = StylePreservationAgent(provider)
        >>> task = TranslationTask(...)
        >>> errors = await agent.evaluate(task)
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2500,
        style_profile: StyleProfile | None = None,
    ):
        """Initialize StylePreservationAgent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
            style_profile: Pre-computed style profile (optional)
        """
        super().__init__(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.style_profile = style_profile

    @property
    def category(self) -> str:
        """Get error category this agent checks."""
        return "style_preservation"

    def get_base_prompt(self) -> str:
        """Get the base prompt template."""
        return STYLE_PRESERVATION_PROMPT

    def set_style_profile(self, profile: StyleProfile) -> None:
        """Set style profile for evaluation context.

        Args:
            profile: StyleProfile from StyleFingerprint analysis
        """
        self.style_profile = profile

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate style preservation and return found errors.

        Args:
            task: Translation task to evaluate

        Returns:
            List of style preservation error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        try:
            # Build style context from profile
            style_context = self._build_style_context()
            additional_questions = self._build_additional_questions()

            # Format prompt
            prompt = STYLE_PRESERVATION_PROMPT.format(
                source_text=task.source_text,
                translation=task.translation,
                source_lang=task.source_lang,
                target_lang=task.target_lang,
                style_context=style_context,
                additional_questions=additional_questions,
            )

            # Get LLM evaluation
            llm_response = await self.llm_provider.complete(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse errors from response
            errors = self._parse_style_errors(llm_response)

            logger.info(f"StylePreservationAgent found {len(errors)} style issues")
            return errors

        except LLMError as e:
            raise AgentEvaluationError(
                f"LLM evaluation failed for style preservation check: {e}"
            ) from e
        except Exception as e:
            raise AgentEvaluationError(
                f"Unexpected error during style preservation evaluation: {e}"
            ) from e

    def _build_style_context(self) -> str:
        """Build context string from style profile."""
        if not self.style_profile:
            return "STYLE CONTEXT: No pre-analysis available. Analyze both texts for stylistic features."

        context_parts = [
            "DETECTED STYLE FEATURES IN SOURCE:",
            f"- Overall deviation score: {self.style_profile.deviation_score:.2f} "
            f"({'HIGH - literary style detected' if self.style_profile.deviation_score > 0.3 else 'normal'})",
            f"- Style pattern: {self.style_profile.detected_pattern.value}",
            f"- Is literary: {self.style_profile.is_literary}",
        ]

        if self.style_profile.detected_deviations:
            context_parts.append("\nDETECTED INTENTIONAL DEVIATIONS (these should be preserved):")
            for dev in self.style_profile.detected_deviations:
                context_parts.append(f"- {dev.type.value}: {dev.interpretation}")
                if dev.examples:
                    context_parts.append(f"  Examples: {', '.join(dev.examples[:3])}")

        return "\n".join(context_parts)

    def _build_additional_questions(self) -> str:
        """Build additional questions based on detected deviations."""
        if not self.style_profile or not self.style_profile.detected_deviations:
            return ""

        questions = []
        seen_types = set()

        for dev in self.style_profile.detected_deviations:
            if dev.type.value in seen_types:
                continue
            seen_types.add(dev.type.value)

            if dev.type.value == "pleonasm":
                questions.append(
                    "6. Are deliberate redundancies/repetitions (pleonasms) preserved?"
                )
            elif dev.type.value == "skaz":
                questions.append("6. Is the folk/oral storytelling voice (skaz) maintained?")
            elif dev.type.value == "syntactic_inversion":
                questions.append("6. Are unusual word order patterns recreated in target language?")
            elif dev.type.value == "stream_of_consciousness":
                questions.append(
                    "6. Is the stream of consciousness flow preserved (long sentences, associative jumps)?"
                )
            elif dev.type.value == "register_mixing":
                questions.append("6. Is the mixing of formal/informal registers preserved?")

        return "\n".join(questions)

    def _parse_style_errors(self, response: str) -> list[ErrorAnnotation]:
        """Parse style errors from LLM response."""
        errors: list[ErrorAnnotation] = []

        if "NO_STYLE_ERRORS_FOUND" in response:
            return errors

        # Parse ERROR lines
        for line in response.split("\n"):
            line = line.strip()
            if not line.startswith("ERROR:"):
                continue

            try:
                # Format: ERROR: [category] | [subcategory] | [severity] | [description]
                parts = line[6:].split("|")
                if len(parts) < 4:
                    continue

                category = parts[0].strip()
                subcategory = parts[1].strip()
                severity_str = parts[2].strip().lower()
                description = parts[3].strip()

                # Map severity
                severity_map = {
                    "minor": ErrorSeverity.MINOR,
                    "major": ErrorSeverity.MAJOR,
                    "critical": ErrorSeverity.CRITICAL,
                }
                severity = severity_map.get(severity_str, ErrorSeverity.MINOR)

                # Only accept style_preservation category
                if category != "style_preservation":
                    category = "style_preservation"

                errors.append(
                    ErrorAnnotation(
                        category=category,
                        subcategory=subcategory,
                        severity=severity,
                        location=(0, 0),  # Style errors often span whole text
                        description=description,
                        suggestion=None,
                    )
                )

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse error line '{line}': {e}")
                continue

        return errors
