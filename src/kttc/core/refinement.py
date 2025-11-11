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

"""Iterative refinement loop for translation quality improvement.

Implements TEaR (Translate, Estimate, Refine) framework with intelligent
convergence detection and targeted error correction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel, Field

from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask

if TYPE_CHECKING:
    from kttc.agents import AgentOrchestrator
    from kttc.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class ConvergenceCheck(TypedDict):
    """Result from convergence check."""

    converged: bool
    reason: str


class RefinementResult(BaseModel):
    """Result from iterative refinement process.

    Contains final translation, history of iterations, and convergence info.
    """

    final_translation: str = Field(description="Best translation after refinement")
    iterations: int = Field(description="Number of refinement iterations performed")
    initial_score: float = Field(description="Initial MQM score before refinement")
    final_score: float = Field(description="Final MQM score after refinement")
    improvement: float = Field(description="MQM score improvement")
    qa_reports: list[QAReport] = Field(
        default_factory=list, description="QA reports from each iteration"
    )
    converged: bool = Field(default=False, description="Whether refinement converged to threshold")
    convergence_reason: str = Field(default="", description="Reason for stopping refinement")


class IterativeRefinement:
    """Intelligent iterative refinement strategy.

    Implements TEaR framework:
    1. Translate: Get initial translation (or provided translation)
    2. Estimate: Evaluate quality with QA agents
    3. Refine: Fix errors and iterate until convergence

    Convergence conditions:
    - MQM score reaches threshold
    - No significant improvement (< min_improvement)
    - Max iterations reached

    Example:
        >>> from kttc.core.refinement import IterativeRefinement
        >>> from kttc.agents import AgentOrchestrator
        >>>
        >>> refinement = IterativeRefinement(
        ...     llm_provider=llm,
        ...     max_iterations=3,
        ...     convergence_threshold=95.0
        ... )
        >>>
        >>> result = await refinement.refine_until_convergence(
        ...     task=task,
        ...     orchestrator=orchestrator
        ... )
        >>>
        >>> print(f"Improvement: {result.improvement:.1f} points")
        >>> print(f"Final score: {result.final_score:.1f}")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        max_iterations: int = 3,
        convergence_threshold: float = 95.0,
        min_improvement: float = 1.0,
    ):
        """Initialize iterative refinement.

        Args:
            llm_provider: LLM provider for generating corrections
            max_iterations: Maximum number of refinement iterations
            convergence_threshold: Target MQM score for convergence
            min_improvement: Minimum improvement to continue (MQM points)
        """
        self.llm = llm_provider
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_improvement = min_improvement

    async def refine_until_convergence(
        self, task: TranslationTask, orchestrator: AgentOrchestrator
    ) -> RefinementResult:
        """Refine translation iteratively until convergence.

        Args:
            task: Translation task with initial translation
            orchestrator: Agent orchestrator for QA evaluation

        Returns:
            RefinementResult with final translation and history
        """
        current_translation = task.translation
        qa_reports: list[QAReport] = []

        logger.info(
            f"Starting iterative refinement (max iterations: {self.max_iterations}, "
            f"target: {self.convergence_threshold})"
        )

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Evaluate current translation
            eval_task = TranslationTask(
                source_text=task.source_text,
                translation=current_translation,
                source_lang=task.source_lang,
                target_lang=task.target_lang,
                context=task.context,
            )

            report = await orchestrator.evaluate(eval_task)
            qa_reports.append(report)

            logger.info(f"MQM Score: {report.mqm_score:.2f}, Errors: {len(report.errors)}")

            # Check convergence conditions
            convergence_check = self._check_convergence(qa_reports, iteration)
            if convergence_check["converged"]:
                logger.info(f"Converged: {convergence_check['reason']}")
                return self._create_result(
                    final_translation=current_translation,
                    qa_reports=qa_reports,
                    converged=True,
                    convergence_reason=convergence_check["reason"],
                )

            # Refine based on errors
            if report.errors:
                current_translation = await self.apply_refinement(
                    source=task.source_text,
                    translation=current_translation,
                    errors=report.errors,
                    iteration=iteration,
                )
            else:
                logger.info("No errors found, stopping refinement")
                return self._create_result(
                    final_translation=current_translation,
                    qa_reports=qa_reports,
                    converged=True,
                    convergence_reason="No errors detected",
                )

        # Max iterations reached
        logger.info(f"Max iterations ({self.max_iterations}) reached")
        return self._create_result(
            final_translation=current_translation,
            qa_reports=qa_reports,
            converged=False,
            convergence_reason=f"Max iterations ({self.max_iterations}) reached",
        )

    def _check_convergence(
        self, qa_reports: list[QAReport], current_iteration: int
    ) -> ConvergenceCheck:
        """Check if refinement has converged.

        Args:
            qa_reports: List of QA reports from iterations
            current_iteration: Current iteration number

        Returns:
            ConvergenceCheck with 'converged' bool and 'reason' string
        """
        if not qa_reports:
            return {"converged": False, "reason": ""}

        latest_score = qa_reports[-1].mqm_score

        # Check threshold met
        if latest_score >= self.convergence_threshold:
            return {
                "converged": True,
                "reason": f"Threshold met: {latest_score:.2f} >= {self.convergence_threshold}",
            }

        # Check improvement stagnation (need at least 2 iterations)
        if current_iteration > 0 and len(qa_reports) >= 2:
            prev_score = qa_reports[-2].mqm_score
            improvement = latest_score - prev_score

            if abs(improvement) < self.min_improvement:
                return {
                    "converged": True,
                    "reason": f"Improvement stagnated: {improvement:+.2f} < {self.min_improvement}",
                }

        return {"converged": False, "reason": ""}

    async def apply_refinement(
        self, source: str, translation: str, errors: list[ErrorAnnotation], iteration: int
    ) -> str:
        """Apply targeted refinement based on error feedback.

        Prioritizes critical and major errors. Uses LLM to generate
        improved translation while preserving correct parts.

        Args:
            source: Source text
            translation: Current translation
            errors: List of detected errors
            iteration: Current iteration number

        Returns:
            Improved translation
        """
        # Prioritize critical and major errors
        critical_errors = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        major_errors = [e for e in errors if e.severity == ErrorSeverity.MAJOR]
        minor_errors = [e for e in errors if e.severity == ErrorSeverity.MINOR]

        # Limit errors to focus on (avoid overwhelming prompt)
        priority_errors = (critical_errors + major_errors)[:5]

        if not priority_errors:
            # Only minor errors, pick top 3
            priority_errors = minor_errors[:3]

        logger.info(
            f"Applying refinement for {len(priority_errors)} errors "
            f"(Critical: {len(critical_errors)}, Major: {len(major_errors)}, "
            f"Minor: {len(minor_errors)})"
        )

        # Build refinement prompt
        prompt = self._build_refinement_prompt(
            source=source,
            translation=translation,
            errors=priority_errors,
            iteration=iteration,
        )

        # Generate improved translation
        response = await self.llm.complete(prompt)
        improved = self._extract_translation(response)

        logger.info(f"Refinement applied (iteration {iteration + 1})")
        return improved

    def _build_refinement_prompt(
        self, source: str, translation: str, errors: list[ErrorAnnotation], iteration: int
    ) -> str:
        """Build refinement prompt for LLM.

        Args:
            source: Source text
            translation: Current translation
            errors: Errors to fix
            iteration: Current iteration

        Returns:
            Refinement prompt
        """
        prompt = f"""You are a professional translator improving a translation.

SOURCE TEXT:
{source}

CURRENT TRANSLATION:
{translation}

ITERATION: {iteration + 1}

ERRORS TO FIX:
"""

        for i, error in enumerate(errors, 1):
            prompt += (
                f"\n{i}. [{error.severity.value.upper()}] {error.category}/{error.subcategory}\n"
            )
            prompt += f"   Description: {error.description}\n"

            if error.suggestion:
                prompt += f"   Suggestion: {error.suggestion}\n"

            if error.location and error.location != (0, 0):
                start, end = error.location
                if 0 <= start < len(translation) and start < end <= len(translation):
                    problematic_text = translation[start:end]
                    prompt += f'   Problematic text: "{problematic_text}"\n'

        prompt += """

INSTRUCTIONS:
1. Fix ONLY the errors listed above
2. Preserve correct parts of the translation
3. Maintain natural flow and readability
4. Keep the same target language style
5. Do NOT add explanations or notes

Output ONLY the improved translation text, nothing else.

IMPROVED TRANSLATION:"""

        return prompt

    def _extract_translation(self, response: str) -> str:
        """Extract translation from LLM response.

        Handles various response formats (plain text, markdown, etc.)

        Args:
            response: LLM response

        Returns:
            Extracted translation
        """
        # Remove common wrappers
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```") and text.endswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            text = "\n".join(lines[1:-1])

        # Remove label if present
        if text.lower().startswith("improved translation:"):
            text = text[len("improved translation:") :].strip()
        elif text.lower().startswith("translation:"):
            text = text[len("translation:") :].strip()

        return text.strip()

    def _create_result(
        self,
        final_translation: str,
        qa_reports: list[QAReport],
        converged: bool,
        convergence_reason: str,
    ) -> RefinementResult:
        """Create refinement result object.

        Args:
            final_translation: Final translation text
            qa_reports: List of QA reports
            converged: Whether convergence was achieved
            convergence_reason: Reason for convergence/stopping

        Returns:
            RefinementResult object
        """
        initial_score = qa_reports[0].mqm_score if qa_reports else 0.0
        final_score = qa_reports[-1].mqm_score if qa_reports else 0.0
        improvement = final_score - initial_score

        return RefinementResult(
            final_translation=final_translation,
            iterations=len(qa_reports),
            initial_score=initial_score,
            final_score=final_score,
            improvement=improvement,
            qa_reports=qa_reports,
            converged=converged,
            convergence_reason=convergence_reason,
        )
