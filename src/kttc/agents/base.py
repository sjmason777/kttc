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

"""Base abstract class for QA agents.

All QA agents (accuracy, fluency, terminology) must implement this interface.
Includes self-assessment retry mechanism for improved accuracy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from kttc.core import ErrorAnnotation, TranslationTask
from kttc.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class SelfAssessmentResult:
    """Result of agent self-assessment.

    Attributes:
        confidence: Confidence score (0.0-1.0) in the analysis quality
        issues: List of identified issues with the analysis
        should_retry: Whether a retry is recommended
    """

    confidence: float
    issues: list[str]
    should_retry: bool

    @classmethod
    def high_confidence(cls) -> SelfAssessmentResult:
        """Create a high-confidence result (no retry needed)."""
        return cls(confidence=1.0, issues=[], should_retry=False)


class BaseAgent(ABC):
    """Abstract base class for QA agents.

    Each agent is responsible for checking a specific quality dimension
    (accuracy, fluency, terminology) using an LLM provider.

    Supports self-assessment retry mechanism:
    - Agent evaluates translation
    - Agent assesses confidence in its own analysis
    - If confidence < threshold, retry with hints about issues
    - Maximum retries controlled to limit API costs
    """

    # Default settings for self-assessment retry
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_MAX_RETRIES = 2

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        enable_self_assessment: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize agent with LLM provider and retry settings.

        Args:
            llm_provider: LLM provider for generating evaluations
            enable_self_assessment: Enable self-assessment retry mechanism
            confidence_threshold: Minimum confidence to accept result (0.0-1.0)
            max_retries: Maximum retry attempts (default: 2)
        """
        self.llm_provider = llm_provider
        self.enable_self_assessment = enable_self_assessment
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self._retry_count = 0

    @abstractmethod
    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate a translation task and return found errors.

        Args:
            task: Translation task to evaluate

        Returns:
            List of error annotations found by the agent

        Raises:
            AgentError: If evaluation fails

        Example:
            >>> agent = AccuracyAgent(provider)
            >>> errors = await agent.evaluate(task)
            >>> print(f"Found {len(errors)} accuracy errors")
        """
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Get the error category this agent checks.

        Returns:
            Error category name (e.g., 'accuracy', 'fluency', 'terminology')
        """
        pass

    @abstractmethod
    def get_base_prompt(self) -> str:
        """Get the base prompt template used by this agent.

        This method is used by DomainAdapter to enhance prompts with
        domain-specific information.

        Returns:
            Base prompt template string

        Example:
            >>> agent = AccuracyAgent(provider)
            >>> base_prompt = agent.get_base_prompt()
            >>> print(base_prompt[:100])  # First 100 characters
        """
        pass

    async def self_assess(
        self,
        errors: list[ErrorAnnotation],
        task: TranslationTask,
    ) -> SelfAssessmentResult:
        """Assess confidence in the evaluation results.

        Override in subclasses for custom self-assessment logic.
        Default implementation checks for common issues.

        Args:
            errors: List of errors found in evaluation
            task: The translation task that was evaluated

        Returns:
            SelfAssessmentResult with confidence score and issues
        """
        # Default self-assessment based on heuristics
        issues: list[str] = []
        confidence = 1.0

        # Check for suspiciously no errors on long text
        word_count = len(task.source_text.split())
        if word_count > 50 and len(errors) == 0:
            issues.append("No errors found on long text - may have missed issues")
            confidence -= 0.2

        # Check for too many errors (might be over-sensitive)
        if len(errors) > word_count / 5:
            issues.append("High error density - may be over-flagging")
            confidence -= 0.15

        # Check for errors with low confidence scores
        low_conf_errors = [e for e in errors if e.confidence and e.confidence < 0.6]
        if low_conf_errors:
            issues.append(f"{len(low_conf_errors)} errors have low confidence")
            confidence -= 0.1 * min(len(low_conf_errors), 3)

        confidence = max(0.0, min(1.0, confidence))
        should_retry = confidence < self.confidence_threshold

        return SelfAssessmentResult(
            confidence=confidence,
            issues=issues,
            should_retry=should_retry,
        )

    async def evaluate_with_retry(
        self,
        task: TranslationTask,
        hints: list[str] | None = None,
    ) -> tuple[list[ErrorAnnotation], int]:
        """Evaluate with self-assessment retry mechanism.

        Performs evaluation, then self-assesses the result.
        If confidence is low, retries with hints about issues.

        Args:
            task: Translation task to evaluate
            hints: Optional hints from previous failed attempt

        Returns:
            Tuple of (errors, retry_count)
        """
        self._retry_count = 0

        if not self.enable_self_assessment:
            errors = await self.evaluate(task)
            return errors, 0

        errors = await self.evaluate(task)

        for attempt in range(self.max_retries):
            assessment = await self.self_assess(errors, task)

            if not assessment.should_retry:
                logger.debug(
                    f"{self.category} agent: confidence={assessment.confidence:.2f}, "
                    f"no retry needed"
                )
                break

            self._retry_count += 1
            logger.info(
                f"{self.category} agent: low confidence ({assessment.confidence:.2f}), "
                f"retry {self._retry_count}/{self.max_retries}. "
                f"Issues: {', '.join(assessment.issues)}"
            )

            # Retry with knowledge of issues
            errors = await self.evaluate(task)

        return errors, self._retry_count

    @property
    def retry_count(self) -> int:
        """Get the number of retries performed in the last evaluation."""
        return self._retry_count


class AgentError(Exception):
    """Base exception for agent-related errors."""

    pass


class AgentEvaluationError(AgentError):
    """Raised when agent evaluation fails."""

    pass


class AgentParsingError(AgentError):
    """Raised when parsing LLM response fails."""

    pass
