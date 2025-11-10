"""Agent Orchestrator for coordinating multiple QA agents.

Coordinates parallel execution of multiple specialized QA agents
(Accuracy, Fluency, Terminology) and aggregates their results into
a comprehensive quality assessment report.

Example:
    >>> from kttc.agents import AgentOrchestrator
    >>> from kttc.llm import OpenAIProvider
    >>> from kttc.core import TranslationTask
    >>>
    >>> provider = OpenAIProvider(api_key="...")
    >>> orchestrator = AgentOrchestrator(provider)
    >>> task = TranslationTask(
    ...     source_text="Hello, world!",
    ...     translation="Hola, mundo!",
    ...     source_lang="en",
    ...     target_lang="es"
    ... )
    >>> report = await orchestrator.evaluate(task)
    >>> print(f"MQM Score: {report.mqm_score}")
"""

import asyncio

from kttc.core import ErrorAnnotation, QAReport, TranslationTask
from kttc.core.mqm import MQMScorer
from kttc.llm import BaseLLMProvider

from .accuracy import AccuracyAgent
from .base import AgentEvaluationError, BaseAgent
from .fluency import FluencyAgent
from .terminology import TerminologyAgent


class AgentOrchestrator:
    """Orchestrates multiple QA agents for comprehensive translation evaluation.

    Coordinates parallel execution of specialized agents (Accuracy, Fluency,
    Terminology) and aggregates their findings into a unified quality report
    with MQM scoring.

    Attributes:
        agents: List of specialized QA agents to run
        scorer: MQM scoring engine for calculating final quality score
        quality_threshold: Minimum MQM score to consider translation as passing

    Example:
        >>> provider = OpenAIProvider(api_key="...")
        >>> orchestrator = AgentOrchestrator(provider, quality_threshold=95.0)
        >>> report = await orchestrator.evaluate(task)
        >>> if report.status == "pass":
        ...     print("Translation quality acceptable")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        quality_threshold: float = 95.0,
        agent_temperature: float = 0.1,
        agent_max_tokens: int = 2000,
    ):
        """Initialize orchestrator with LLM provider and configuration.

        Args:
            llm_provider: LLM provider for agent evaluations
            quality_threshold: Minimum MQM score to pass (default: 95.0)
            agent_temperature: Temperature setting for all agents (default: 0.1)
            agent_max_tokens: Max tokens for agent responses (default: 2000)
        """
        self.agents: list[BaseAgent] = [
            AccuracyAgent(llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens),
            FluencyAgent(llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens),
            TerminologyAgent(
                llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens
            ),
        ]
        self.scorer = MQMScorer()
        self.quality_threshold = quality_threshold

    async def evaluate(self, task: TranslationTask) -> QAReport:
        """Evaluate translation quality using all agents in parallel.

        Runs all configured agents concurrently, aggregates their error findings,
        calculates MQM score, and generates a comprehensive quality report.

        Args:
            task: Translation task to evaluate

        Returns:
            QAReport with aggregated errors and MQM score

        Raises:
            AgentEvaluationError: If agent evaluation fails

        Example:
            >>> task = TranslationTask(
            ...     source_text="The API uses REST architecture",
            ...     translation="La API usa arquitectura REST",
            ...     source_lang="en",
            ...     target_lang="es"
            ... )
            >>> report = await orchestrator.evaluate(task)
            >>> print(f"Found {len(report.errors)} errors")
            >>> print(f"MQM Score: {report.mqm_score}")
        """
        try:
            # Run all agents in parallel using asyncio.gather
            results = await asyncio.gather(*[agent.evaluate(task) for agent in self.agents])

            # Flatten error lists from all agents
            all_errors: list[ErrorAnnotation] = []
            for agent_errors in results:
                all_errors.extend(agent_errors)

            # Calculate word count for MQM scoring
            word_count = len(task.source_text.split())

            # Calculate MQM score based on aggregated errors
            mqm_score = self.scorer.calculate_score(all_errors, word_count)

            # Determine pass/fail status
            status = "pass" if mqm_score >= self.quality_threshold else "fail"

            # Build comprehensive quality report
            return QAReport(
                task=task, mqm_score=mqm_score, errors=all_errors, status=status, comet_score=None
            )

        except AgentEvaluationError as e:
            raise AgentEvaluationError(f"Orchestrator evaluation failed: {e}") from e
        except Exception as e:
            raise AgentEvaluationError(f"Unexpected error in orchestrator: {e}") from e

    async def evaluate_with_breakdown(
        self, task: TranslationTask
    ) -> tuple[QAReport, dict[str, list[ErrorAnnotation]]]:
        """Evaluate translation and return report with per-agent error breakdown.

        Similar to evaluate() but also returns a dictionary mapping agent categories
        to their specific errors for detailed analysis.

        Args:
            task: Translation task to evaluate

        Returns:
            Tuple of (QAReport, dict mapping agent category to its errors)

        Example:
            >>> report, breakdown = await orchestrator.evaluate_with_breakdown(task)
            >>> print(f"Accuracy errors: {len(breakdown['accuracy'])}")
            >>> print(f"Fluency errors: {len(breakdown['fluency'])}")
            >>> print(f"Terminology errors: {len(breakdown['terminology'])}")
        """
        try:
            # Run all agents in parallel
            results = await asyncio.gather(*[agent.evaluate(task) for agent in self.agents])

            # Build breakdown by agent category
            breakdown: dict[str, list[ErrorAnnotation]] = {}
            all_errors: list[ErrorAnnotation] = []

            for agent, agent_errors in zip(self.agents, results):
                category = agent.category
                breakdown[category] = agent_errors
                all_errors.extend(agent_errors)

            # Calculate MQM score
            word_count = len(task.source_text.split())
            mqm_score = self.scorer.calculate_score(all_errors, word_count)

            # Determine status
            status = "pass" if mqm_score >= self.quality_threshold else "fail"

            # Build report
            report = QAReport(
                task=task, mqm_score=mqm_score, errors=all_errors, status=status, comet_score=None
            )

            return report, breakdown

        except AgentEvaluationError as e:
            raise AgentEvaluationError(f"Orchestrator evaluation with breakdown failed: {e}") from e
        except Exception as e:
            raise AgentEvaluationError(f"Unexpected error in orchestrator breakdown: {e}") from e

    def set_quality_threshold(self, threshold: float) -> None:
        """Update the quality threshold for pass/fail determination.

        Args:
            threshold: New MQM score threshold (0-100)

        Raises:
            ValueError: If threshold is not between 0 and 100
        """
        if not 0 <= threshold <= 100:
            raise ValueError(f"Threshold must be between 0 and 100, got {threshold}")
        self.quality_threshold = threshold
