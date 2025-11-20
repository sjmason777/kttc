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
import logging

from kttc.core import ErrorAnnotation, QAReport, TranslationTask
from kttc.core.mqm import MQMScorer
from kttc.helpers import get_helper_for_language
from kttc.llm import BaseLLMProvider

from .accuracy import AccuracyAgent
from .base import AgentEvaluationError, BaseAgent
from .consensus import WeightedConsensus
from .domain_profiles import DomainDetector, get_domain_profile
from .fluency import FluencyAgent
from .fluency_hindi import HindiFluencyAgent
from .fluency_persian import PersianFluencyAgent
from .fluency_russian import RussianFluencyAgent
from .terminology import TerminologyAgent

logger = logging.getLogger(__name__)


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
        use_weighted_consensus: bool = True,
        agent_weights: dict[str, float] | None = None,
        enable_domain_adaptation: bool = True,
        enable_dynamic_selection: bool = True,
    ):
        """Initialize orchestrator with LLM provider and configuration.

        Args:
            llm_provider: LLM provider for agent evaluations
            quality_threshold: Minimum MQM score to pass (default: 95.0)
            agent_temperature: Temperature setting for all agents (default: 0.1)
            agent_max_tokens: Max tokens for agent responses (default: 2000)
            use_weighted_consensus: Enable weighted consensus mode (default: True)
            agent_weights: Custom agent trust weights (overrides defaults if provided)
            enable_domain_adaptation: Enable domain-adaptive agent selection (default: True)
            enable_dynamic_selection: Enable dynamic agent selection for cost optimization (default: True)
        """
        self.llm_provider = llm_provider
        self.agent_temperature = agent_temperature
        self.agent_max_tokens = agent_max_tokens
        self.use_weighted_consensus = use_weighted_consensus
        self.enable_domain_adaptation = enable_domain_adaptation
        self.enable_dynamic_selection = enable_dynamic_selection

        # Core agents (used when dynamic selection is disabled)
        self.agents: list[BaseAgent] = [
            AccuracyAgent(llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens),
            FluencyAgent(llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens),
            TerminologyAgent(
                llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens
            ),
        ]
        self.scorer = MQMScorer()
        self.quality_threshold = quality_threshold

        # Initialize weighted consensus system
        self.consensus = WeightedConsensus(agent_weights=agent_weights, mqm_scorer=self.scorer)

        # Initialize domain detector for adaptive agent selection
        self.domain_detector = DomainDetector() if enable_domain_adaptation else None

        # Initialize dynamic agent selector for cost optimization (Phase 4)
        from .dynamic_selector import DynamicAgentSelector

        self.dynamic_selector = (
            DynamicAgentSelector(
                llm_provider,
                agent_temperature=agent_temperature,
                agent_max_tokens=agent_max_tokens,
            )
            if enable_dynamic_selection
            else None
        )

    def _get_language_specific_agents(self, task: TranslationTask) -> list[BaseAgent]:
        """Get language-specific agents based on target language.

        Automatically creates appropriate agents with NLP helpers when available.

        Args:
            task: Translation task with target language info

        Returns:
            List of language-specific agent instances
        """
        language_agents: list[BaseAgent] = []

        # Russian-specific fluency agent with NLP helper
        if task.target_lang == "ru":
            # Try to get Russian NLP helper
            from kttc.helpers.russian import RussianLanguageHelper

            helper = get_helper_for_language("ru")

            # Cast to RussianLanguageHelper or None
            russian_helper: RussianLanguageHelper | None = None
            if isinstance(helper, RussianLanguageHelper):
                russian_helper = helper

            if russian_helper and russian_helper.is_available():
                logger.info(
                    "Using RussianFluencyAgent with MAWO NLP helper (mawo-pymorphy3 + mawo-razdel)"
                )
            else:
                logger.info("Using RussianFluencyAgent in LLM-only mode (MAWO NLP not available)")

            language_agents.append(
                RussianFluencyAgent(
                    self.llm_provider,
                    temperature=self.agent_temperature,
                    max_tokens=self.agent_max_tokens,
                    helper=russian_helper,  # Pass properly typed helper
                )
            )

        # Hindi-specific fluency agent with Indic NLP + Stanza + Spello
        elif task.target_lang == "hi":
            # Try to get Hindi NLP helper
            from kttc.helpers.hindi import HindiLanguageHelper

            helper = get_helper_for_language("hi")

            # Cast to HindiLanguageHelper or None
            hindi_helper: HindiLanguageHelper | None = None
            if isinstance(helper, HindiLanguageHelper):
                hindi_helper = helper

            if hindi_helper and hindi_helper.is_available():
                logger.info("Using HindiFluencyAgent with Indic NLP + Stanza + Spello")
            else:
                logger.info("Using HindiFluencyAgent in LLM-only mode (helpers not available)")

            language_agents.append(
                HindiFluencyAgent(
                    self.llm_provider,
                    temperature=self.agent_temperature,
                    max_tokens=self.agent_max_tokens,
                    helper=hindi_helper,  # Pass properly typed helper
                )
            )

        # Persian-specific fluency agent with DadmaTools
        elif task.target_lang == "fa":
            # Try to get Persian NLP helper
            from kttc.helpers.persian import PersianLanguageHelper

            helper = get_helper_for_language("fa")

            # Cast to PersianLanguageHelper or None
            persian_helper: PersianLanguageHelper | None = None
            if isinstance(helper, PersianLanguageHelper):
                persian_helper = helper

            if persian_helper and persian_helper.is_available():
                logger.info("Using PersianFluencyAgent with DadmaTools v2 (spaCy-based all-in-one)")
            else:
                logger.info("Using PersianFluencyAgent in LLM-only mode (DadmaTools not available)")

            language_agents.append(
                PersianFluencyAgent(
                    self.llm_provider,
                    temperature=self.agent_temperature,
                    max_tokens=self.agent_max_tokens,
                    helper=persian_helper,  # Pass properly typed helper
                )
            )

        return language_agents

    async def evaluate(self, task: TranslationTask) -> QAReport:
        """Evaluate translation quality using all agents in parallel.

        Runs all configured agents concurrently, including language-specific agents,
        aggregates their error findings, calculates MQM score, and generates a
        comprehensive quality report.

        When weighted consensus is enabled, also calculates:
        - Confidence level based on agent agreement
        - Individual agent scores
        - Agreement metrics

        Args:
            task: Translation task to evaluate

        Returns:
            QAReport with aggregated errors, MQM score, and optional consensus metrics

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
            >>> if report.confidence:
            ...     print(f"Confidence: {report.confidence:.2f}")
        """
        try:
            # Domain-adaptive agent selection (Phase 3)
            detected_domain = None
            domain_profile = None
            domain_confidence = None

            if self.enable_domain_adaptation and self.domain_detector:
                detected_domain = self.domain_detector.detect_domain(
                    task.source_text, task.target_lang, task.context
                )
                domain_profile = get_domain_profile(detected_domain)
                domain_confidence = self.domain_detector.get_domain_confidence(
                    task.source_text, detected_domain
                )

                logger.info(
                    f"Domain detection: {detected_domain} " f"(confidence: {domain_confidence:.2f})"
                )

            # Select agents: dynamic (Phase 4) or static
            if self.dynamic_selector:
                # Dynamic agent selection based on complexity and domain
                complexity = task.context.get("complexity", "auto") if task.context else "auto"
                all_agents = self.dynamic_selector.select_agents(
                    task, complexity=complexity, domain_profile=domain_profile
                )
            else:
                # Static agent selection (original behavior)
                language_agents = self._get_language_specific_agents(task)
                all_agents = self.agents + language_agents

            # Run all agents in parallel using asyncio.gather
            results = await asyncio.gather(*[agent.evaluate(task) for agent in all_agents])

            # Build agent results dictionary (agent category -> errors)
            agent_results: dict[str, list[ErrorAnnotation]] = {}
            all_errors: list[ErrorAnnotation] = []

            for agent, agent_errors in zip(all_agents, results):
                category = agent.category
                agent_results[category] = agent_errors
                all_errors.extend(agent_errors)

            # Calculate word count for MQM scoring
            word_count = len(task.source_text.split())

            # Use domain-specific weights if available
            domain_weights = domain_profile.agent_weights if domain_profile else None

            # Calculate MQM score and consensus metrics
            if self.use_weighted_consensus and len(agent_results) > 0:
                # Use weighted consensus calculation with domain-specific weights
                consensus_data = self.consensus.calculate_weighted_score(
                    agent_results, word_count, agent_weights_override=domain_weights
                )

                mqm_score = consensus_data["weighted_mqm_score"]
                confidence = consensus_data["confidence"]
                agent_agreement = consensus_data["agent_agreement"]
                agent_scores = consensus_data["agent_scores"]
                consensus_metadata = {
                    "agent_weights_used": consensus_data["agent_weights_used"],
                    "total_weight": consensus_data["total_weight"],
                    **consensus_data["metadata"],
                }

                logger.info(
                    f"Weighted consensus: MQM={mqm_score:.2f}, "
                    f"confidence={confidence:.2f}, "
                    f"agreement={agent_agreement:.2f}"
                )
            else:
                # Use traditional simple aggregation
                mqm_score = self.scorer.calculate_score(all_errors, word_count)
                confidence = None
                agent_agreement = None
                agent_scores = None
                consensus_metadata = None

            # Use domain-specific threshold if available
            quality_threshold = (
                domain_profile.quality_threshold if domain_profile else self.quality_threshold
            )

            # Determine pass/fail status
            status = "pass" if mqm_score >= quality_threshold else "fail"

            # Build domain information for report
            domain_details = None
            if detected_domain and domain_profile:
                domain_details = {
                    "detected_domain": detected_domain,
                    "domain_confidence": domain_confidence,
                    "domain_complexity": domain_profile.complexity,
                    "quality_threshold_used": quality_threshold,
                    "domain_description": domain_profile.description,
                }

            # Build comprehensive quality report
            return QAReport(
                task=task,
                mqm_score=mqm_score,
                errors=all_errors,
                status=status,
                comet_score=None,
                confidence=confidence,
                agent_agreement=agent_agreement,
                agent_scores=agent_scores,
                consensus_metadata=consensus_metadata,
                agent_details=domain_details,  # Add domain info
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

        When weighted consensus is enabled, also calculates confidence and agreement metrics.

        Args:
            task: Translation task to evaluate

        Returns:
            Tuple of (QAReport, dict mapping agent category to its errors)

        Example:
            >>> report, breakdown = await orchestrator.evaluate_with_breakdown(task)
            >>> print(f"Accuracy errors: {len(breakdown['accuracy'])}")
            >>> print(f"Fluency errors: {len(breakdown['fluency'])}")
            >>> print(f"Terminology errors: {len(breakdown['terminology'])}")
            >>> if report.confidence:
            ...     print(f"Confidence: {report.confidence:.2f}")
        """
        try:
            # Domain-adaptive agent selection (Phase 3)
            detected_domain = None
            domain_profile = None
            domain_confidence = None

            if self.enable_domain_adaptation and self.domain_detector:
                detected_domain = self.domain_detector.detect_domain(
                    task.source_text, task.target_lang, task.context
                )
                domain_profile = get_domain_profile(detected_domain)
                domain_confidence = self.domain_detector.get_domain_confidence(
                    task.source_text, detected_domain
                )

            # Select agents: dynamic (Phase 4) or static
            if self.dynamic_selector:
                # Dynamic agent selection based on complexity and domain
                complexity = task.context.get("complexity", "auto") if task.context else "auto"
                all_agents = self.dynamic_selector.select_agents(
                    task, complexity=complexity, domain_profile=domain_profile
                )
            else:
                # Static agent selection (original behavior)
                language_agents = self._get_language_specific_agents(task)
                all_agents = self.agents + language_agents

            # Run all agents in parallel
            results = await asyncio.gather(*[agent.evaluate(task) for agent in all_agents])

            # Build breakdown by agent category
            breakdown: dict[str, list[ErrorAnnotation]] = {}
            all_errors: list[ErrorAnnotation] = []

            for agent, agent_errors in zip(all_agents, results):
                category = agent.category
                breakdown[category] = agent_errors
                all_errors.extend(agent_errors)

            # Calculate word count
            word_count = len(task.source_text.split())

            # Use domain-specific weights if available
            domain_weights = domain_profile.agent_weights if domain_profile else None

            # Calculate MQM score and consensus metrics
            if self.use_weighted_consensus and len(breakdown) > 0:
                # Use weighted consensus with domain-specific weights
                consensus_data = self.consensus.calculate_weighted_score(
                    breakdown, word_count, agent_weights_override=domain_weights
                )

                mqm_score = consensus_data["weighted_mqm_score"]
                confidence = consensus_data["confidence"]
                agent_agreement = consensus_data["agent_agreement"]
                agent_scores = consensus_data["agent_scores"]
                consensus_metadata = {
                    "agent_weights_used": consensus_data["agent_weights_used"],
                    "total_weight": consensus_data["total_weight"],
                    **consensus_data["metadata"],
                }
            else:
                # Use traditional aggregation
                mqm_score = self.scorer.calculate_score(all_errors, word_count)
                confidence = None
                agent_agreement = None
                agent_scores = None
                consensus_metadata = None

            # Use domain-specific threshold if available
            quality_threshold = (
                domain_profile.quality_threshold if domain_profile else self.quality_threshold
            )

            # Determine status
            status = "pass" if mqm_score >= quality_threshold else "fail"

            # Build domain information for report
            domain_details = None
            if detected_domain and domain_profile:
                domain_details = {
                    "detected_domain": detected_domain,
                    "domain_confidence": domain_confidence,
                    "domain_complexity": domain_profile.complexity,
                    "quality_threshold_used": quality_threshold,
                    "domain_description": domain_profile.description,
                }

            # Build report
            report = QAReport(
                task=task,
                mqm_score=mqm_score,
                errors=all_errors,
                status=status,
                comet_score=None,
                confidence=confidence,
                agent_agreement=agent_agreement,
                agent_scores=agent_scores,
                consensus_metadata=consensus_metadata,
                agent_details=domain_details,  # Add domain info
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
