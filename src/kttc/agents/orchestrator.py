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

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from kttc.core import ErrorAnnotation, QAReport, TranslationTask

if TYPE_CHECKING:
    from kttc.agents.domain_profiles import DomainProfile
    from kttc.style import StyleFingerprint, StyleProfile
from kttc.core.mqm import MQMScorer
from kttc.helpers import get_helper_for_language
from kttc.llm import BaseLLMProvider

from .accuracy import AccuracyAgent
from .base import AgentEvaluationError, BaseAgent
from .consensus import WeightedConsensus
from .context import ContextAgent
from .domain_profiles import (
    DomainDetector,
    get_domain_profile,
    get_literary_profile_for_style,
)
from .fluency import FluencyAgent
from .fluency_chinese import ChineseFluencyAgent
from .fluency_hindi import HindiFluencyAgent
from .fluency_persian import PersianFluencyAgent
from .fluency_russian import RussianFluencyAgent
from .hallucination import HallucinationAgent
from .style_preservation import StylePreservationAgent
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

    # Mapping from agent names to agent classes
    # Note: Using Any type because agent subclasses have different __init__ signatures
    AGENT_CLASSES: dict[str, Any] = {
        "accuracy": AccuracyAgent,
        "fluency": FluencyAgent,
        "terminology": TerminologyAgent,
        "hallucination": HallucinationAgent,
        "context": ContextAgent,
        "style": StylePreservationAgent,
    }

    # Default core agents (used when no specific agents selected)
    DEFAULT_AGENTS = ["accuracy", "fluency", "terminology"]

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
        quick_mode: bool = False,
        selected_agents: list[str] | None = None,
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
            quick_mode: Enable quick mode with only 3 core agents (default: False)
            selected_agents: List of agent names to use (default: None = use core agents)
                             Valid names: accuracy, fluency, terminology, hallucination, context, style
        """
        self.llm_provider = llm_provider
        self.agent_temperature = agent_temperature
        self.agent_max_tokens = agent_max_tokens
        self.use_weighted_consensus = use_weighted_consensus
        self.enable_domain_adaptation = enable_domain_adaptation and not quick_mode
        self.enable_dynamic_selection = enable_dynamic_selection and not quick_mode
        self.quick_mode = quick_mode
        self.selected_agents = selected_agents

        # Build agents list based on selection or use defaults
        agents_to_use = selected_agents if selected_agents else self.DEFAULT_AGENTS
        self.agents: list[BaseAgent] = []
        for agent_name in agents_to_use:
            agent_name_lower = agent_name.lower()
            if agent_name_lower in self.AGENT_CLASSES:
                agent_class = self.AGENT_CLASSES[agent_name_lower]
                self.agents.append(
                    agent_class(
                        llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens
                    )
                )
            else:
                logger.warning(
                    f"Unknown agent '{agent_name}', skipping. Valid: {list(self.AGENT_CLASSES.keys())}"
                )
        self.scorer = MQMScorer()
        self.quality_threshold = quality_threshold

        # Initialize weighted consensus system
        self.consensus = WeightedConsensus(agent_weights=agent_weights, mqm_scorer=self.scorer)

        # Initialize domain detector for adaptive agent selection (disabled in quick mode)
        self.domain_detector = DomainDetector() if self.enable_domain_adaptation else None

        # Initialize style fingerprint analyzer for literary text detection
        self.style_analyzer = self._init_style_analyzer() if not quick_mode else None

        # Initialize dynamic agent selector for cost optimization (disabled in quick mode)
        from .dynamic_selector import DynamicAgentSelector

        self.dynamic_selector = (
            DynamicAgentSelector(
                llm_provider,
                agent_temperature=agent_temperature,
                agent_max_tokens=agent_max_tokens,
            )
            if self.enable_dynamic_selection
            else None
        )

    def _init_style_analyzer(self) -> StyleFingerprint | None:
        """Initialize style analyzer for literary text detection."""
        try:
            from kttc.style import StyleFingerprint as StyleFP

            return StyleFP()
        except ImportError:
            logger.warning("Style module not available, literary detection disabled")
            return None

    def _analyze_source_style(self, task: TranslationTask) -> StyleProfile | None:
        """Analyze source text style for automatic literary detection.

        Returns:
            StyleProfile or None if analysis not available
        """
        if not self.style_analyzer:
            return None

        try:

            profile = self.style_analyzer.analyze(task.source_text, task.source_lang)

            if profile.has_significant_deviations:
                logger.info(
                    f"Literary style detected: pattern={profile.detected_pattern.value}, "
                    f"deviation_score={profile.deviation_score:.2f}, "
                    f"deviations={len(profile.detected_deviations)}"
                )
            return profile
        except Exception as e:
            logger.warning(f"Style analysis failed: {e}")
            return None

    def _get_style_aware_profile(
        self, task: TranslationTask, style_profile: StyleProfile | None
    ) -> DomainProfile | None:
        """Get domain profile adjusted for style analysis results.

        If literary style is detected, returns appropriate literary profile
        instead of keyword-based domain detection.
        """
        if style_profile is None or not style_profile.has_significant_deviations:
            return None

        # Get literary profile based on detected style pattern
        literary_profile = get_literary_profile_for_style(
            style_profile.detected_pattern.value,
            style_profile.deviation_score,
        )

        logger.info(
            f"Using style-aware profile: {literary_profile.domain_type} "
            f"(fluency weight: {literary_profile.agent_weights.get('fluency', 1.0):.2f})"
        )

        return literary_profile

    def _get_style_preservation_agent(self, style_profile: StyleProfile | None) -> BaseAgent | None:
        """Create StylePreservationAgent if literary style detected."""
        if style_profile is None or not style_profile.has_significant_deviations:
            return None

        agent = StylePreservationAgent(
            self.llm_provider,
            temperature=self.agent_temperature,
            max_tokens=self.agent_max_tokens,
            style_profile=style_profile,
        )
        return agent

    def _create_language_agent(
        self,
        lang: str,
        agent_cls: type[BaseAgent],
        helper_cls: type,
        msg_avail: str,
        msg_fallback: str,
    ) -> BaseAgent:
        """Create a language-specific agent with appropriate helper."""
        helper = get_helper_for_language(lang)
        typed_helper = helper if isinstance(helper, helper_cls) else None

        if typed_helper and hasattr(typed_helper, "is_available") and typed_helper.is_available():
            logger.info(msg_avail)
        else:
            logger.info(msg_fallback)
            typed_helper = None

        # Language-specific agents accept these extra kwargs
        return agent_cls(  # type: ignore[call-arg]
            self.llm_provider,
            temperature=self.agent_temperature,
            max_tokens=self.agent_max_tokens,
            helper=typed_helper,
        )

    def _get_language_specific_agents(self, task: TranslationTask) -> list[BaseAgent]:
        """Get language-specific agents based on target language."""
        from kttc.helpers.chinese import ChineseLanguageHelper
        from kttc.helpers.hindi import HindiLanguageHelper
        from kttc.helpers.persian import PersianLanguageHelper
        from kttc.helpers.russian import RussianLanguageHelper

        lang = task.target_lang

        if lang == "ru":
            return [
                self._create_language_agent(
                    "ru",
                    RussianFluencyAgent,
                    RussianLanguageHelper,
                    "Using RussianFluencyAgent with MAWO NLP helper (mawo-pymorphy3 + mawo-razdel)",
                    "Using RussianFluencyAgent in LLM-only mode (MAWO NLP not available)",
                )
            ]

        if lang == "hi":
            return [
                self._create_language_agent(
                    "hi",
                    HindiFluencyAgent,
                    HindiLanguageHelper,
                    "Using HindiFluencyAgent with Indic NLP + Stanza + Spello",
                    "Using HindiFluencyAgent in LLM-only mode (helpers not available)",
                )
            ]

        if lang == "fa":
            return [
                self._create_language_agent(
                    "fa",
                    PersianFluencyAgent,
                    PersianLanguageHelper,
                    "Using PersianFluencyAgent with DadmaTools v2 (spaCy-based all-in-one)",
                    "Using PersianFluencyAgent in LLM-only mode (DadmaTools not available)",
                )
            ]

        if lang.startswith("zh"):
            return [
                self._create_language_agent(
                    "zh",
                    ChineseFluencyAgent,
                    ChineseLanguageHelper,
                    "Using ChineseFluencyAgent with HanLP + jieba + spaCy",
                    "Using ChineseFluencyAgent in LLM-only mode (NLP helpers not available)",
                )
            ]

        return []

    def _get_domain_context(
        self,
        task: TranslationTask,
        style_profile: StyleProfile | None,
        style_aware_profile: DomainProfile | None,
    ) -> tuple[str | None, DomainProfile | None, float | None]:
        """Get domain detection context for task evaluation."""
        if style_aware_profile:
            domain = style_aware_profile.domain_type
            confidence = style_profile.deviation_score if style_profile else 0.5
            logger.info(f"Style-aware domain: {domain} (deviation_score: {confidence:.2f})")
            return domain, style_aware_profile, confidence

        if self.enable_domain_adaptation and self.domain_detector:
            domain = self.domain_detector.detect_domain(
                task.source_text, task.target_lang, task.context
            )
            profile = get_domain_profile(domain)
            confidence = self.domain_detector.get_domain_confidence(task.source_text, domain)
            logger.info(f"Domain detection: {domain} (confidence: {confidence:.2f})")
            return domain, profile, confidence

        return None, None, None

    def _select_agents(
        self,
        task: TranslationTask,
        domain_profile: DomainProfile | None,
        style_profile: StyleProfile | None,
    ) -> list[BaseAgent]:
        """Select agents for task evaluation."""
        if self.dynamic_selector:
            complexity = task.context.get("complexity", "auto") if task.context else "auto"
            agents = list(
                self.dynamic_selector.select_agents(
                    task, complexity=complexity, domain_profile=domain_profile
                )
            )
        else:
            language_agents = self._get_language_specific_agents(task)
            agents = list(self.agents) + list(language_agents)

        style_agent = self._get_style_preservation_agent(style_profile)
        if style_agent:
            agents = [style_agent] + agents
            logger.info("Added StylePreservationAgent for literary text evaluation")

        return agents

    def _calculate_evaluation_scores(
        self,
        agent_results: dict[str, list[ErrorAnnotation]],
        all_errors: list[ErrorAnnotation],
        word_count: int,
        domain_profile: DomainProfile | None,
    ) -> tuple[float, float | None, float | None, dict[str, float] | None, dict[str, Any] | None]:
        """Calculate MQM scores and consensus metrics."""
        domain_weights = domain_profile.agent_weights if domain_profile else None

        if self.use_weighted_consensus and len(agent_results) > 0:
            consensus_data = self.consensus.calculate_weighted_score(
                agent_results, word_count, agent_weights_override=domain_weights
            )
            logger.info(
                f"Weighted consensus: MQM={consensus_data['weighted_mqm_score']:.2f}, "
                f"confidence={consensus_data['confidence']:.2f}, "
                f"agreement={consensus_data['agent_agreement']:.2f}"
            )
            return (
                consensus_data["weighted_mqm_score"],
                consensus_data["confidence"],
                consensus_data["agent_agreement"],
                consensus_data["agent_scores"],
                {
                    "agent_weights_used": consensus_data["agent_weights_used"],
                    "total_weight": consensus_data["total_weight"],
                    **consensus_data["metadata"],
                },
            )

        return self.scorer.calculate_score(all_errors, word_count), None, None, None, None

    def _build_domain_and_style_details(
        self,
        detected_domain: str | None,
        domain_profile: DomainProfile | None,
        domain_confidence: float | None,
        quality_threshold: float,
        style_profile: StyleProfile | None,
    ) -> dict[str, Any] | None:
        """Build domain and style details for report."""
        domain_details: dict[str, Any] | None = None

        if detected_domain and domain_profile:
            domain_details = {
                "detected_domain": detected_domain,
                "domain_confidence": domain_confidence,
                "domain_complexity": domain_profile.complexity,
                "quality_threshold_used": quality_threshold,
                "domain_description": domain_profile.description,
            }

        if style_profile and style_profile.has_significant_deviations:
            style_details = {
                "style_detected": True,
                "deviation_score": style_profile.deviation_score,
                "style_pattern": style_profile.detected_pattern.value,
                "is_literary": style_profile.is_literary,
                "deviations_count": len(style_profile.detected_deviations),
                "fluency_tolerance": style_profile.recommended_fluency_tolerance,
                "detected_features": [d.type.value for d in style_profile.detected_deviations],
            }
            if domain_details:
                domain_details["style_analysis"] = style_details
            else:
                domain_details = {"style_analysis": style_details}

        return domain_details

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
            # Style and domain analysis
            style_profile = self._analyze_source_style(task)
            style_aware_profile = self._get_style_aware_profile(task, style_profile)
            detected_domain, domain_profile, domain_confidence = self._get_domain_context(
                task, style_profile, style_aware_profile
            )

            # Select and run agents
            all_agents = self._select_agents(task, domain_profile, style_profile)
            results = await asyncio.gather(*[agent.evaluate(task) for agent in all_agents])

            # Collect agent results
            agent_results: dict[str, list[ErrorAnnotation]] = {}
            all_errors: list[ErrorAnnotation] = []
            for agent, agent_errors in zip(all_agents, results):
                agent_results[agent.category] = agent_errors
                all_errors.extend(agent_errors)

            # Calculate scores
            word_count = len(task.source_text.split())
            mqm_score, confidence, agent_agreement, agent_scores, consensus_metadata = (
                self._calculate_evaluation_scores(
                    agent_results, all_errors, word_count, domain_profile
                )
            )

            # Determine status and build details
            quality_threshold = (
                domain_profile.quality_threshold if domain_profile else self.quality_threshold
            )
            status = "pass" if mqm_score >= quality_threshold else "fail"
            domain_details = self._build_domain_and_style_details(
                detected_domain, domain_profile, domain_confidence, quality_threshold, style_profile
            )

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

    def _calculate_mqm_with_consensus_breakdown(
        self,
        breakdown: dict[str, list[ErrorAnnotation]],
        word_count: int,
        domain_weights: dict[str, float] | None,
        all_errors: list[ErrorAnnotation],
    ) -> tuple[float, float | None, float | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Calculate MQM score using consensus or traditional aggregation.

        Returns:
            Tuple of (mqm_score, confidence, agent_agreement, agent_scores, consensus_metadata)
        """
        if self.use_weighted_consensus and len(breakdown) > 0:
            consensus_data = self.consensus.calculate_weighted_score(
                breakdown, word_count, agent_weights_override=domain_weights
            )
            return (
                consensus_data["weighted_mqm_score"],
                consensus_data["confidence"],
                consensus_data["agent_agreement"],
                consensus_data["agent_scores"],
                {
                    "agent_weights_used": consensus_data["agent_weights_used"],
                    "total_weight": consensus_data["total_weight"],
                    **consensus_data["metadata"],
                },
            )
        return self.scorer.calculate_score(all_errors, word_count), None, None, None, None

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
            mqm_score, confidence, agent_agreement, agent_scores, consensus_metadata = (
                self._calculate_mqm_with_consensus_breakdown(
                    breakdown, word_count, domain_weights, all_errors
                )
            )

            # Use domain-specific threshold if available
            quality_threshold = (
                domain_profile.quality_threshold if domain_profile else self.quality_threshold
            )

            # Determine status
            status = "pass" if mqm_score >= quality_threshold else "fail"

            # Build domain information for report
            domain_details: dict[str, Any] | None = None
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
