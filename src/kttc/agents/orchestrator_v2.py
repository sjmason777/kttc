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

"""Enhanced Agent Orchestrator with memory support.

Coordinates multiple QA agents with support for:
- Translation Memory integration
- Terminology Base validation
- Hallucination detection
- Context awareness
"""

import asyncio
import logging
from typing import Any, cast

from kttc.core import ErrorAnnotation, QAReport, TranslationTask
from kttc.core.mqm import MQMScorer
from kttc.llm import BaseLLMProvider

from .accuracy import AccuracyAgent
from .base import AgentEvaluationError, BaseAgent
from .context import ContextAgent
from .fluency import FluencyAgent
from .hallucination import HallucinationAgent
from .terminology import TerminologyAgent

logger = logging.getLogger(__name__)


class EnhancedAgentOrchestrator:
    """Enhanced orchestrator with memory support.

    Coordinates all QA agents and integrates:
    - Translation Memory for context
    - Terminology Base for validation
    - Hallucination detection
    - Document-level context awareness

    Example:
        >>> from kttc.memory import TranslationMemory, TerminologyBase
        >>>
        >>> # Initialize components
        >>> tm = TranslationMemory()
        >>> await tm.initialize()
        >>> termbase = TerminologyBase()
        >>> await termbase.initialize()
        >>>
        >>> # Create orchestrator
        >>> orchestrator = EnhancedAgentOrchestrator(
        ...     llm_provider=provider,
        ...     translation_memory=tm,
        ...     terminology_base=termbase
        ... )
        >>>
        >>> # Evaluate translation
        >>> report = await orchestrator.evaluate(task)
        >>> print(f"MQM: {report.mqm_score}")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        quality_threshold: float = 95.0,
        agent_temperature: float = 0.1,
        agent_max_tokens: int = 2000,
        translation_memory: Any = None,  # TranslationMemory instance
        terminology_base: Any = None,  # TerminologyBase instance
        enable_hallucination_detection: bool = True,
        enable_context_checking: bool = True,
    ):
        """Initialize enhanced orchestrator.

        Args:
            llm_provider: LLM provider for agents
            quality_threshold: Minimum MQM score to pass (default: 95.0)
            agent_temperature: Temperature for all agents (default: 0.1)
            agent_max_tokens: Max tokens for agent responses (default: 2000)
            translation_memory: TranslationMemory instance (optional)
            terminology_base: TerminologyBase instance (optional)
            enable_hallucination_detection: Enable hallucination agent (default: True)
            enable_context_checking: Enable context agent (default: True)
        """
        # Core agents
        self.agents: list[BaseAgent] = [
            AccuracyAgent(llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens),
            FluencyAgent(llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens),
            TerminologyAgent(
                llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens
            ),
        ]

        # Additional agents
        if enable_hallucination_detection:
            self.agents.append(
                HallucinationAgent(
                    llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens
                )
            )
            logger.info("Hallucination detection enabled")

        # Context agent (optional)
        self.context_agent: ContextAgent | None = None
        if enable_context_checking:
            self.context_agent = ContextAgent(
                llm_provider, temperature=agent_temperature, max_tokens=agent_max_tokens
            )
            self.agents.append(self.context_agent)
            logger.info("Context checking enabled")

        # Components
        self.scorer = MQMScorer()
        self.quality_threshold = quality_threshold
        self.translation_memory = translation_memory
        self.terminology_base = terminology_base

        logger.info(f"Enhanced orchestrator initialized with {len(self.agents)} agents")

    async def evaluate(
        self,
        task: TranslationTask,
        use_translation_memory: bool = True,
        use_terminology_base: bool = True,
    ) -> QAReport:
        """Evaluate translation quality with all enabled features.

        Args:
            task: Translation task to evaluate
            use_translation_memory: Whether to check TM (default: True)
            use_terminology_base: Whether to validate against termbase (default: True)

        Returns:
            Comprehensive QA report with all metrics

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        try:
            # Run all agents in parallel
            logger.info(f"Running {len(self.agents)} agents in parallel...")
            agent_results = await asyncio.gather(*[agent.evaluate(task) for agent in self.agents])

            # Flatten errors
            all_errors: list[ErrorAnnotation] = []
            for agent_errors in agent_results:
                all_errors.extend(agent_errors)

            # Terminology Base validation
            if use_terminology_base and self.terminology_base is not None:
                logger.info("Validating against terminology base...")
                term_violations = await self.terminology_base.validate_translation(
                    source_text=task.source_text,
                    translation=task.translation,
                    source_lang=task.source_lang,
                    target_lang=task.target_lang,
                    domain=task.context.get("domain") if task.context else None,
                )

                # Convert violations to errors
                for violation in term_violations:
                    from kttc.core import ErrorSeverity

                    all_errors.append(
                        ErrorAnnotation(
                            category="terminology",
                            subcategory="glossary_violation",
                            severity=ErrorSeverity(violation.severity),
                            location=(0, 10),  # Approximate
                            description=(
                                f"Term '{violation.source_term}' should be translated as: "
                                f"{', '.join(violation.expected_terms)}"
                            ),
                            suggestion=(
                                violation.expected_terms[0] if violation.expected_terms else None
                            ),
                        )
                    )

            # Calculate MQM score
            word_count = len(task.source_text.split())
            mqm_score = self.scorer.calculate_score(all_errors, word_count)

            # Translation Memory (for future reference)
            if use_translation_memory and self.translation_memory is not None:
                # If quality is high, add to TM
                if mqm_score >= 90.0:
                    logger.info("Adding high-quality translation to TM...")
                    try:
                        await self.translation_memory.add_translation(
                            source=task.source_text,
                            translation=task.translation,
                            source_lang=task.source_lang,
                            target_lang=task.target_lang,
                            mqm_score=mqm_score,
                            domain=task.context.get("domain") if task.context else None,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add to TM: {e}")

            # Determine status
            status = "pass" if mqm_score >= self.quality_threshold else "fail"

            # Build report
            return QAReport(
                task=task,
                mqm_score=mqm_score,
                errors=all_errors,
                status=status,
            )

        except AgentEvaluationError as e:
            raise AgentEvaluationError(f"Enhanced orchestrator evaluation failed: {e}") from e
        except Exception as e:
            raise AgentEvaluationError(f"Unexpected error in orchestrator: {e}") from e

    async def get_tm_suggestions(
        self, source: str, source_lang: str, target_lang: str, limit: int = 3
    ) -> list[Any]:
        """Get Translation Memory suggestions for source text.

        Args:
            source: Source text
            source_lang: Source language code
            target_lang: Target language code
            limit: Maximum number of suggestions

        Returns:
            List of TM search results
        """
        if self.translation_memory is None:
            return []

        try:
            results = await self.translation_memory.search_similar(
                source=source,
                source_lang=source_lang,
                target_lang=target_lang,
                threshold=0.80,
                limit=limit,
            )

            return cast(list[Any], results)

        except Exception as e:
            logger.warning(f"TM search failed: {e}")
            return []

    def set_quality_threshold(self, threshold: float) -> None:
        """Update quality threshold.

        Args:
            threshold: New MQM score threshold (0-100)

        Raises:
            ValueError: If threshold is not between 0 and 100
        """
        if not 0 <= threshold <= 100:
            raise ValueError(f"Threshold must be between 0 and 100, got {threshold}")
        self.quality_threshold = threshold
        logger.info(f"Quality threshold updated to {threshold}")

    def set_document_context(self, full_document: str) -> None:
        """Set document context for Context Agent.

        Args:
            full_document: Full document text for context
        """
        if self.context_agent is not None:
            self.context_agent.set_document_context(full_document)
            logger.info("Document context set for Context Agent")

    def add_segment_to_context(
        self, source: str, translation: str, segment_id: str | None = None
    ) -> None:
        """Add segment to Context Agent for consistency tracking.

        Args:
            source: Source text segment
            translation: Translation segment
            segment_id: Optional segment identifier
        """
        if self.context_agent is not None:
            self.context_agent.add_segment(source, translation, segment_id)

    def clear_context(self) -> None:
        """Clear document context from Context Agent."""
        if self.context_agent is not None:
            self.context_agent.clear_context()
            logger.info("Context cleared")
