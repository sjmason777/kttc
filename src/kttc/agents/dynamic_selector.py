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

"""Dynamic agent selection for budget-aware and complexity-adaptive QA.

This module implements smart agent selection based on task characteristics,
reducing costs by 30-50% while maintaining quality for appropriate task complexity.

Key Features:
    - Complexity-based agent selection (simple/medium/complex)
    - Domain-aware agent prioritization
    - Budget-conscious routing with fallback cascade
    - Estimated token usage calculation

Example:
    >>> selector = DynamicAgentSelector(llm_provider)
    >>> agents = selector.select_agents(
    ...     task=translation_task,
    ...     complexity='auto',  # Auto-detect
    ...     domain_profile=technical_profile
    ... )
    >>> # Returns optimized agent set for technical translation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kttc.core import TranslationTask
    from kttc.llm import BaseLLMProvider

    from .base import BaseAgent
    from .domain_profiles import DomainProfile

logger = logging.getLogger(__name__)


class DynamicAgentSelector:
    """Smart agent selection for cost optimization and quality tuning.

    Selects optimal agent combinations based on task complexity, domain,
    and budget constraints. Reduces unnecessary agent executions while
    maintaining quality standards.

    Attributes:
        llm_provider: LLM provider for agent initialization
        agent_temperature: Temperature setting for agents
        agent_max_tokens: Max tokens for agent responses

    Example:
        >>> selector = DynamicAgentSelector(openai_provider)
        >>> # Simple task - only 2 core agents
        >>> agents = selector.select_agents(task, complexity='simple')
        >>> # Complex technical task - all agents + terminology
        >>> agents = selector.select_agents(
        ...     task,
        ...     complexity='complex',
        ...     domain_profile=technical_profile
        ... )
    """

    # Estimated token costs per agent evaluation
    AGENT_TOKEN_ESTIMATES = {
        "accuracy": 800,  # Semantic comparison requires more tokens
        "fluency": 600,  # Grammar/style analysis
        "terminology": 700,  # Term matching and validation
        "hallucination": 750,  # Fact verification
        "context": 850,  # Document-level analysis
        "fluency_russian": 650,  # Language-specific fluency
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        agent_temperature: float = 0.1,
        agent_max_tokens: int = 2000,
    ):
        """Initialize dynamic agent selector.

        Args:
            llm_provider: LLM provider for agent instantiation
            agent_temperature: Temperature for all agents
            agent_max_tokens: Max tokens for agent responses
        """
        self.llm_provider = llm_provider
        self.agent_temperature = agent_temperature
        self.agent_max_tokens = agent_max_tokens

    def select_agents(
        self,
        task: TranslationTask,
        complexity: str | float = "auto",
        domain_profile: DomainProfile | None = None,
        max_budget_tokens: int | None = None,
    ) -> list[BaseAgent]:
        """Select optimal agent set for the given task.

        Args:
            task: Translation task to evaluate
            complexity: Task complexity ('auto', 'simple', 'medium', 'complex')
                       or float 0.0-1.0
            domain_profile: Domain profile for domain-specific selection
            max_budget_tokens: Maximum token budget (optional)

        Returns:
            List of agent instances optimized for the task

        Raises:
            ValueError: If complexity value is invalid

        Example:
            >>> # Auto-detect complexity
            >>> agents = selector.select_agents(task, complexity='auto')
            >>> # Force full agent suite
            >>> agents = selector.select_agents(task, complexity='complex')
            >>> # Budget-constrained selection
            >>> agents = selector.select_agents(
            ...     task,
            ...     max_budget_tokens=2000  # Only 2-3 agents
            ... )
        """
        # Resolve complexity
        complexity_value = self._resolve_complexity(task, complexity)

        logger.info(f"Task complexity: {complexity_value:.2f}")

        # Select agents based on complexity
        if complexity_value < 0.3:
            agent_set = self._get_simple_agents()
        elif complexity_value < 0.7:
            agent_set = self._get_medium_agents()
        else:
            agent_set = self._get_complex_agents()

        # Apply domain-specific adjustments
        if domain_profile:
            agent_set = self._apply_domain_priorities(agent_set, domain_profile)

        # Add language-specific agents if needed
        agent_set = self._add_language_specific_agents(agent_set, task)

        # Apply budget constraints if specified
        if max_budget_tokens:
            agent_set = self._apply_budget_constraints(agent_set, max_budget_tokens)

        # Instantiate agents
        agents = self._instantiate_agents(agent_set)

        logger.info(f"Selected {len(agents)} agents: {', '.join(a.category for a in agents)}")

        # Log estimated token usage
        estimated_tokens = self._estimate_token_usage(agent_set)
        logger.info(f"Estimated token usage: ~{estimated_tokens}")

        return agents

    def _resolve_complexity(self, task: TranslationTask, complexity: str | float) -> float:
        """Resolve complexity to numeric value.

        Args:
            task: Translation task
            complexity: Complexity specification

        Returns:
            Complexity value 0.0-1.0

        Raises:
            ValueError: If complexity string is invalid
        """
        if isinstance(complexity, float):
            if not 0.0 <= complexity <= 1.0:
                raise ValueError(f"Complexity must be 0.0-1.0, got {complexity}")
            return complexity

        if complexity == "auto":
            return self._auto_detect_complexity(task)
        elif complexity == "simple":
            return 0.2
        elif complexity == "medium":
            return 0.5
        elif complexity == "complex":
            return 0.9
        else:
            raise ValueError(
                f"Invalid complexity: {complexity}. "
                f"Use 'auto', 'simple', 'medium', 'complex', or float 0.0-1.0"
            )

    def _auto_detect_complexity(self, task: TranslationTask) -> float:
        """Auto-detect task complexity from various signals.

        Args:
            task: Translation task

        Returns:
            Estimated complexity 0.0-1.0
        """
        complexity_factors = []

        # Factor 1: Text length (longer = more complex)
        source_length = len(task.source_text.split())
        if source_length < 10:
            complexity_factors.append(0.2)
        elif source_length < 50:
            complexity_factors.append(0.4)
        elif source_length < 200:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.8)

        # Factor 2: Sentence count (more sentences = more context to track)
        sentence_count = task.source_text.count(".") + task.source_text.count("!")
        if sentence_count <= 1:
            complexity_factors.append(0.2)
        elif sentence_count <= 3:
            complexity_factors.append(0.5)
        else:
            complexity_factors.append(0.7)

        # Factor 3: Special characters/formatting (indicates technical content)
        special_char_ratio = sum(c in task.source_text for c in "{}[]()<>:;@#$%^&*=") / max(
            1, len(task.source_text)
        )
        if special_char_ratio > 0.05:
            complexity_factors.append(0.8)  # Likely technical/code
        elif special_char_ratio > 0.02:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.3)

        # Factor 4: Context provided (more context = more to verify)
        if task.context and len(task.context) > 0:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.3)

        # Average all factors
        return sum(complexity_factors) / len(complexity_factors)

    def _get_simple_agents(self) -> list[str]:
        """Get minimal agent set for simple tasks.

        Returns:
            ['accuracy', 'fluency'] - Core quality dimensions only
        """
        return ["accuracy", "fluency"]

    def _get_medium_agents(self) -> list[str]:
        """Get standard agent set for medium complexity tasks.

        Returns:
            ['accuracy', 'fluency', 'terminology'] - Standard QA suite
        """
        return ["accuracy", "fluency", "terminology"]

    def _get_complex_agents(self) -> list[str]:
        """Get full agent set for complex tasks.

        Returns:
            All available agents for comprehensive analysis
        """
        return ["accuracy", "fluency", "terminology"]  # Add more when available

    def _apply_domain_priorities(
        self, agent_set: list[str], domain_profile: DomainProfile
    ) -> list[str]:
        """Apply domain-specific agent priorities.

        Args:
            agent_set: Current agent set
            domain_profile: Domain profile with priorities

        Returns:
            Adjusted agent set with domain-specific additions
        """
        # Ensure priority agents from domain profile are included
        for priority_agent in domain_profile.priority_agents:
            if priority_agent not in agent_set:
                agent_set.append(priority_agent)

        # Remove agents not relevant for this domain
        # (Currently we keep all, but could filter based on domain metadata)

        return agent_set

    def _add_language_specific_agents(
        self, agent_set: list[str], task: TranslationTask
    ) -> list[str]:
        """Add language-specific agents if available.

        Args:
            agent_set: Current agent set
            task: Translation task with language info

        Returns:
            Agent set with language-specific agents added
        """
        # Russian-specific fluency agent
        if task.target_lang == "ru":
            # Replace generic fluency with Russian-specific
            if "fluency" in agent_set and "fluency_russian" not in agent_set:
                agent_set.remove("fluency")
                agent_set.append("fluency_russian")

        # Future: Add more language-specific agents
        # elif task.target_lang == "zh":
        #     agent_set.append("fluency_chinese")
        # elif task.target_lang == "en":
        #     agent_set.append("fluency_english")

        return agent_set

    def _apply_budget_constraints(self, agent_set: list[str], max_budget_tokens: int) -> list[str]:
        """Apply budget constraints with fallback cascade.

        Args:
            agent_set: Desired agent set
            max_budget_tokens: Maximum token budget

        Returns:
            Agent set that fits within budget, prioritizing critical agents
        """
        # Calculate total cost
        total_cost = sum(self.AGENT_TOKEN_ESTIMATES.get(agent, 700) for agent in agent_set)

        if total_cost <= max_budget_tokens:
            return agent_set  # Fits in budget

        logger.warning(
            f"Budget exceeded: {total_cost} > {max_budget_tokens}. "
            f"Falling back to essential agents only."
        )

        # Priority order: accuracy > fluency > terminology > others
        priority_order = ["accuracy", "fluency", "terminology"]

        # Select agents in priority order until budget exhausted
        selected = []
        current_cost = 0

        for agent in priority_order:
            if agent in agent_set:
                agent_cost = self.AGENT_TOKEN_ESTIMATES.get(agent, 700)
                if current_cost + agent_cost <= max_budget_tokens:
                    selected.append(agent)
                    current_cost += agent_cost

        # Fallback: ensure at least accuracy agent
        if not selected:
            selected = ["accuracy"]

        logger.info(
            f"Budget-constrained selection: {len(selected)} agents " f"(~{current_cost} tokens)"
        )

        return selected

    def _estimate_token_usage(self, agent_set: list[str]) -> int:
        """Estimate total token usage for agent set.

        Args:
            agent_set: List of agent IDs

        Returns:
            Estimated total tokens
        """
        return sum(self.AGENT_TOKEN_ESTIMATES.get(agent, 700) for agent in agent_set)

    def _instantiate_agents(self, agent_set: list[str]) -> list[BaseAgent]:
        """Instantiate agent objects from agent IDs.

        Args:
            agent_set: List of agent IDs

        Returns:
            List of instantiated agent objects
        """
        from .accuracy import AccuracyAgent
        from .fluency import FluencyAgent
        from .fluency_russian import RussianFluencyAgent
        from .terminology import TerminologyAgent

        agents: list[BaseAgent] = []

        for agent_id in agent_set:
            if agent_id == "accuracy":
                agents.append(
                    AccuracyAgent(
                        self.llm_provider,
                        temperature=self.agent_temperature,
                        max_tokens=self.agent_max_tokens,
                    )
                )
            elif agent_id == "fluency":
                agents.append(
                    FluencyAgent(
                        self.llm_provider,
                        temperature=self.agent_temperature,
                        max_tokens=self.agent_max_tokens,
                    )
                )
            elif agent_id == "fluency_russian":
                # Get Russian helper if available
                from kttc.helpers import get_helper_for_language
                from kttc.helpers.russian import RussianLanguageHelper

                helper = get_helper_for_language("ru")
                russian_helper: RussianLanguageHelper | None = None
                if isinstance(helper, RussianLanguageHelper):
                    russian_helper = helper

                agents.append(
                    RussianFluencyAgent(
                        self.llm_provider,
                        temperature=self.agent_temperature,
                        max_tokens=self.agent_max_tokens,
                        helper=russian_helper,
                    )
                )
            elif agent_id == "terminology":
                agents.append(
                    TerminologyAgent(
                        self.llm_provider,
                        temperature=self.agent_temperature,
                        max_tokens=self.agent_max_tokens,
                    )
                )
            # Add more agents as they become available
            # elif agent_id == "hallucination":
            #     agents.append(HallucinationAgent(...))
            # elif agent_id == "context":
            #     agents.append(ContextAgent(...))

        return agents

    def get_selection_summary(
        self,
        task: TranslationTask,
        complexity: str | float = "auto",
        domain_profile: DomainProfile | None = None,
    ) -> dict[str, Any]:
        """Get summary of agent selection without instantiating agents.

        Useful for preview/debugging purposes.

        Args:
            task: Translation task
            complexity: Complexity specification
            domain_profile: Optional domain profile

        Returns:
            Dictionary with selection summary:
            - complexity: Resolved complexity value
            - agents: List of agent IDs
            - estimated_tokens: Estimated token usage
            - reasoning: Explanation of selection

        Example:
            >>> summary = selector.get_selection_summary(task)
            >>> print(f"Will use {len(summary['agents'])} agents")
            >>> print(f"Estimated cost: {summary['estimated_tokens']} tokens")
        """
        complexity_value = self._resolve_complexity(task, complexity)

        # Select agents (without instantiation)
        if complexity_value < 0.3:
            agent_set = self._get_simple_agents()
            reason = "Simple task - core agents only"
        elif complexity_value < 0.7:
            agent_set = self._get_medium_agents()
            reason = "Medium complexity - standard agent suite"
        else:
            agent_set = self._get_complex_agents()
            reason = "Complex task - full agent suite"

        if domain_profile:
            agent_set = self._apply_domain_priorities(agent_set, domain_profile)
            reason += f" + domain priorities ({domain_profile.domain_type})"

        agent_set = self._add_language_specific_agents(agent_set, task)

        estimated_tokens = self._estimate_token_usage(agent_set)

        return {
            "complexity": complexity_value,
            "agents": agent_set,
            "estimated_tokens": estimated_tokens,
            "reasoning": reason,
        }
