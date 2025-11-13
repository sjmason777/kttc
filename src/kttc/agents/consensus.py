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

"""Weighted consensus system for multi-agent quality evaluation.

This module implements weighted consensus mechanisms inspired by multi-agent
thinking systems, where different agents have different trust weights based
on their reliability and importance for the task.

Key concepts:
- Agent Trust Weights: Different agents have different reliability levels
- Weighted Consensus: Final score is weighted average of agent scores
- Confidence Calculation: Based on agreement/variance between agents
- Agent Agreement Metrics: Measure how much agents agree with each other

References:
    - Multi-agent thinking systems with weighted consensus
    - Research from semantic graph processor architecture
"""

from __future__ import annotations

import logging
from typing import Any

from kttc.core import ErrorAnnotation
from kttc.core.mqm import MQMScorer

logger = logging.getLogger(__name__)


class WeightedConsensus:
    """Weighted consensus mechanism for multi-agent QA evaluation.

    Implements weighted averaging of agent results where different agents
    have different trust weights. This allows more reliable agents to have
    greater influence on the final quality assessment.

    Agent trust weights philosophy:
    - Accuracy: 1.0 (highest - semantic correctness is critical)
    - Hallucination: 0.95 (very high - false additions are dangerous)
    - Fluency: 0.9 (high - naturalness is important)
    - Terminology: 0.85 (important but domain-specific)
    - Context: 0.8 (moderate - depends on document type)
    - Style: 0.7 (moderate - subjective aspects)

    Example:
        >>> consensus = WeightedConsensus()
        >>> agent_results = {
        ...     'accuracy': [error1, error2],
        ...     'fluency': [error3],
        ...     'terminology': []
        ... }
        >>> result = consensus.calculate_weighted_score(agent_results, word_count=100)
        >>> print(f"MQM: {result['weighted_mqm_score']:.2f}")
        >>> print(f"Confidence: {result['confidence']:.2f}")
    """

    # Default agent trust weights (can be overridden per task type)
    DEFAULT_AGENT_WEIGHTS = {
        "accuracy": 1.0,  # Highest trust - semantic correctness critical
        "hallucination": 0.95,  # Very high - false additions dangerous
        "fluency": 0.9,  # High trust - naturalness important
        "fluency_russian": 0.9,  # Same as fluency - language-specific variant
        "terminology": 0.85,  # Important but domain-specific
        "context": 0.8,  # Moderate - depends on document type
        "style": 0.7,  # Moderate - more subjective
    }

    def __init__(
        self, agent_weights: dict[str, float] | None = None, mqm_scorer: MQMScorer | None = None
    ):
        """Initialize weighted consensus system.

        Args:
            agent_weights: Custom agent trust weights (overrides defaults)
            mqm_scorer: MQM scoring engine (creates new if not provided)
        """
        self.agent_weights = agent_weights or self.DEFAULT_AGENT_WEIGHTS
        self.scorer = mqm_scorer or MQMScorer()

    def calculate_weighted_score(
        self,
        agent_results: dict[str, list[ErrorAnnotation]],
        word_count: int,
        agent_weights_override: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Calculate weighted MQM score from multiple agent results.

        Uses weighted averaging where each agent's contribution is multiplied
        by its trust weight. Also calculates confidence based on agent agreement.

        Args:
            agent_results: Dictionary mapping agent category to its errors
            word_count: Number of words in source text
            agent_weights_override: Optional domain-specific weights to override defaults

        Returns:
            Dictionary containing:
            - weighted_mqm_score: Final weighted MQM score
            - confidence: Confidence level (0.0-1.0) based on agent agreement
            - agent_agreement: Agreement metric (inverse of variance)
            - agent_scores: Individual MQM score per agent
            - agent_weights_used: Trust weights used for each agent
            - total_weight: Sum of all agent weights
            - metadata: Additional metrics and statistics

        Raises:
            ValueError: If word_count <= 0 or agent_results is empty

        Example:
            >>> consensus = WeightedConsensus()
            >>> results = {
            ...     'accuracy': [ErrorAnnotation(...)],
            ...     'fluency': []
            ... }
            >>> score_data = consensus.calculate_weighted_score(results, 100)
            >>> print(f"Score: {score_data['weighted_mqm_score']:.2f}")
            >>> print(f"Confidence: {score_data['confidence']:.2f}")
        """
        if word_count <= 0:
            raise ValueError("word_count must be greater than 0")

        if not agent_results:
            raise ValueError("agent_results cannot be empty")

        # Use domain-specific weights if provided, otherwise use configured weights
        active_weights = agent_weights_override if agent_weights_override else self.agent_weights

        # Calculate individual MQM scores for each agent
        agent_scores: dict[str, float] = {}
        agent_weights_used: dict[str, float] = {}

        for agent_id, errors in agent_results.items():
            # Calculate MQM score for this agent's errors
            agent_mqm = self.scorer.calculate_score(errors, word_count)
            agent_scores[agent_id] = agent_mqm

            # Get agent weight (default to 1.0 for unknown agents)
            agent_weights_used[agent_id] = active_weights.get(agent_id, 1.0)

        # Calculate weighted average MQM score
        weighted_sum = 0.0
        total_weight = 0.0

        for agent_id, agent_mqm in agent_scores.items():
            weight = agent_weights_used[agent_id]
            weighted_sum += agent_mqm * weight
            total_weight += weight

        weighted_mqm_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate confidence based on agent agreement
        confidence = self._calculate_confidence(agent_scores, agent_weights_used)

        # Calculate agreement metric (inverse of variance)
        agreement = self._calculate_agreement(agent_scores)

        # Build metadata
        metadata = {
            "num_agents": len(agent_results),
            "total_errors": sum(len(errors) for errors in agent_results.values()),
            "score_variance": self._calculate_variance(list(agent_scores.values())),
            "score_std_dev": self._calculate_std_dev(list(agent_scores.values())),
            "min_agent_score": min(agent_scores.values()) if agent_scores else 0.0,
            "max_agent_score": max(agent_scores.values()) if agent_scores else 0.0,
        }

        return {
            "weighted_mqm_score": round(weighted_mqm_score, 2),
            "confidence": round(confidence, 2),
            "agent_agreement": round(agreement, 2),
            "agent_scores": {k: round(v, 2) for k, v in agent_scores.items()},
            "agent_weights_used": agent_weights_used,
            "total_weight": round(total_weight, 2),
            "metadata": metadata,
        }

    def _calculate_confidence(
        self, agent_scores: dict[str, float], agent_weights: dict[str, float]
    ) -> float:
        """Calculate confidence based on weighted agent agreement.

        High agreement between agents = high confidence
        High variance = low confidence

        The formula weights agreement by agent trust:
        - High variance among high-trust agents → lower confidence
        - High variance among low-trust agents → less impact

        Args:
            agent_scores: Individual agent MQM scores
            agent_weights: Trust weight for each agent

        Returns:
            Confidence score (0.0-1.0)
        """
        if len(agent_scores) < 2:
            # Single agent - moderate confidence (0.7)
            return 0.7

        scores = list(agent_scores.values())
        weights = [agent_weights[agent_id] for agent_id in agent_scores.keys()]

        # Calculate weighted mean
        weighted_mean = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        # Calculate weighted variance
        weighted_variance = sum(
            w * (s - weighted_mean) ** 2 for s, w in zip(scores, weights)
        ) / sum(weights)

        # Convert variance to confidence
        # Low variance (< 1.0) → high confidence (~1.0)
        # High variance (> 25.0) → low confidence (~0.0)
        # Using linear decay with clamping
        confidence = max(0.0, min(1.0, (1.0 - weighted_variance / 100.0)))

        # Boost confidence if agents generally agree (all scores > 90 or all < 80)
        if all(s >= 90 for s in scores) or all(s < 80 for s in scores):
            confidence = min(1.0, confidence * 1.1)

        return confidence

    def _calculate_agreement(self, agent_scores: dict[str, float]) -> float:
        """Calculate agent agreement metric (inverse of coefficient of variation).

        High agreement = low variance relative to mean
        Low agreement = high variance relative to mean

        Args:
            agent_scores: Individual agent MQM scores

        Returns:
            Agreement score (0.0-1.0), where 1.0 = perfect agreement
        """
        if len(agent_scores) < 2:
            return 1.0  # Single agent - perfect agreement with itself

        scores = list(agent_scores.values())
        mean = sum(scores) / len(scores)

        if mean == 0:
            return 1.0  # All zeros - perfect agreement

        std_dev = self._calculate_std_dev(scores)
        coefficient_of_variation = std_dev / mean

        # Convert CV to agreement score (0-1 scale)
        # Low CV (< 0.05) → high agreement (~1.0)
        # High CV (> 0.20) → low agreement (~0.0)
        agreement = max(0.0, min(1.0, 1.0 - coefficient_of_variation * 5.0))

        return agreement

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values.

        Args:
            values: List of numeric values

        Returns:
            Variance value
        """
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _calculate_std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation of a list of values.

        Args:
            values: List of numeric values

        Returns:
            Standard deviation value
        """
        variance = self._calculate_variance(values)
        return float(variance**0.5)

    def calculate_confidence_only(
        self, agent_scores: dict[str, float], agent_weights: dict[str, float] | None = None
    ) -> float:
        """Calculate confidence score from agent scores without full calculation.

        Useful for quick confidence checks without recalculating MQM scores.

        Args:
            agent_scores: Individual agent MQM scores
            agent_weights: Optional agent weights (uses defaults if not provided)

        Returns:
            Confidence score (0.0-1.0)

        Example:
            >>> consensus = WeightedConsensus()
            >>> scores = {'accuracy': 95.0, 'fluency': 93.0, 'terminology': 94.0}
            >>> confidence = consensus.calculate_confidence_only(scores)
            >>> print(f"Confidence: {confidence:.2f}")
        """
        weights = agent_weights or self.agent_weights
        weights_used = {agent: weights.get(agent, 1.0) for agent in agent_scores.keys()}
        return self._calculate_confidence(agent_scores, weights_used)
