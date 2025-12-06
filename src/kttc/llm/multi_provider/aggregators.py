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

"""Aggregation strategies for multi-provider ensemble results.

Implements various strategies for combining responses from multiple LLM providers:
- FirstSuccessAggregator: Return first successful response
- MajorityVoteAggregator: Majority voting for categorical outputs
- WeightedVoteAggregator: Weighted voting based on provider trust
- QualityScoreAggregator: Score-based selection

References:
    - ArXiv 2025 LLM Ensemble Survey (BUAA)
    - https://zhuanlan.zhihu.com/p/1896280002019972023
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregation operation.

    Attributes:
        response: Aggregated response text
        confidence: Confidence score (0.0-1.0)
        metadata: Additional aggregation metadata
    """

    response: str | None
    confidence: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass
class ProviderResponse:
    """Response from a single provider for aggregation.

    Attributes:
        provider_name: Name of the provider
        response: Response text
        weight: Provider weight for voting
        quality_score: Provider quality score
        latency: Response latency
    """

    provider_name: str
    response: str
    weight: float = 1.0
    quality_score: float = 0.8
    latency: float = 0.0


class BaseAggregator(ABC):
    """Base class for response aggregators.

    Aggregators combine responses from multiple LLM providers
    into a single final response.
    """

    @abstractmethod
    def aggregate(self, responses: list[ProviderResponse]) -> AggregationResult:
        """Aggregate responses from multiple providers.

        Args:
            responses: List of provider responses

        Returns:
            Aggregated result
        """
        ...

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio (0.0-1.0)
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _group_similar_responses(
        self,
        responses: list[ProviderResponse],
        threshold: float = 0.85,
    ) -> list[list[ProviderResponse]]:
        """Group responses by similarity.

        Args:
            responses: List of responses
            threshold: Similarity threshold for grouping

        Returns:
            List of response groups
        """
        if not responses:
            return []

        groups: list[list[ProviderResponse]] = []

        for response in responses:
            added = False
            for group in groups:
                # Check similarity with first response in group
                if self._calculate_similarity(response.response, group[0].response) >= threshold:
                    group.append(response)
                    added = True
                    break

            if not added:
                groups.append([response])

        return groups


class FirstSuccessAggregator(BaseAggregator):
    """Return the first successful response.

    Simple aggregator that returns the first response received.
    Useful when speed is more important than consensus.
    """

    def aggregate(self, responses: list[ProviderResponse]) -> AggregationResult:
        """Return first response.

        Args:
            responses: List of provider responses

        Returns:
            First response
        """
        if not responses:
            return AggregationResult(
                response=None,
                confidence=0.0,
                metadata={"error": "No responses"},
            )

        first = responses[0]
        return AggregationResult(
            response=first.response,
            confidence=first.quality_score,
            metadata={
                "provider": first.provider_name,
                "strategy": "first_success",
            },
        )


class MajorityVoteAggregator(BaseAggregator):
    """Majority voting aggregator.

    Groups similar responses and returns the response from
    the largest group. Works well for categorical outputs.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize aggregator.

        Args:
            similarity_threshold: Threshold for grouping similar responses
        """
        self.similarity_threshold = similarity_threshold

    def aggregate(self, responses: list[ProviderResponse]) -> AggregationResult:
        """Aggregate using majority vote.

        Args:
            responses: List of provider responses

        Returns:
            Most common response
        """
        if not responses:
            return AggregationResult(
                response=None,
                confidence=0.0,
                metadata={"error": "No responses"},
            )

        if len(responses) == 1:
            return AggregationResult(
                response=responses[0].response,
                confidence=responses[0].quality_score,
                metadata={
                    "provider": responses[0].provider_name,
                    "strategy": "majority_vote",
                    "vote_count": 1,
                },
            )

        # Group similar responses
        groups = self._group_similar_responses(responses, self.similarity_threshold)

        # Find largest group
        largest_group = max(groups, key=len)
        vote_count = len(largest_group)
        total_votes = len(responses)

        # Calculate confidence based on vote ratio
        confidence = vote_count / total_votes

        # Use response from highest quality provider in winning group
        best_response = max(largest_group, key=lambda r: r.quality_score)

        return AggregationResult(
            response=best_response.response,
            confidence=confidence,
            metadata={
                "provider": best_response.provider_name,
                "strategy": "majority_vote",
                "vote_count": vote_count,
                "total_votes": total_votes,
                "group_count": len(groups),
            },
        )


class WeightedVoteAggregator(BaseAggregator):
    """Weighted voting aggregator.

    Each provider's vote is weighted by their configured weight
    and quality score. Useful when some providers are more
    reliable than others.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_quality_score: bool = True,
    ):
        """Initialize aggregator.

        Args:
            similarity_threshold: Threshold for grouping similar responses
            use_quality_score: Whether to factor in quality score
        """
        self.similarity_threshold = similarity_threshold
        self.use_quality_score = use_quality_score

    def aggregate(self, responses: list[ProviderResponse]) -> AggregationResult:
        """Aggregate using weighted voting.

        Args:
            responses: List of provider responses

        Returns:
            Response with highest weighted votes
        """
        if not responses:
            return AggregationResult(
                response=None,
                confidence=0.0,
                metadata={"error": "No responses"},
            )

        if len(responses) == 1:
            r = responses[0]
            weight = r.weight * (r.quality_score if self.use_quality_score else 1.0)
            return AggregationResult(
                response=r.response,
                confidence=r.quality_score,
                metadata={
                    "provider": r.provider_name,
                    "strategy": "weighted_vote",
                    "weight": weight,
                },
            )

        # Group similar responses
        groups = self._group_similar_responses(responses, self.similarity_threshold)

        # Calculate weighted score for each group
        group_weights: list[tuple[list[ProviderResponse], float]] = []
        for group in groups:
            total_weight = sum(
                r.weight * (r.quality_score if self.use_quality_score else 1.0) for r in group
            )
            group_weights.append((group, total_weight))

        # Find highest weighted group
        winning_group, winning_weight = max(group_weights, key=lambda x: x[1])

        # Calculate total weight for confidence
        total_weight = sum(w for _, w in group_weights)
        confidence = winning_weight / total_weight if total_weight > 0 else 0.0

        # Use response from highest quality provider in winning group
        best_response = max(winning_group, key=lambda r: r.quality_score)

        return AggregationResult(
            response=best_response.response,
            confidence=confidence,
            metadata={
                "provider": best_response.provider_name,
                "strategy": "weighted_vote",
                "winning_weight": winning_weight,
                "total_weight": total_weight,
                "group_size": len(winning_group),
            },
        )


class QualityScoreAggregator(BaseAggregator):
    """Quality score-based aggregator.

    Returns the response from the provider with highest quality score.
    Optionally considers latency as a tie-breaker.
    """

    def __init__(self, consider_latency: bool = False, latency_penalty: float = 0.01):
        """Initialize aggregator.

        Args:
            consider_latency: Whether to penalize slow responses
            latency_penalty: Penalty per second of latency
        """
        self.consider_latency = consider_latency
        self.latency_penalty = latency_penalty

    def aggregate(self, responses: list[ProviderResponse]) -> AggregationResult:
        """Aggregate by selecting highest quality response.

        Args:
            responses: List of provider responses

        Returns:
            Response from highest quality provider
        """
        if not responses:
            return AggregationResult(
                response=None,
                confidence=0.0,
                metadata={"error": "No responses"},
            )

        def score_response(r: ProviderResponse) -> float:
            score = r.quality_score
            if self.consider_latency:
                score -= r.latency * self.latency_penalty
            return score

        best = max(responses, key=score_response)

        return AggregationResult(
            response=best.response,
            confidence=best.quality_score,
            metadata={
                "provider": best.provider_name,
                "strategy": "quality_score",
                "score": score_response(best),
                "latency": best.latency,
            },
        )


class ConsensusAggregator(BaseAggregator):
    """Consensus-based aggregator.

    Requires a minimum level of agreement between providers.
    Returns None if consensus cannot be reached.
    """

    def __init__(
        self,
        min_agreement: float = 0.6,
        similarity_threshold: float = 0.85,
    ):
        """Initialize aggregator.

        Args:
            min_agreement: Minimum fraction of providers that must agree
            similarity_threshold: Threshold for considering responses "same"
        """
        self.min_agreement = min_agreement
        self.similarity_threshold = similarity_threshold

    def aggregate(self, responses: list[ProviderResponse]) -> AggregationResult:
        """Aggregate with consensus requirement.

        Args:
            responses: List of provider responses

        Returns:
            Consensus response or None if not reached
        """
        if not responses:
            return AggregationResult(
                response=None,
                confidence=0.0,
                metadata={"error": "No responses"},
            )

        if len(responses) == 1:
            return AggregationResult(
                response=responses[0].response,
                confidence=responses[0].quality_score,
                metadata={
                    "provider": responses[0].provider_name,
                    "strategy": "consensus",
                    "consensus_reached": True,
                    "agreement": 1.0,
                },
            )

        # Group similar responses
        groups = self._group_similar_responses(responses, self.similarity_threshold)

        # Find largest group
        largest_group = max(groups, key=len)
        agreement = len(largest_group) / len(responses)

        if agreement < self.min_agreement:
            return AggregationResult(
                response=None,
                confidence=0.0,
                metadata={
                    "strategy": "consensus",
                    "consensus_reached": False,
                    "agreement": agreement,
                    "required": self.min_agreement,
                    "group_count": len(groups),
                },
            )

        # Use response from highest quality provider in consensus group
        best_response = max(largest_group, key=lambda r: r.quality_score)

        return AggregationResult(
            response=best_response.response,
            confidence=agreement,
            metadata={
                "provider": best_response.provider_name,
                "strategy": "consensus",
                "consensus_reached": True,
                "agreement": agreement,
                "group_size": len(largest_group),
            },
        )


class LLMJudgeAggregator(BaseAggregator):
    """LLM-as-Judge aggregator.

    Uses another LLM to evaluate and select the best response.
    Requires an LLM provider to be set.

    Note: This is a synchronous interface. For async usage,
    use the LLMJudgeAggregatorAsync class.
    """

    def __init__(self, judge_provider: Any = None):
        """Initialize aggregator.

        Args:
            judge_provider: LLM provider to use as judge
        """
        self.judge_provider = judge_provider

    def aggregate(self, responses: list[ProviderResponse]) -> AggregationResult:
        """Aggregate using LLM judge.

        Note: This is a simplified sync implementation.
        For production, use async version with actual LLM call.

        Args:
            responses: List of provider responses

        Returns:
            Response selected by judge
        """
        if not responses:
            return AggregationResult(
                response=None,
                confidence=0.0,
                metadata={"error": "No responses"},
            )

        if len(responses) == 1:
            return AggregationResult(
                response=responses[0].response,
                confidence=responses[0].quality_score,
                metadata={
                    "provider": responses[0].provider_name,
                    "strategy": "llm_judge",
                },
            )

        # Without actual LLM, fall back to weighted vote
        logger.warning("LLMJudgeAggregator: No judge provider set, falling back to weighted vote")

        weighted = WeightedVoteAggregator()
        result = weighted.aggregate(responses)
        result.metadata = result.metadata or {}
        result.metadata["strategy"] = "llm_judge_fallback"
        return result

    def set_judge_provider(self, provider: Any) -> None:
        """Set the judge provider.

        Args:
            provider: LLM provider to use as judge
        """
        self.judge_provider = provider


def get_aggregator(strategy: str, **kwargs: Any) -> BaseAggregator:
    """Factory function to create aggregator by name.

    Args:
        strategy: Aggregation strategy name
        **kwargs: Strategy-specific parameters

    Returns:
        Aggregator instance
    """
    aggregators = {
        "first_success": FirstSuccessAggregator,
        "majority_vote": MajorityVoteAggregator,
        "weighted_vote": WeightedVoteAggregator,
        "quality_score": QualityScoreAggregator,
        "consensus": ConsensusAggregator,
        "llm_judge": LLMJudgeAggregator,
    }

    aggregator_class = aggregators.get(strategy.lower())
    if not aggregator_class:
        logger.warning(f"Unknown aggregator '{strategy}', using first_success")
        return FirstSuccessAggregator()

    result: BaseAggregator = aggregator_class(**kwargs)
    return result
