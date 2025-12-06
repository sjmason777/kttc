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

"""Multi-provider agent orchestrator for ensemble translation QA.

Coordinates parallel evaluation across multiple LLM providers, aggregates
errors from all providers, and produces a unified quality report with
cross-provider validation.

Key features:
- Parallel calls to multiple LLM providers (Yandex, GigaChat, OpenAI, etc.)
- Error aggregation with cross-provider validation
- Consensus-based error confirmation (errors found by 2+ providers)
- Detailed per-provider metrics in the report

Example:
    >>> orchestrator = MultiProviderAgentOrchestrator(
    ...     providers={"yandex": yandex_provider, "anthropic": anthropic_provider},
    ...     aggregation_strategy="weighted_vote"
    ... )
    >>> report = await orchestrator.evaluate(task)
    >>> print(f"Ensemble score: {report.mqm_score}")
    >>> print(f"Providers used: {report.ensemble_metadata['providers']}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kttc.core import ErrorAnnotation, MQMScorer, QAReport, TranslationTask
from kttc.llm.base import BaseLLMProvider

from .orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class ProviderEvaluationResult:
    """Result from a single provider's evaluation.

    Attributes:
        provider_name: Name of the provider
        success: Whether evaluation succeeded
        errors: List of errors found
        mqm_score: MQM score calculated
        latency: Time taken in seconds
        error_message: Error message if failed
    """

    provider_name: str
    success: bool
    errors: list[ErrorAnnotation] = field(default_factory=list)
    mqm_score: float = 0.0
    latency: float = 0.0
    error_message: str | None = None


@dataclass
class EnsembleEvaluationResult:
    """Result from ensemble evaluation across multiple providers.

    Attributes:
        aggregated_errors: Errors confirmed by multiple providers
        all_errors: All errors from all providers (with source info)
        provider_results: Results from each provider
        aggregation_strategy: Strategy used for aggregation
        consensus_threshold: Minimum providers that must agree
    """

    aggregated_errors: list[ErrorAnnotation]
    all_errors: list[ErrorAnnotation]
    provider_results: list[ProviderEvaluationResult]
    aggregation_strategy: str
    consensus_threshold: int


class ErrorAggregator:
    """Aggregates errors from multiple providers.

    Implements cross-provider validation to reduce false positives.
    Errors found by multiple providers are considered more reliable.
    """

    def __init__(
        self,
        consensus_threshold: int = 2,
        similarity_threshold: float = 0.8,
    ):
        """Initialize error aggregator.

        Args:
            consensus_threshold: Minimum providers that must find an error
            similarity_threshold: Text similarity for matching errors (0.0-1.0)
        """
        self.consensus_threshold = consensus_threshold
        self.similarity_threshold = similarity_threshold

    def aggregate(
        self,
        provider_results: list[ProviderEvaluationResult],
    ) -> tuple[list[ErrorAnnotation], list[ErrorAnnotation]]:
        """Aggregate errors from multiple providers.

        Args:
            provider_results: Results from each provider

        Returns:
            Tuple of (confirmed_errors, all_errors)
            - confirmed_errors: Errors found by >= consensus_threshold providers
            - all_errors: All errors with provider source metadata
        """
        if not provider_results:
            return [], []

        # If only one provider, return all its errors as confirmed
        successful_results = [r for r in provider_results if r.success]
        if len(successful_results) <= 1:
            if successful_results:
                return successful_results[0].errors, successful_results[0].errors
            return [], []

        # Collect all errors with source tracking
        all_errors_with_source: list[tuple[ErrorAnnotation, str]] = []
        for result in successful_results:
            for error in result.errors:
                all_errors_with_source.append((error, result.provider_name))

        # Group similar errors
        error_groups = self._group_similar_errors(all_errors_with_source)

        # Build confirmed and all error lists
        confirmed_errors: list[ErrorAnnotation] = []
        all_errors: list[ErrorAnnotation] = []

        for group in error_groups:
            # Get unique providers that found this error
            providers = set(source for _, source in group)
            representative_error, _ = group[0]

            # Add provider info to error metadata
            error_with_meta = self._add_provider_metadata(representative_error, list(providers))
            all_errors.append(error_with_meta)

            # Check consensus
            if len(providers) >= self.consensus_threshold:
                confirmed_errors.append(error_with_meta)

        return confirmed_errors, all_errors

    def _group_similar_errors(
        self,
        errors_with_source: list[tuple[ErrorAnnotation, str]],
    ) -> list[list[tuple[ErrorAnnotation, str]]]:
        """Group similar errors together.

        Args:
            errors_with_source: List of (error, provider_name) tuples

        Returns:
            List of groups, each group contains similar errors
        """
        if not errors_with_source:
            return []

        groups: list[list[tuple[ErrorAnnotation, str]]] = []

        for error, source in errors_with_source:
            matched = False

            for group in groups:
                representative, _ = group[0]
                if self._errors_are_similar(error, representative):
                    group.append((error, source))
                    matched = True
                    break

            if not matched:
                groups.append([(error, source)])

        return groups

    def _errors_are_similar(
        self,
        error1: ErrorAnnotation,
        error2: ErrorAnnotation,
    ) -> bool:
        """Check if two errors are similar enough to be considered the same.

        Args:
            error1: First error
            error2: Second error

        Returns:
            True if errors are similar
        """
        # Must be same category and severity
        if error1.category != error2.category:
            return False
        if error1.severity != error2.severity:
            return False

        # Check location overlap
        loc1 = error1.location
        loc2 = error2.location
        if loc1 and loc2:
            # Check if locations overlap significantly
            start_diff = abs(loc1[0] - loc2[0])
            end_diff = abs(loc1[1] - loc2[1])
            if start_diff <= 10 and end_diff <= 10:
                return True

        # Compare by description
        desc1 = error1.description.lower() if error1.description else ""
        desc2 = error2.description.lower() if error2.description else ""
        return self._text_similarity(desc1, desc2) >= self.similarity_threshold

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio (0.0-1.0)
        """
        from difflib import SequenceMatcher

        return SequenceMatcher(None, text1, text2).ratio()

    def _add_provider_metadata(
        self,
        error: ErrorAnnotation,
        providers: list[str],
    ) -> ErrorAnnotation:
        """Add provider metadata to error description.

        Args:
            error: Error annotation
            providers: List of providers that found this error

        Returns:
            Error with provider info in description
        """
        # Since ErrorAnnotation doesn't have a metadata field,
        # we append provider info to description
        provider_info = f" [Found by: {', '.join(providers)}]"
        cross_validated = len(providers) >= self.consensus_threshold

        # Create new error with updated description
        enhanced_description = error.description
        if cross_validated:
            enhanced_description = f"✓ {error.description}{provider_info}"
        else:
            enhanced_description = f"{error.description}{provider_info}"

        return ErrorAnnotation(
            category=error.category,
            subcategory=error.subcategory,
            severity=error.severity,
            location=error.location,
            description=enhanced_description,
            suggestion=error.suggestion,
            confidence=error.confidence,
        )


class MultiProviderAgentOrchestrator:
    """Orchestrates QA evaluation across multiple LLM providers.

    Runs evaluation in parallel on multiple providers, aggregates results,
    and produces a unified report with cross-provider validation.

    Default provider priority:
    1. Yandex GPT
    2. GigaChat
    3. OpenAI
    4. Anthropic
    5. Gemini

    Example:
        >>> providers = {
        ...     "yandex": YandexProvider(api_key="..."),
        ...     "anthropic": AnthropicProvider(api_key="..."),
        ... }
        >>> orchestrator = MultiProviderAgentOrchestrator(
        ...     providers=providers,
        ...     quality_threshold=95.0,
        ...     consensus_threshold=2,
        ... )
        >>> report = await orchestrator.evaluate(task)
    """

    # Provider quality scores for weighted aggregation
    PROVIDER_QUALITY_SCORES: dict[str, float] = {
        "yandex": 0.85,
        "gigachat": 0.83,
        "openai": 0.90,
        "anthropic": 0.92,
        "gemini": 0.85,
    }

    def __init__(
        self,
        providers: dict[str, BaseLLMProvider],
        quality_threshold: float = 95.0,
        consensus_threshold: int = 2,
        aggregation_strategy: str = "weighted_vote",
        agent_temperature: float = 0.1,
        agent_max_tokens: int = 2000,
        quick_mode: bool = False,
        selected_agents: list[str] | None = None,
    ):
        """Initialize multi-provider orchestrator.

        Args:
            providers: Dict of provider_name -> LLM provider instance
            quality_threshold: Minimum MQM score to pass
            consensus_threshold: Min providers that must find an error to confirm it
            aggregation_strategy: How to aggregate results (weighted_vote, consensus, first_success)
            agent_temperature: Temperature for all agents
            agent_max_tokens: Max tokens for agent responses
            quick_mode: Enable quick mode (3 core agents only)
            selected_agents: Specific agents to use (None = defaults)
        """
        self.providers = providers
        self.quality_threshold = quality_threshold
        self.consensus_threshold = min(consensus_threshold, len(providers))
        self.aggregation_strategy = aggregation_strategy
        self.agent_temperature = agent_temperature
        self.agent_max_tokens = agent_max_tokens
        self.quick_mode = quick_mode
        self.selected_agents = selected_agents

        # Create orchestrator for each provider
        self._orchestrators: dict[str, AgentOrchestrator] = {}
        for name, provider in providers.items():
            self._orchestrators[name] = AgentOrchestrator(
                llm_provider=provider,
                quality_threshold=quality_threshold,
                agent_temperature=agent_temperature,
                agent_max_tokens=agent_max_tokens,
                quick_mode=quick_mode,
                selected_agents=selected_agents,
            )

        # Error aggregator
        self._aggregator = ErrorAggregator(
            consensus_threshold=self.consensus_threshold,
        )

        # MQM scorer for final score
        self._scorer = MQMScorer()

    async def evaluate(self, task: TranslationTask) -> QAReport:
        """Evaluate translation using all providers in parallel.

        Args:
            task: Translation task to evaluate

        Returns:
            QAReport with aggregated results and ensemble metadata
        """
        start_time = time.monotonic()

        # Run all providers in parallel
        logger.info(
            f"Ensemble evaluation with {len(self.providers)} providers: {list(self.providers.keys())}"
        )

        tasks = [self._evaluate_with_provider(name, task) for name in self.providers.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        provider_results: list[ProviderEvaluationResult] = []
        for result in results:
            if isinstance(result, ProviderEvaluationResult):
                provider_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Provider evaluation failed: {result}")

        total_latency = time.monotonic() - start_time

        # Aggregate errors
        confirmed_errors, all_errors = self._aggregator.aggregate(provider_results)

        # Calculate final MQM score from confirmed errors
        word_count = len(task.source_text.split())
        mqm_score = self._scorer.calculate_score(confirmed_errors, word_count)

        # Calculate aggregated score using provider weights
        if self.aggregation_strategy == "weighted_vote":
            mqm_score = self._calculate_weighted_score(provider_results)
        elif self.aggregation_strategy == "consensus":
            # Use score from confirmed errors only
            pass  # mqm_score already calculated from confirmed_errors

        # Determine status
        status = "pass" if mqm_score >= self.quality_threshold else "fail"

        # Build ensemble metadata
        ensemble_metadata = self._build_ensemble_metadata(
            provider_results, confirmed_errors, all_errors, total_latency
        )

        # Calculate confidence based on provider agreement
        confidence = self._calculate_confidence(provider_results, confirmed_errors, all_errors)

        return QAReport(
            task=task,
            mqm_score=mqm_score,
            errors=confirmed_errors,
            status=status,
            comet_score=None,
            confidence=confidence,
            agent_agreement=self._calculate_agreement(provider_results),
            agent_scores=self._get_provider_scores(provider_results),
            consensus_metadata=None,
            agent_details=None,
            ensemble_metadata=ensemble_metadata,
        )

    async def _evaluate_with_provider(
        self,
        provider_name: str,
        task: TranslationTask,
    ) -> ProviderEvaluationResult:
        """Evaluate translation with a single provider.

        Args:
            provider_name: Name of the provider
            task: Translation task

        Returns:
            ProviderEvaluationResult
        """
        start_time = time.monotonic()

        try:
            orchestrator = self._orchestrators[provider_name]
            report = await orchestrator.evaluate(task)

            latency = time.monotonic() - start_time

            return ProviderEvaluationResult(
                provider_name=provider_name,
                success=True,
                errors=report.errors,
                mqm_score=report.mqm_score,
                latency=latency,
            )

        except Exception as e:
            latency = time.monotonic() - start_time
            logger.error(f"Provider '{provider_name}' failed: {e}")

            return ProviderEvaluationResult(
                provider_name=provider_name,
                success=False,
                errors=[],
                mqm_score=0.0,
                latency=latency,
                error_message=str(e),
            )

    def _calculate_weighted_score(
        self,
        results: list[ProviderEvaluationResult],
    ) -> float:
        """Calculate weighted average MQM score.

        Args:
            results: Results from each provider

        Returns:
            Weighted average MQM score
        """
        successful = [r for r in results if r.success]
        if not successful:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for result in successful:
            weight = self.PROVIDER_QUALITY_SCORES.get(result.provider_name, 0.5)
            weighted_sum += result.mqm_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _calculate_confidence(
        self,
        results: list[ProviderEvaluationResult],
        confirmed_errors: list[ErrorAnnotation],
        all_errors: list[ErrorAnnotation],
    ) -> float:
        """Calculate confidence level based on provider agreement.

        Args:
            results: Results from each provider
            confirmed_errors: Cross-validated errors
            all_errors: All errors found

        Returns:
            Confidence score (0.0-1.0)
        """
        successful = [r for r in results if r.success]
        if len(successful) < 2:
            return 0.5  # Low confidence with single provider

        # Score agreement
        scores = [r.mqm_score for r in successful]
        score_variance = max(scores) - min(scores) if scores else 0
        score_agreement = max(0, 1 - score_variance / 20)  # 20 points = 0 agreement

        # Error agreement
        if all_errors:
            error_agreement = len(confirmed_errors) / len(all_errors)
        else:
            error_agreement = 1.0  # No errors = full agreement

        # Combined confidence
        return (score_agreement + error_agreement) / 2

    def _calculate_agreement(
        self,
        results: list[ProviderEvaluationResult],
    ) -> float:
        """Calculate agreement level between providers.

        Args:
            results: Results from each provider

        Returns:
            Agreement score (0.0-1.0)
        """
        successful = [r for r in results if r.success]
        if len(successful) < 2:
            return 1.0  # Single provider = perfect agreement with itself

        scores = [r.mqm_score for r in successful]
        avg_score = sum(scores) / len(scores)

        # Calculate variance from average
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)

        # Convert to agreement (lower variance = higher agreement)
        # Max expected variance of 100 (scores range 0-100)
        agreement = max(0, 1 - variance / 100)

        return agreement

    def _get_provider_scores(
        self,
        results: list[ProviderEvaluationResult],
    ) -> dict[str, float]:
        """Get MQM scores from each provider.

        Args:
            results: Results from each provider

        Returns:
            Dict of provider_name -> mqm_score
        """
        return {r.provider_name: r.mqm_score for r in results if r.success}

    def _build_ensemble_metadata(
        self,
        results: list[ProviderEvaluationResult],
        confirmed_errors: list[ErrorAnnotation],
        all_errors: list[ErrorAnnotation],
        total_latency: float,
    ) -> dict[str, Any]:
        """Build detailed ensemble metadata for the report.

        Args:
            results: Results from each provider
            confirmed_errors: Cross-validated errors
            all_errors: All errors found
            total_latency: Total evaluation time

        Returns:
            Ensemble metadata dictionary
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        provider_details = []
        for r in results:
            detail = {
                "name": r.provider_name,
                "success": r.success,
                "mqm_score": round(r.mqm_score, 2) if r.success else None,
                "errors_found": len(r.errors) if r.success else 0,
                "latency": round(r.latency, 2),
                "quality_weight": self.PROVIDER_QUALITY_SCORES.get(r.provider_name, 0.5),
            }
            if r.error_message:
                detail["error"] = r.error_message
            provider_details.append(detail)

        return {
            "ensemble_mode": True,
            "providers_total": len(results),
            "providers_successful": len(successful),
            "providers_failed": len(failed),
            "provider_details": provider_details,
            "aggregation_strategy": self.aggregation_strategy,
            "consensus_threshold": self.consensus_threshold,
            "total_errors_found": len(all_errors),
            "confirmed_errors": len(confirmed_errors),
            "rejected_errors": len(all_errors) - len(confirmed_errors),
            "cross_validation_rate": (
                len(confirmed_errors) / len(all_errors) if all_errors else 1.0
            ),
            "total_latency": round(total_latency, 2),
            "avg_provider_latency": (
                round(sum(r.latency for r in successful) / len(successful), 2) if successful else 0
            ),
        }
