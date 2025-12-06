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

"""Tests for MultiProviderAgentOrchestrator and ErrorAggregator.

These tests verify the ensemble mode functionality including:
- Error aggregation and cross-provider validation
- Weighted score calculation
- Consensus threshold logic
- Provider failure handling
"""

import pytest

from kttc.agents.multi_provider_orchestrator import (
    ErrorAggregator,
    MultiProviderAgentOrchestrator,
    ProviderEvaluationResult,
)
from kttc.core import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask

# =============================================================================
# ErrorAggregator Tests
# =============================================================================


class TestErrorAggregator:
    """Tests for ErrorAggregator class."""

    def test_empty_results_returns_empty_lists(self):
        """Empty provider results should return empty error lists."""
        aggregator = ErrorAggregator(consensus_threshold=2)
        confirmed, all_errors = aggregator.aggregate([])

        assert confirmed == []
        assert all_errors == []

    def test_single_provider_returns_all_errors_as_confirmed(self):
        """With single provider, all errors should be confirmed."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test error",
        )

        result = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )

        confirmed, all_errors = aggregator.aggregate([result])

        assert len(confirmed) == 1
        assert len(all_errors) == 1

    def test_consensus_threshold_filters_errors(self):
        """Errors found by fewer providers than threshold should be filtered."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        # Error found only by one provider
        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Error only in OpenAI",
        )

        # Error found by both providers (similar location and category)
        error2_openai = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(20, 30),
            description="Grammar error in translation",
        )
        error2_anthropic = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(22, 32),  # Similar location (within 10 chars)
            description="Grammar error detected",
        )

        result_openai = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error1, error2_openai],
            mqm_score=85.0,
        )
        result_anthropic = ProviderEvaluationResult(
            provider_name="anthropic",
            success=True,
            errors=[error2_anthropic],
            mqm_score=90.0,
        )

        confirmed, all_errors = aggregator.aggregate([result_openai, result_anthropic])

        # Only error2 should be confirmed (found by both)
        assert len(confirmed) == 1
        assert "fluency" in confirmed[0].category

        # All unique errors should be in all_errors
        assert len(all_errors) == 2

    def test_failed_provider_results_are_ignored(self):
        """Failed provider results should be excluded from aggregation."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test error",
        )

        success_result = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )
        failed_result = ProviderEvaluationResult(
            provider_name="anthropic",
            success=False,
            errors=[],
            error_message="API timeout",
        )

        confirmed, all_errors = aggregator.aggregate([success_result, failed_result])

        # Single successful provider - all its errors confirmed
        assert len(confirmed) == 1
        assert len(all_errors) == 1

    def test_errors_are_similar_same_category_severity_location(self):
        """Errors with same category, severity, and close location are similar."""
        aggregator = ErrorAggregator(consensus_threshold=2, similarity_threshold=0.8)

        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(100, 120),
            description="Word mistranslated",
        )
        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(102, 118),  # Within 10 chars
            description="Incorrect translation",
        )

        assert aggregator._errors_are_similar(error1, error2) is True

    def test_errors_are_not_similar_different_category(self):
        """Errors with different categories are not similar."""
        aggregator = ErrorAggregator()

        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test",
        )
        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test",
        )

        assert aggregator._errors_are_similar(error1, error2) is False

    def test_errors_are_not_similar_different_severity(self):
        """Errors with different severities are not similar."""
        aggregator = ErrorAggregator()

        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test",
        )
        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MINOR,
            location=(0, 10),
            description="Test",
        )

        assert aggregator._errors_are_similar(error1, error2) is False

    def test_errors_are_similar_by_description_when_locations_far(self):
        """Errors with far locations but similar descriptions are similar."""
        aggregator = ErrorAggregator(similarity_threshold=0.7)

        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="The word 'hello' is mistranslated",
        )
        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(500, 510),  # Far location
            description="The word 'hello' was mistranslated",
        )

        assert aggregator._errors_are_similar(error1, error2) is True

    def test_provider_metadata_added_to_errors(self):
        """Provider information should be added to error descriptions."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test error",
        )

        result1 = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )
        result2 = ProviderEvaluationResult(
            provider_name="anthropic",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )

        confirmed, _ = aggregator.aggregate([result1, result2])

        # Check that provider info is in description
        assert "[Found by:" in confirmed[0].description
        assert "openai" in confirmed[0].description or "anthropic" in confirmed[0].description

    def test_cross_validated_error_has_checkmark(self):
        """Cross-validated errors should have ✓ prefix."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test error",
        )

        result1 = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )
        result2 = ProviderEvaluationResult(
            provider_name="anthropic",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )

        confirmed, _ = aggregator.aggregate([result1, result2])

        # Cross-validated errors start with ✓
        assert confirmed[0].description.startswith("✓")

    def test_text_similarity_calculation(self):
        """Text similarity should work correctly."""
        aggregator = ErrorAggregator()

        # Identical texts
        assert aggregator._text_similarity("hello world", "hello world") == 1.0

        # Similar texts
        similarity = aggregator._text_similarity("hello world", "hello worlds")
        assert similarity > 0.8

        # Different texts
        similarity = aggregator._text_similarity("hello", "goodbye")
        assert similarity < 0.5


# =============================================================================
# ProviderEvaluationResult Tests
# =============================================================================


class TestProviderEvaluationResult:
    """Tests for ProviderEvaluationResult dataclass."""

    def test_successful_result_creation(self):
        """Test creating a successful provider result."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test",
        )

        result = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error],
            mqm_score=85.5,
            latency=1.5,
        )

        assert result.provider_name == "openai"
        assert result.success is True
        assert len(result.errors) == 1
        assert result.mqm_score == 85.5
        assert result.latency == 1.5
        assert result.error_message is None

    def test_failed_result_creation(self):
        """Test creating a failed provider result."""
        result = ProviderEvaluationResult(
            provider_name="anthropic",
            success=False,
            errors=[],
            mqm_score=0.0,
            latency=5.0,
            error_message="API rate limit exceeded",
        )

        assert result.provider_name == "anthropic"
        assert result.success is False
        assert len(result.errors) == 0
        assert result.error_message == "API rate limit exceeded"


# =============================================================================
# MultiProviderAgentOrchestrator Tests
# =============================================================================


class TestMultiProviderAgentOrchestrator:
    """Tests for MultiProviderAgentOrchestrator class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        from unittest.mock import AsyncMock, MagicMock

        provider = MagicMock()
        provider.complete = AsyncMock(return_value="Mocked response")
        return provider

    @pytest.fixture
    def sample_task(self):
        """Create a sample translation task."""
        return TranslationTask(
            source_text="Hello world",
            translation="Привет мир",
            source_lang="en",
            target_lang="ru",
        )

    def test_orchestrator_initialization(self, mock_provider):
        """Test orchestrator initializes correctly."""
        providers = {
            "openai": mock_provider,
            "anthropic": mock_provider,
        }

        orchestrator = MultiProviderAgentOrchestrator(
            providers=providers,
            quality_threshold=95.0,
            consensus_threshold=2,
        )

        assert len(orchestrator.providers) == 2
        assert orchestrator.quality_threshold == 95.0
        assert orchestrator.consensus_threshold == 2
        assert len(orchestrator._orchestrators) == 2

    def test_consensus_threshold_capped_at_provider_count(self, mock_provider):
        """Consensus threshold should not exceed provider count."""
        providers = {"openai": mock_provider}

        orchestrator = MultiProviderAgentOrchestrator(
            providers=providers,
            consensus_threshold=5,  # Higher than provider count
        )

        # Should be capped at 1
        assert orchestrator.consensus_threshold == 1

    def test_provider_quality_scores_defined(self):
        """All default providers should have quality scores."""
        expected_providers = ["yandex", "gigachat", "openai", "anthropic", "gemini"]

        for provider in expected_providers:
            assert provider in MultiProviderAgentOrchestrator.PROVIDER_QUALITY_SCORES
            score = MultiProviderAgentOrchestrator.PROVIDER_QUALITY_SCORES[provider]
            assert 0.0 <= score <= 1.0

    def test_weighted_score_calculation(self, mock_provider):
        """Test weighted score calculation."""
        providers = {"openai": mock_provider, "anthropic": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(
                provider_name="openai",
                success=True,
                mqm_score=90.0,
            ),
            ProviderEvaluationResult(
                provider_name="anthropic",
                success=True,
                mqm_score=95.0,
            ),
        ]

        weighted_score = orchestrator._calculate_weighted_score(results)

        # openai weight: 0.90, anthropic weight: 0.92
        # Expected: (90 * 0.90 + 95 * 0.92) / (0.90 + 0.92)
        expected = (90 * 0.90 + 95 * 0.92) / (0.90 + 0.92)
        assert abs(weighted_score - expected) < 0.01

    def test_weighted_score_with_all_failures_returns_zero(self, mock_provider):
        """Weighted score should be 0 when all providers fail."""
        providers = {"openai": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(
                provider_name="openai",
                success=False,
                error_message="Failed",
            ),
        ]

        weighted_score = orchestrator._calculate_weighted_score(results)
        assert weighted_score == 0.0

    def test_confidence_calculation_high_agreement(self, mock_provider):
        """High agreement between providers should result in high confidence."""
        providers = {"openai": mock_provider, "anthropic": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(
                provider_name="openai",
                success=True,
                mqm_score=95.0,
            ),
            ProviderEvaluationResult(
                provider_name="anthropic",
                success=True,
                mqm_score=95.0,  # Same score = high agreement
            ),
        ]

        confidence = orchestrator._calculate_confidence(results, [], [])
        assert confidence >= 0.9

    def test_confidence_calculation_low_agreement(self, mock_provider):
        """Low agreement between providers should result in lower confidence."""
        providers = {"openai": mock_provider, "anthropic": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(
                provider_name="openai",
                success=True,
                mqm_score=60.0,
            ),
            ProviderEvaluationResult(
                provider_name="anthropic",
                success=True,
                mqm_score=95.0,  # 35 point difference
            ),
        ]

        confidence = orchestrator._calculate_confidence(results, [], [])
        assert confidence < 0.9

    def test_confidence_single_provider_is_moderate(self, mock_provider):
        """Single provider should give moderate confidence."""
        providers = {"openai": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(
                provider_name="openai",
                success=True,
                mqm_score=95.0,
            ),
        ]

        confidence = orchestrator._calculate_confidence(results, [], [])
        assert confidence == 0.5

    def test_agreement_calculation_perfect_agreement(self, mock_provider):
        """Perfect agreement when all scores are identical."""
        providers = {"openai": mock_provider, "anthropic": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(provider_name="openai", success=True, mqm_score=90.0),
            ProviderEvaluationResult(provider_name="anthropic", success=True, mqm_score=90.0),
        ]

        agreement = orchestrator._calculate_agreement(results)
        assert agreement == 1.0

    def test_agreement_calculation_no_agreement(self, mock_provider):
        """Low agreement when scores vary significantly."""
        providers = {"openai": mock_provider, "anthropic": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(provider_name="openai", success=True, mqm_score=50.0),
            ProviderEvaluationResult(provider_name="anthropic", success=True, mqm_score=100.0),
        ]

        agreement = orchestrator._calculate_agreement(results)
        # Variance = 625, agreement = max(0, 1 - 625/100) = 0
        assert agreement < 0.5

    def test_get_provider_scores(self, mock_provider):
        """Provider scores should be returned correctly."""
        providers = {"openai": mock_provider, "anthropic": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        results = [
            ProviderEvaluationResult(provider_name="openai", success=True, mqm_score=90.0),
            ProviderEvaluationResult(provider_name="anthropic", success=True, mqm_score=95.0),
            ProviderEvaluationResult(provider_name="gemini", success=False, mqm_score=0.0),
        ]

        scores = orchestrator._get_provider_scores(results)

        assert scores == {"openai": 90.0, "anthropic": 95.0}
        assert "gemini" not in scores  # Failed provider excluded

    def test_build_ensemble_metadata(self, mock_provider):
        """Ensemble metadata should contain all required fields."""
        providers = {"openai": mock_provider, "anthropic": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test",
        )

        results = [
            ProviderEvaluationResult(
                provider_name="openai",
                success=True,
                errors=[error],
                mqm_score=90.0,
                latency=1.0,
            ),
            ProviderEvaluationResult(
                provider_name="anthropic",
                success=False,
                errors=[],
                mqm_score=0.0,
                latency=5.0,
                error_message="Timeout",
            ),
        ]

        metadata = orchestrator._build_ensemble_metadata(
            results,
            confirmed_errors=[error],
            all_errors=[error],
            total_latency=6.0,
        )

        assert metadata["ensemble_mode"] is True
        assert metadata["providers_total"] == 2
        assert metadata["providers_successful"] == 1
        assert metadata["providers_failed"] == 1
        assert metadata["confirmed_errors"] == 1
        assert metadata["total_errors_found"] == 1
        assert metadata["rejected_errors"] == 0
        assert metadata["total_latency"] == 6.0
        assert len(metadata["provider_details"]) == 2

    def test_build_ensemble_metadata_cross_validation_rate(self, mock_provider):
        """Cross-validation rate should be calculated correctly."""
        providers = {"openai": mock_provider}
        orchestrator = MultiProviderAgentOrchestrator(providers=providers)

        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Confirmed",
        )
        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(20, 30),
            description="Rejected",
        )

        results = []
        metadata = orchestrator._build_ensemble_metadata(
            results,
            confirmed_errors=[error1],  # 1 confirmed
            all_errors=[error1, error2],  # 2 total
            total_latency=1.0,
        )

        # 1/2 = 0.5
        assert metadata["cross_validation_rate"] == 0.5
        assert metadata["rejected_errors"] == 1


# =============================================================================
# QAReport Ensemble Tests
# =============================================================================


class TestQAReportEnsemble:
    """Tests for QAReport ensemble functionality."""

    @pytest.fixture
    def sample_task(self):
        """Create a sample translation task."""
        return TranslationTask(
            source_text="Hello world",
            translation="Привет мир",
            source_lang="en",
            target_lang="ru",
        )

    def test_is_ensemble_false_without_metadata(self, sample_task):
        """is_ensemble should be False when no ensemble_metadata."""
        report = QAReport(
            task=sample_task,
            mqm_score=95.0,
            errors=[],
            status="pass",
        )

        assert report.is_ensemble is False

    def test_is_ensemble_false_with_empty_metadata(self, sample_task):
        """is_ensemble should be False when ensemble_metadata is empty."""
        report = QAReport(
            task=sample_task,
            mqm_score=95.0,
            errors=[],
            status="pass",
            ensemble_metadata={},
        )

        assert report.is_ensemble is False

    def test_is_ensemble_false_when_ensemble_mode_false(self, sample_task):
        """is_ensemble should be False when ensemble_mode is False."""
        report = QAReport(
            task=sample_task,
            mqm_score=95.0,
            errors=[],
            status="pass",
            ensemble_metadata={"ensemble_mode": False},
        )

        assert report.is_ensemble is False

    def test_is_ensemble_true_when_ensemble_mode_true(self, sample_task):
        """is_ensemble should be True when ensemble_mode is True."""
        report = QAReport(
            task=sample_task,
            mqm_score=95.0,
            errors=[],
            status="pass",
            ensemble_metadata={
                "ensemble_mode": True,
                "providers_total": 3,
            },
        )

        assert report.is_ensemble is True

    def test_ensemble_metadata_serialization(self, sample_task):
        """Ensemble metadata should serialize to JSON correctly."""
        ensemble_meta = {
            "ensemble_mode": True,
            "providers_total": 3,
            "providers_successful": 2,
            "provider_details": [
                {"name": "openai", "mqm_score": 90.0},
                {"name": "anthropic", "mqm_score": 95.0},
            ],
        }

        report = QAReport(
            task=sample_task,
            mqm_score=92.5,
            errors=[],
            status="pass",
            ensemble_metadata=ensemble_meta,
        )

        # Serialize and deserialize
        json_data = report.model_dump_json()
        restored = QAReport.model_validate_json(json_data)

        assert restored.is_ensemble is True
        assert restored.ensemble_metadata["providers_total"] == 3
        assert len(restored.ensemble_metadata["provider_details"]) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestEnsembleIntegration:
    """Integration tests for ensemble mode."""

    @pytest.fixture
    def sample_task(self):
        """Create a sample translation task."""
        return TranslationTask(
            source_text="Hello, world! This is a test.",
            translation="Привет мир! Это тест.",
            source_lang="en",
            target_lang="ru",
        )

    def test_aggregator_with_three_providers(self):
        """Test aggregation with three providers and varying agreement."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        # Error found by all 3 providers
        error_all = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Missing comma after greeting",
        )

        # Error found by 2 providers
        error_two = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(20, 30),
            description="Incorrect word form",
        )

        # Error found by only 1 provider
        error_one = ErrorAnnotation(
            category="terminology",
            subcategory="inconsistency",
            severity=ErrorSeverity.MINOR,
            location=(50, 60),
            description="Terminology inconsistency",
        )

        result_openai = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error_all, error_two, error_one],
            mqm_score=85.0,
        )
        result_anthropic = ProviderEvaluationResult(
            provider_name="anthropic",
            success=True,
            errors=[error_all, error_two],
            mqm_score=90.0,
        )
        result_gemini = ProviderEvaluationResult(
            provider_name="gemini",
            success=True,
            errors=[error_all],
            mqm_score=92.0,
        )

        confirmed, all_errors = aggregator.aggregate(
            [result_openai, result_anthropic, result_gemini]
        )

        # error_all: 3 providers (confirmed)
        # error_two: 2 providers (confirmed)
        # error_one: 1 provider (not confirmed)
        assert len(confirmed) == 2
        assert len(all_errors) == 3

    def test_aggregator_handles_duplicate_errors_correctly(self):
        """Duplicate errors from same provider should not inflate consensus count."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test error",
        )

        # Provider A has same error 3 times - should count as 1 for consensus
        result_a = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error, error, error],  # Duplicates
            mqm_score=85.0,
        )

        # Provider B has the same error once
        result_b = ProviderEvaluationResult(
            provider_name="anthropic",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )

        confirmed, all_errors = aggregator.aggregate([result_a, result_b])

        # Error should be confirmed (2 unique providers)
        # Not 4 (3+1) which would happen if duplicates inflated count
        assert len(confirmed) == 1
        # Should group to 1 unique error
        assert len(all_errors) == 1

    def test_mixed_success_and_failure(self):
        """Test with mix of successful and failed providers."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description="Test",
        )

        results = [
            ProviderEvaluationResult(
                provider_name="openai", success=True, errors=[error], mqm_score=90.0
            ),
            ProviderEvaluationResult(
                provider_name="anthropic",
                success=False,
                errors=[],
                error_message="Rate limit",
            ),
            ProviderEvaluationResult(
                provider_name="gemini", success=True, errors=[error], mqm_score=88.0
            ),
        ]

        confirmed, all_errors = aggregator.aggregate(results)

        # 2 successful providers found the error
        assert len(confirmed) == 1
        # Failed provider ignored
        assert "anthropic" not in confirmed[0].description


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_error_list_from_all_providers(self):
        """No errors from any provider should return empty lists."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        results = [
            ProviderEvaluationResult(
                provider_name="openai", success=True, errors=[], mqm_score=100.0
            ),
            ProviderEvaluationResult(
                provider_name="anthropic", success=True, errors=[], mqm_score=100.0
            ),
        ]

        confirmed, all_errors = aggregator.aggregate(results)

        assert confirmed == []
        assert all_errors == []

    def test_all_providers_failed(self):
        """All providers failing should return empty lists."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        results = [
            ProviderEvaluationResult(
                provider_name="openai",
                success=False,
                errors=[],
                error_message="Timeout",
            ),
            ProviderEvaluationResult(
                provider_name="anthropic",
                success=False,
                errors=[],
                error_message="Rate limit",
            ),
        ]

        confirmed, all_errors = aggregator.aggregate(results)

        assert confirmed == []
        assert all_errors == []

    def test_very_long_error_descriptions(self):
        """Long error descriptions should be handled correctly."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        long_desc = "A" * 10000  # Very long description

        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 10),
            description=long_desc,
        )

        # Use 2 providers to test with metadata appending
        result_a = ProviderEvaluationResult(
            provider_name="openai",
            success=True,
            errors=[error],
            mqm_score=85.0,
        )
        result_b = ProviderEvaluationResult(
            provider_name="anthropic",
            success=True,
            errors=[error],
            mqm_score=90.0,
        )

        confirmed, all_errors = aggregator.aggregate([result_a, result_b])

        assert len(confirmed) == 1
        # Description should include original long text plus provider metadata
        assert len(confirmed[0].description) > 10000
        assert long_desc in confirmed[0].description

    def test_zero_location_errors(self):
        """Errors at location (0, 0) should be handled."""
        aggregator = ErrorAggregator(consensus_threshold=2)

        error1 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 0),
            description="Error at start",
        )
        error2 = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 0),
            description="Error at start",
        )

        result1 = ProviderEvaluationResult(
            provider_name="openai", success=True, errors=[error1], mqm_score=90.0
        )
        result2 = ProviderEvaluationResult(
            provider_name="anthropic", success=True, errors=[error2], mqm_score=90.0
        )

        confirmed, _ = aggregator.aggregate([result1, result2])

        assert len(confirmed) == 1
