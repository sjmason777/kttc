"""Unit tests for WeightedConsensus."""

import pytest

from kttc.agents.consensus import WeightedConsensus
from kttc.core import ErrorAnnotation, ErrorSeverity


class TestWeightedConsensus:
    """Tests for WeightedConsensus system."""

    def test_calculate_weighted_score_basic(self) -> None:
        """Test basic weighted score calculation."""
        consensus = WeightedConsensus()

        # Create agent results with different error counts
        agent_results = {
            "accuracy": [
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity=ErrorSeverity.MAJOR,
                    location=(0, 5),
                    description="Wrong meaning",
                )
            ],
            "fluency": [],  # No errors
            "terminology": [
                ErrorAnnotation(
                    category="terminology",
                    subcategory="inconsistency",
                    severity=ErrorSeverity.MINOR,
                    location=(6, 10),
                    description="Term inconsistency",
                )
            ],
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=100)

        # Check that all expected fields are present
        assert "weighted_mqm_score" in result
        assert "confidence" in result
        assert "agent_agreement" in result
        assert "agent_scores" in result
        assert "agent_weights_used" in result

        # MQM score should be calculated
        assert 0.0 <= result["weighted_mqm_score"] <= 100.0

        # Confidence should be between 0 and 1
        assert 0.0 <= result["confidence"] <= 1.0

        # Agreement should be between 0 and 1
        assert 0.0 <= result["agent_agreement"] <= 1.0

        # Should have individual scores for each agent
        assert "accuracy" in result["agent_scores"]
        assert "fluency" in result["agent_scores"]
        assert "terminology" in result["agent_scores"]

    def test_calculate_weighted_score_perfect_translation(self) -> None:
        """Test weighted score for perfect translation (no errors)."""
        consensus = WeightedConsensus()

        agent_results = {
            "accuracy": [],
            "fluency": [],
            "terminology": [],
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=50)

        # Perfect translation should have perfect score
        assert result["weighted_mqm_score"] == 100.0

        # All agents should agree (all gave 100.0)
        assert result["agent_agreement"] >= 0.99  # Almost perfect agreement

        # Confidence should be high but not maximum (only 3 agents)
        assert result["confidence"] >= 0.7

    def test_calculate_weighted_score_high_variance(self) -> None:
        """Test that high variance between agents reduces confidence."""
        consensus = WeightedConsensus()

        # Create very different error patterns
        agent_results = {
            "accuracy": [
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity=ErrorSeverity.CRITICAL,
                    location=(0, 10),
                    description="Critical error",
                ),
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="omission",
                    severity=ErrorSeverity.MAJOR,
                    location=(11, 20),
                    description="Major error",
                ),
            ],
            "fluency": [],  # Perfect fluency
            "terminology": [],  # Perfect terminology
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=50)

        # High variance should reduce confidence
        # (accuracy found major issues, others found nothing)
        assert result["confidence"] < 0.8  # Lower confidence

        # Agent agreement should be lower
        assert result["agent_agreement"] < 0.9

    def test_agent_weights_affect_final_score(self) -> None:
        """Test that agent weights properly affect final score."""
        # Custom weights: accuracy has very high weight, fluency very low
        custom_weights = {
            "accuracy": 1.0,  # Maximum weight
            "fluency": 0.1,  # Very low weight
        }

        consensus = WeightedConsensus(agent_weights=custom_weights)

        agent_results = {
            "accuracy": [
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity=ErrorSeverity.MAJOR,
                    location=(0, 5),
                    description="Major accuracy error",
                )
            ],
            "fluency": [
                ErrorAnnotation(
                    category="fluency",
                    subcategory="grammar",
                    severity=ErrorSeverity.MINOR,
                    location=(6, 10),
                    description="Minor fluency error",
                )
            ],
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=100)

        # Accuracy has high weight, so accuracy score should dominate
        assert result["agent_weights_used"]["accuracy"] == 1.0
        assert result["agent_weights_used"]["fluency"] == 0.1

        # Total weight should be sum of weights
        assert result["total_weight"] == pytest.approx(1.1, rel=0.01)

    def test_calculate_confidence_only(self) -> None:
        """Test confidence-only calculation."""
        consensus = WeightedConsensus()

        # High agreement scenario
        agent_scores_high_agreement = {
            "accuracy": 95.0,
            "fluency": 94.5,
            "terminology": 95.2,
        }

        confidence_high = consensus.calculate_confidence_only(agent_scores_high_agreement)

        # Should have high confidence
        assert confidence_high >= 0.8

        # Low agreement scenario
        agent_scores_low_agreement = {
            "accuracy": 50.0,
            "fluency": 95.0,
            "terminology": 70.0,
        }

        confidence_low = consensus.calculate_confidence_only(agent_scores_low_agreement)

        # Should have lower confidence
        assert confidence_low < confidence_high

    def test_single_agent_confidence(self) -> None:
        """Test confidence calculation with single agent."""
        consensus = WeightedConsensus()

        agent_results = {
            "accuracy": [
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity=ErrorSeverity.MINOR,
                    location=(0, 5),
                    description="Minor error",
                )
            ]
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=100)

        # Single agent should have moderate confidence (0.7)
        assert result["confidence"] == pytest.approx(0.7, rel=0.01)

    def test_metadata_fields(self) -> None:
        """Test that metadata contains expected statistics."""
        consensus = WeightedConsensus()

        agent_results = {
            "accuracy": [
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity=ErrorSeverity.MAJOR,
                    location=(0, 5),
                    description="Error 1",
                )
            ],
            "fluency": [],
            "terminology": [
                ErrorAnnotation(
                    category="terminology",
                    subcategory="inconsistency",
                    severity=ErrorSeverity.MINOR,
                    location=(6, 10),
                    description="Error 2",
                )
            ],
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=100)

        metadata = result["metadata"]

        # Check expected metadata fields
        assert "num_agents" in metadata
        assert metadata["num_agents"] == 3

        assert "total_errors" in metadata
        assert metadata["total_errors"] == 2

        assert "score_variance" in metadata
        assert "score_std_dev" in metadata
        assert "min_agent_score" in metadata
        assert "max_agent_score" in metadata

    def test_empty_agent_results_raises_error(self) -> None:
        """Test that empty agent results raises ValueError."""
        consensus = WeightedConsensus()

        with pytest.raises(ValueError, match="agent_results cannot be empty"):
            consensus.calculate_weighted_score({}, word_count=100)

    def test_zero_word_count_raises_error(self) -> None:
        """Test that zero word count raises ValueError."""
        consensus = WeightedConsensus()

        agent_results = {
            "accuracy": [],
        }

        with pytest.raises(ValueError, match="word_count must be greater than 0"):
            consensus.calculate_weighted_score(agent_results, word_count=0)

    def test_default_weights_used_for_unknown_agents(self) -> None:
        """Test that unknown agents get default weight of 1.0."""
        consensus = WeightedConsensus()

        agent_results = {
            "unknown_agent": [
                ErrorAnnotation(
                    category="unknown_agent",
                    subcategory="test",
                    severity=ErrorSeverity.MINOR,
                    location=(0, 5),
                    description="Test error",
                )
            ]
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=100)

        # Unknown agent should get default weight of 1.0
        assert result["agent_weights_used"]["unknown_agent"] == 1.0

    def test_all_agents_perfect_high_confidence(self) -> None:
        """Test that all perfect agents lead to high confidence."""
        consensus = WeightedConsensus()

        agent_results = {
            "accuracy": [],
            "fluency": [],
            "terminology": [],
            "hallucination": [],
            "context": [],
        }

        result = consensus.calculate_weighted_score(agent_results, word_count=100)

        # All agents perfect = high confidence
        assert result["confidence"] >= 0.8
        assert result["weighted_mqm_score"] == 100.0
        assert result["agent_agreement"] >= 0.99
