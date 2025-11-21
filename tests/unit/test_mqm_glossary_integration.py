"""Integration tests for MQM scorer with new glossary files.

This module tests that the MQM scoring system can successfully:
1. Load the new MQM error taxonomy glossary
2. Load the MQM core glossary
3. Use glossary-based weights for scoring
4. Access severity levels from glossaries
"""

import pytest

from kttc.core.models import ErrorAnnotation
from kttc.core.mqm import MQMScorer
from kttc.terminology import GlossaryManager


class TestMQMGlossaryIntegration:
    """Test MQM scorer integration with glossary system."""

    def test_mqm_scorer_loads_glossary_weights(self):
        """Test that MQM scorer can load category weights from glossary."""
        scorer = MQMScorer(use_glossary_weights=True)

        assert scorer.glossary_manager is not None
        assert scorer.category_weights is not None
        assert len(scorer.category_weights) > 0

    def test_mqm_scorer_can_load_error_taxonomy(self):
        """Test that glossary manager can load the new MQM error taxonomy."""
        manager = GlossaryManager()

        # Should be able to load the new comprehensive error taxonomy
        taxonomy = manager.load_glossary("en", "mqm_error_taxonomy")

        assert taxonomy is not None
        assert "metadata" in taxonomy
        assert "severity_levels" in taxonomy
        assert "accuracy" in taxonomy
        assert "fluency" in taxonomy

    def test_mqm_scorer_severity_levels(self):
        """Test that MQM scorer can access severity levels from glossary."""
        manager = GlossaryManager()
        severity_levels = manager.get_severity_levels("en")

        # Should have standard MQM severity levels
        assert "minor" in severity_levels or "neutral" in severity_levels

    def test_mqm_scorer_calculates_score(self):
        """Test that MQM scorer can calculate scores using glossary data."""
        scorer = MQMScorer(use_glossary_weights=True)

        # Create test errors with proper location tuples
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity="major",
                location=(0, 10),
                description="Incorrect translation",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity="minor",
                location=(15, 20),
                description="Minor grammar error",
            ),
        ]

        # Calculate score
        score = scorer.calculate_score(errors, word_count=100)

        # Score should be between 0 and 100
        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_mqm_error_taxonomy_has_required_categories(self):
        """Test that new error taxonomy has all required MQM categories."""
        manager = GlossaryManager()
        taxonomy = manager.load_glossary("en", "mqm_error_taxonomy")

        # Check for main MQM categories
        required_categories = ["accuracy", "fluency", "terminology", "style"]
        for category in required_categories:
            assert category in taxonomy, f"Missing required category: {category}"

    def test_mqm_taxonomy_has_severity_weights(self):
        """Test that taxonomy includes severity level weights."""
        manager = GlossaryManager()
        taxonomy = manager.load_glossary("en", "mqm_error_taxonomy")

        assert "severity_levels" in taxonomy
        severity = taxonomy["severity_levels"]

        # Check standard MQM weights
        assert "minor" in severity
        assert "major" in severity
        assert "critical" in severity

        # Verify weights match MQM standard (1, 5, 10)
        assert severity["minor"]["weight"] == 1
        assert severity["major"]["weight"] == 5
        assert severity["critical"]["weight"] == 10

    def test_mqm_taxonomy_metadata(self):
        """Test that taxonomy has proper metadata."""
        manager = GlossaryManager()
        taxonomy = manager.load_glossary("en", "mqm_error_taxonomy")

        # Check metadata in the glossary
        assert "metadata" in taxonomy
        metadata_dict = taxonomy["metadata"]
        assert metadata_dict["language"] == "en"
        assert "version" in metadata_dict
        assert (
            "MQM" in metadata_dict["glossary_type"]
            or "mqm" in metadata_dict["glossary_type"].lower()
        )


class TestTranslationMetricsGlossary:
    """Test translation metrics glossary integration."""

    def test_translation_metrics_loads(self):
        """Test that translation metrics glossary can be loaded."""
        manager = GlossaryManager()
        metrics = manager.load_glossary("en", "translation_metrics")

        assert metrics is not None
        assert "metadata" in metrics
        assert "automated_metrics" in metrics

    def test_translation_metrics_has_bleu(self):
        """Test that BLEU metric is documented."""
        manager = GlossaryManager()
        metrics = manager.load_glossary("en", "translation_metrics")

        assert "automated_metrics" in metrics
        assert "bleu" in metrics["automated_metrics"]

        bleu = metrics["automated_metrics"]["bleu"]
        assert "acronym" in bleu
        assert bleu["acronym"] == "BLEU"

    def test_translation_metrics_has_comet(self):
        """Test that COMET metric is documented."""
        manager = GlossaryManager()
        metrics = manager.load_glossary("en", "translation_metrics")

        assert "automated_metrics" in metrics
        assert "comet" in metrics["automated_metrics"]

        comet = metrics["automated_metrics"]["comet"]
        assert "versions" in comet
        assert "comet_22" in comet["versions"]


class TestLLMTerminologyGlossary:
    """Test LLM terminology glossary integration."""

    def test_llm_terminology_loads(self):
        """Test that LLM terminology glossary can be loaded."""
        manager = GlossaryManager()
        llm_terms = manager.load_glossary("en", "llm_terminology")

        assert llm_terms is not None
        assert "metadata" in llm_terms
        assert "hallucination" in llm_terms

    def test_llm_hallucination_definition(self):
        """Test that hallucination is properly defined."""
        manager = GlossaryManager()
        llm_terms = manager.load_glossary("en", "llm_terminology")

        assert "hallucination" in llm_terms
        hallucination = llm_terms["hallucination"]
        assert "definition" in hallucination

    def test_llm_rlhf_documented(self):
        """Test that RLHF is documented."""
        manager = GlossaryManager()
        llm_terms = manager.load_glossary("en", "llm_terminology")

        assert "rlhf" in llm_terms
        rlhf = llm_terms["rlhf"]
        assert "process" in rlhf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
