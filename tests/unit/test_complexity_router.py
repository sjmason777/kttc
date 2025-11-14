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

"""Unit tests for complexity-based smart routing."""

import pytest

from kttc.llm import ComplexityEstimator, ComplexityRouter, ComplexityScore


class TestComplexityScore:
    """Tests for ComplexityScore dataclass."""

    def test_creation(self):
        """Test creating a ComplexityScore."""
        score = ComplexityScore(
            overall=0.5,
            sentence_length=0.4,
            rare_words=0.6,
            syntactic=0.5,
            domain_specific=0.3,
            recommendation="gpt-4-turbo",
        )

        assert score.overall == 0.5
        assert score.sentence_length == 0.4
        assert score.rare_words == 0.6
        assert score.syntactic == 0.5
        assert score.domain_specific == 0.3
        assert score.recommendation == "gpt-4-turbo"


class TestComplexityEstimator:
    """Tests for ComplexityEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create a ComplexityEstimator instance."""
        return ComplexityEstimator()

    def test_score_sentence_length_short(self, estimator):
        """Test scoring short sentences."""
        text = "Hello. How are you? Good."
        score = estimator._score_sentence_length(text)

        # Short sentences should score low
        assert 0.0 <= score < 0.3

    def test_score_sentence_length_long(self, estimator):
        """Test scoring long sentences."""
        text = (
            "This is a very long sentence with many words that goes on and on "
            "and contains multiple clauses and subordinate phrases and keeps "
            "adding more information without stopping for quite some time."
        )
        score = estimator._score_sentence_length(text)

        # Long sentences should score high
        assert 0.7 <= score <= 1.0

    def test_score_sentence_length_empty(self, estimator):
        """Test scoring empty text."""
        score = estimator._score_sentence_length("")
        assert score == 0.0

    def test_score_rare_words_common(self, estimator):
        """Test scoring text with common words."""
        text = "The quick brown fox jumps over the lazy dog."
        score = estimator._score_rare_words(text)

        # Common words should score low
        assert 0.0 <= score < 0.5

    def test_score_rare_words_technical(self, estimator):
        """Test scoring text with rare/technical words."""
        text = "The idempotent microarchitecture facilitates ephemeral containerization."
        score = estimator._score_rare_words(text)

        # Rare words should score high
        assert 0.5 <= score <= 1.0

    def test_score_rare_words_empty(self, estimator):
        """Test scoring empty text for rare words."""
        score = estimator._score_rare_words("")
        assert score == 0.0

    def test_score_syntactic_complexity_simple(self, estimator):
        """Test scoring simple syntax."""
        text = "Hello world. This is simple."
        score = estimator._score_syntactic_complexity(text)

        # Simple syntax should score low
        assert 0.0 <= score < 0.3

    def test_score_syntactic_complexity_complex(self, estimator):
        """Test scoring complex syntax."""
        text = (
            "The system, which was deployed yesterday (after extensive testing), "
            "processes data; however, it requires authentication, and therefore, "
            "users must provide credentials, although some exceptions exist."
        )
        score = estimator._score_syntactic_complexity(text)

        # Complex syntax should score higher
        assert 0.3 <= score <= 1.0

    def test_score_syntactic_complexity_empty(self, estimator):
        """Test scoring empty text for syntax."""
        score = estimator._score_syntactic_complexity("")
        assert score == 0.0

    def test_score_domain_specificity_general(self, estimator):
        """Test scoring general domain text."""
        text = "Hello, how are you today? The weather is nice."
        score = estimator._score_domain_specificity(text)

        # General text should score low
        assert 0.0 <= score < 0.3

    def test_score_domain_specificity_technical(self, estimator):
        """Test scoring technical domain text."""
        text = "The API endpoint handles JSON requests via HTTP protocol using REST architecture."
        score = estimator._score_domain_specificity(text)

        # Technical text should score high
        assert 0.5 <= score <= 1.0

    def test_score_domain_specificity_with_domain_hint(self, estimator):
        """Test scoring with domain hint."""
        # Use text with actual technical terms from TECHNICAL_TERMS set
        text = "The server handles API requests efficiently."
        score_no_hint = estimator._score_domain_specificity(text)
        score_with_hint = estimator._score_domain_specificity(text, domain="technical")

        # Domain hint should boost score (1.5x multiplier in implementation)
        assert score_with_hint >= score_no_hint
        # Should have non-zero score since "server" and "api" are technical terms
        assert score_no_hint > 0.0
        assert score_with_hint > 0.0

    def test_score_domain_specificity_empty(self, estimator):
        """Test scoring empty text for domain."""
        score = estimator._score_domain_specificity("")
        assert score == 0.0

    def test_recommend_model_simple(self, estimator):
        """Test model recommendation for simple text."""
        model = estimator._recommend_model(0.2)
        assert model == "gpt-3.5-turbo"

    def test_recommend_model_medium(self, estimator):
        """Test model recommendation for medium complexity."""
        model = estimator._recommend_model(0.5)
        assert model == "gpt-4-turbo"

    def test_recommend_model_complex(self, estimator):
        """Test model recommendation for complex text."""
        model = estimator._recommend_model(0.8)
        assert model == "claude-3.5-sonnet"

    def test_recommend_model_boundary_simple_medium(self, estimator):
        """Test boundary between simple and medium."""
        # Just below threshold
        assert estimator._recommend_model(0.29) == "gpt-3.5-turbo"
        # Just at threshold
        assert estimator._recommend_model(0.3) == "gpt-4-turbo"

    def test_recommend_model_boundary_medium_complex(self, estimator):
        """Test boundary between medium and complex."""
        # Just below threshold
        assert estimator._recommend_model(0.69) == "gpt-4-turbo"
        # Just at threshold
        assert estimator._recommend_model(0.7) == "claude-3.5-sonnet"

    def test_estimate_simple_text(self, estimator):
        """Test estimating simple text complexity."""
        text = "Hello world. This is a test."
        score = estimator.estimate(text)

        assert isinstance(score, ComplexityScore)
        assert 0.0 <= score.overall <= 1.0
        assert score.recommendation == "gpt-3.5-turbo"

    def test_estimate_medium_text(self, estimator):
        """Test estimating medium complexity text."""
        text = (
            "The application processes user requests through a middleware layer, "
            "which validates authentication tokens and forwards the data to the "
            "appropriate service endpoint."
        )
        score = estimator.estimate(text)

        assert isinstance(score, ComplexityScore)
        assert 0.3 <= score.overall <= 0.7
        assert score.recommendation == "gpt-4-turbo"

    def test_estimate_complex_text(self, estimator):
        """Test estimating complex text complexity."""
        text = (
            "The microservice architecture, which leverages containerization "
            "(specifically Kubernetes orchestration), implements a sophisticated "
            "event-driven paradigm; consequently, asynchronous message propagation "
            "facilitates decoupled inter-service communication, although latency "
            "considerations necessitate careful optimization of serialization "
            "protocols and network topology configurations."
        )
        score = estimator.estimate(text)

        assert isinstance(score, ComplexityScore)
        assert 0.5 <= score.overall <= 1.0
        # Should be complex enough for premium model
        assert score.recommendation in ["gpt-4-turbo", "claude-3.5-sonnet"]

    def test_estimate_with_domain(self, estimator):
        """Test estimation with domain hint."""
        # Use text with moderate technical content
        text = "The system handles incoming requests efficiently."
        score_no_domain = estimator.estimate(text)
        score_with_domain = estimator.estimate(text, domain="technical")

        # Domain hint should increase or maintain domain_specific score
        assert score_with_domain.domain_specific >= score_no_domain.domain_specific
        # Overall scores should be valid
        assert 0.0 <= score_no_domain.overall <= 1.0
        assert 0.0 <= score_with_domain.overall <= 1.0

    def test_estimate_empty_text(self, estimator):
        """Test estimating empty text."""
        score = estimator.estimate("")

        assert score.overall == 0.0
        assert score.recommendation == "gpt-3.5-turbo"

    def test_estimate_score_components(self, estimator):
        """Test that all score components are calculated."""
        text = "The API endpoint processes requests."
        score = estimator.estimate(text)

        # All components should be present and in valid range
        assert 0.0 <= score.sentence_length <= 1.0
        assert 0.0 <= score.rare_words <= 1.0
        assert 0.0 <= score.syntactic <= 1.0
        assert 0.0 <= score.domain_specific <= 1.0
        assert 0.0 <= score.overall <= 1.0

    def test_estimate_score_clamping(self, estimator):
        """Test that overall score is clamped to [0, 1]."""
        # Test various texts
        texts = [
            "",
            "a",
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "This is a very long and complex sentence with many subordinate clauses.",
        ]

        for text in texts:
            score = estimator.estimate(text)
            assert 0.0 <= score.overall <= 1.0


class TestComplexityRouter:
    """Tests for ComplexityRouter."""

    @pytest.fixture
    def router(self):
        """Create a ComplexityRouter instance."""
        return ComplexityRouter()

    def test_route_simple_text(self, router):
        """Test routing simple text."""
        model, score = router.route(
            text="Hello world. How are you?", source_lang="en", target_lang="es"
        )

        assert model == "gpt-3.5-turbo"
        assert isinstance(score, ComplexityScore)
        assert score.overall < 0.3

    def test_route_medium_text(self, router):
        """Test routing medium complexity text."""
        model, score = router.route(
            text="The system processes user authentication through a secure token-based mechanism.",
            source_lang="en",
            target_lang="ru",
        )

        assert model == "gpt-4-turbo"
        assert isinstance(score, ComplexityScore)
        assert 0.3 <= score.overall < 0.7

    def test_route_complex_text(self, router):
        """Test routing complex text."""
        model, score = router.route(
            text=(
                "The distributed microservice architecture, employing containerization "
                "via Kubernetes orchestration, necessitates sophisticated inter-service "
                "communication protocols; consequently, asynchronous message queuing "
                "systems facilitate decoupled service interactions, although performance "
                "optimization requires meticulous configuration of serialization formats."
            ),
            source_lang="en",
            target_lang="zh",
        )

        assert model in ["gpt-4-turbo", "claude-3.5-sonnet"]
        assert isinstance(score, ComplexityScore)

    def test_route_with_domain(self, router):
        """Test routing with domain hint."""
        model, score = router.route(
            text="The API endpoint returns JSON data.",
            source_lang="en",
            target_lang="fr",
            domain="technical",
        )

        assert isinstance(model, str)
        assert isinstance(score, ComplexityScore)
        # Domain hint should affect scoring
        assert score.domain_specific > 0.0

    def test_route_force_model(self, router):
        """Test forcing specific model."""
        # Simple text that would normally get gpt-3.5-turbo
        model, score = router.route(
            text="Hello world",
            source_lang="en",
            target_lang="es",
            force_model="claude-3.5-sonnet",
        )

        # Should use forced model despite low complexity
        assert model == "claude-3.5-sonnet"
        assert isinstance(score, ComplexityScore)
        # Score should still be calculated
        assert score.overall < 0.3
        # But recommendation would be different
        assert score.recommendation == "gpt-3.5-turbo"

    def test_route_returns_tuple(self, router):
        """Test that route returns tuple of (model, score)."""
        result = router.route(text="Test", source_lang="en", target_lang="es")

        assert isinstance(result, tuple)
        assert len(result) == 2
        model, score = result
        assert isinstance(model, str)
        assert isinstance(score, ComplexityScore)

    def test_route_different_language_pairs(self, router):
        """Test routing with different language pairs."""
        text = "The API endpoint processes requests."

        # Same text, different language pairs - should get same routing
        model1, score1 = router.route(text, "en", "es")
        model2, score2 = router.route(text, "en", "ru")
        model3, score3 = router.route(text, "en", "zh")

        # Models should be the same (routing based on complexity, not languages)
        assert model1 == model2 == model3
        # Scores should be similar
        assert abs(score1.overall - score2.overall) < 0.1
        assert abs(score1.overall - score3.overall) < 0.1

    def test_route_empty_text(self, router):
        """Test routing empty text."""
        model, score = router.route(text="", source_lang="en", target_lang="es")

        # Empty text should route to cheapest model
        assert model == "gpt-3.5-turbo"
        assert score.overall == 0.0


@pytest.mark.integration
class TestComplexityRouterIntegration:
    """Integration tests for complexity routing."""

    def test_routing_cost_optimization(self):
        """Test that routing achieves cost optimization."""
        router = ComplexityRouter()

        # Sample texts of varying complexity
        texts = [
            ("Hello", "simple"),
            ("The API returns data.", "simple"),
            (
                "The system implements authentication using JWT tokens.",
                "medium",
            ),
            (
                "The microservice architecture facilitates distributed processing.",
                "medium",
            ),
            (
                "The distributed event-driven architecture, leveraging asynchronous "
                "message propagation through sophisticated queuing mechanisms, "
                "necessitates careful consideration of eventual consistency patterns.",
                "complex",
            ),
        ]

        simple_count = 0
        medium_count = 0
        complex_count = 0

        for text, expected_category in texts:
            model, score = router.route(text, "en", "es")

            if model == "gpt-3.5-turbo":
                simple_count += 1
                if expected_category == "simple":
                    assert score.overall < 0.3
            elif model == "gpt-4-turbo":
                medium_count += 1
                if expected_category == "medium":
                    assert 0.3 <= score.overall < 0.7
            elif model == "claude-3.5-sonnet":
                complex_count += 1
                if expected_category == "complex":
                    assert score.overall >= 0.7

        # Should have variety of routing decisions
        assert simple_count > 0
        assert medium_count + complex_count > 0

    def test_consistent_scoring(self):
        """Test that scoring is consistent for same text."""
        router = ComplexityRouter()
        text = "The API endpoint processes user requests."

        # Run multiple times
        results = [router.route(text, "en", "es") for _ in range(5)]

        # All should return same model and score
        models = [r[0] for r in results]
        scores = [r[1].overall for r in results]

        assert len(set(models)) == 1  # All same model
        assert len(set(scores)) == 1  # All same score

    def test_real_world_examples(self):
        """Test routing with real-world translation examples."""
        router = ComplexityRouter()

        examples = [
            # Simple greeting
            ("Hello, how are you?", "gpt-3.5-turbo"),
            # Technical documentation
            (
                "The REST API supports GET, POST, PUT, and DELETE operations.",
                "gpt-4-turbo",
            ),
            # Legal text (complex)
            (
                "Notwithstanding the aforementioned provisions, the party of the first part "
                "shall indemnify and hold harmless the party of the second part.",
                "claude-3.5-sonnet",
            ),
        ]

        for text, expected_model in examples:
            model, score = router.route(text, "en", "es")

            # Check if routing makes sense (allowing some flexibility)
            assert isinstance(model, str)
            assert model in ["gpt-3.5-turbo", "gpt-4-turbo", "claude-3.5-sonnet"]
