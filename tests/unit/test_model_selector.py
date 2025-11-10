"""Tests for LLM Model Selector."""

import pytest

from kttc.llm.model_selector import ModelSelector


class TestModelSelector:
    """Tests for ModelSelector class."""

    @pytest.fixture
    def selector(self):
        """Create ModelSelector instance."""
        return ModelSelector()

    def test_select_by_domain_legal(self, selector):
        """Test domain-based selection for legal domain."""
        model = selector.select_best_model(source_lang="en", target_lang="es", domain="legal")
        assert model == "gpt-4.5"  # Best for legal

    def test_select_by_domain_medical(self, selector):
        """Test domain-based selection for medical domain."""
        model = selector.select_best_model(source_lang="en", target_lang="fr", domain="medical")
        assert model == "gpt-4.5"  # Best for medical

    def test_select_by_language_pair_en_ru(self, selector):
        """Test selection for English-Russian pair."""
        model = selector.select_best_model(source_lang="en", target_lang="ru")
        # Should select YandexGPT (best for Russian)
        assert model in ["yandexgpt", "gigachat", "claude-3.5-sonnet"]

    def test_select_by_language_pair_en_zh(self, selector):
        """Test selection for English-Chinese pair."""
        model = selector.select_best_model(source_lang="en", target_lang="zh")
        assert model == "gpt-4.5"  # Best for Chinese

    def test_optimize_for_cost(self, selector):
        """Test cost optimization."""
        model = selector.select_best_model(source_lang="en", target_lang="ru", optimize_for="cost")
        # Should prefer cost-effective options
        assert model in selector.MODEL_COSTS

    def test_fallback_for_unsupported_pair(self, selector):
        """Test fallback for unsupported language pair."""
        model = selector.select_best_model(
            source_lang="xx", target_lang="yy"  # Unsupported  # Unsupported
        )
        assert model == "claude-3.5-sonnet"  # Default fallback

    def test_get_provider_for_model(self, selector):
        """Test provider mapping."""
        assert selector.get_provider_for_model("claude-3.5-sonnet") == "anthropic"
        assert selector.get_provider_for_model("gpt-4.5") == "openai"
        assert selector.get_provider_for_model("yandexgpt") == "yandex"

    def test_get_provider_for_unknown_model(self, selector):
        """Test error handling for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            selector.get_provider_for_model("unknown-model")

    def test_get_supported_language_pairs(self, selector):
        """Test getting supported language pairs."""
        pairs = selector.get_supported_language_pairs()
        assert len(pairs) > 0
        assert ("en", "es") in pairs
        assert ("en", "ru") in pairs

    def test_get_model_info(self, selector):
        """Test getting model information."""
        info = selector.get_model_info("claude-3.5-sonnet")
        assert info["model"] == "claude-3.5-sonnet"
        assert info["provider"] == "anthropic"
        assert info["cost"] > 0
        assert isinstance(info["domains"], list)

    def test_recommend_models(self, selector):
        """Test getting model recommendations."""
        recommendations = selector.recommend_models(source_lang="en", target_lang="ru", top_n=3)
        assert len(recommendations) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recommendations)
        # Should be sorted by score
        scores = [r[1] for r in recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_domain_overrides_language_pair(self, selector):
        """Test that domain preference overrides language pair."""
        # Even though yandexgpt is best for en-ru, legal domain should pick gpt-4.5
        model = selector.select_best_model(source_lang="en", target_lang="ru", domain="legal")
        assert model == "gpt-4.5"
