"""Unit tests for model selector module.

Tests intelligent LLM model selection based on task characteristics.
"""

import pytest

from kttc.llm.model_selector import ModelSelector


@pytest.mark.unit
class TestModelSelector:
    """Test ModelSelector functionality."""

    @pytest.fixture
    def selector(self) -> ModelSelector:
        """Create a model selector instance."""
        return ModelSelector()

    def test_selector_initialization(self, selector: ModelSelector) -> None:
        """Test selector initializes correctly."""
        assert selector is not None
        assert hasattr(selector, "PERFORMANCE_MATRIX")
        assert hasattr(selector, "DOMAIN_PREFERENCES")
        assert hasattr(selector, "MODEL_COSTS")
        assert hasattr(selector, "PROVIDER_MAP")

    def test_select_best_model_for_domain_legal(self, selector: ModelSelector) -> None:
        """Test domain preference for legal domain."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="ru",
            domain="legal",
        )
        assert result == "gpt-4.5"

    def test_select_best_model_for_domain_medical(self, selector: ModelSelector) -> None:
        """Test domain preference for medical domain."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="es",
            domain="medical",
        )
        assert result == "gpt-4.5"

    def test_select_best_model_for_domain_general(self, selector: ModelSelector) -> None:
        """Test domain preference for general domain."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="de",
            domain="general",
        )
        assert result == "claude-3.5-sonnet"

    def test_select_best_model_for_domain_creative(self, selector: ModelSelector) -> None:
        """Test domain preference for creative domain."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="fr",
            domain="creative",
        )
        assert result == "claude-3.5-sonnet"

    def test_select_best_model_for_domain_financial(self, selector: ModelSelector) -> None:
        """Test domain preference for financial domain."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="zh",
            domain="financial",
        )
        assert result == "gpt-4.5"

    def test_select_best_model_language_pair_en_ru(self, selector: ModelSelector) -> None:
        """Test language pair selection for en->ru without domain."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="ru",
        )
        # yandexgpt has highest score for en->ru (0.88)
        assert result == "yandexgpt"

    def test_select_best_model_language_pair_en_fr(self, selector: ModelSelector) -> None:
        """Test language pair selection for en->fr without domain."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="fr",
        )
        # claude-3.5-sonnet has highest score for en->fr (0.90)
        assert result == "claude-3.5-sonnet"

    def test_select_best_model_language_pair_ru_en(self, selector: ModelSelector) -> None:
        """Test language pair selection for ru->en."""
        result = selector.select_best_model(
            source_lang="ru",
            target_lang="en",
        )
        # yandexgpt has highest score for ru->en (0.87)
        assert result == "yandexgpt"

    def test_select_best_model_fallback_unknown_pair(self, selector: ModelSelector) -> None:
        """Test fallback for unknown language pair."""
        result = selector.select_best_model(
            source_lang="xx",
            target_lang="yy",
        )
        # Should fallback to claude-3.5-sonnet
        assert result == "claude-3.5-sonnet"

    def test_select_best_model_optimize_for_cost(self, selector: ModelSelector) -> None:
        """Test cost optimization selection."""
        result = selector.select_best_model(
            source_lang="en",
            target_lang="ru",
            optimize_for="cost",
        )
        # Should select cost-effective model
        assert result in ["gigachat", "yandexgpt", "claude-3.5-sonnet", "gpt-4.5"]

    def test_select_cost_effective(self, selector: ModelSelector) -> None:
        """Test _select_cost_effective method."""
        scores = {
            "claude-3.5-sonnet": 0.85,
            "gpt-4.5": 0.87,
            "gigachat": 0.86,
        }
        result = selector._select_cost_effective(scores)
        # gigachat should win (0.86 / 1.0 = 0.86 value) over claude (0.85/3.0=0.28)
        assert result == "gigachat"

    def test_get_provider_for_model_claude(self, selector: ModelSelector) -> None:
        """Test provider lookup for Claude model."""
        result = selector.get_provider_for_model("claude-3.5-sonnet")
        assert result == "anthropic"

    def test_get_provider_for_model_gpt(self, selector: ModelSelector) -> None:
        """Test provider lookup for GPT model."""
        result = selector.get_provider_for_model("gpt-4.5")
        assert result == "openai"

    def test_get_provider_for_model_gigachat(self, selector: ModelSelector) -> None:
        """Test provider lookup for GigaChat model."""
        result = selector.get_provider_for_model("gigachat")
        assert result == "gigachat"

    def test_get_provider_for_model_yandex(self, selector: ModelSelector) -> None:
        """Test provider lookup for Yandex model."""
        result = selector.get_provider_for_model("yandexgpt")
        assert result == "yandex"

    def test_get_provider_for_unknown_model_raises(self, selector: ModelSelector) -> None:
        """Test provider lookup for unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            selector.get_provider_for_model("unknown-model")


@pytest.mark.unit
class TestModelSelectorConstants:
    """Test ModelSelector class constants."""

    def test_performance_matrix_structure(self) -> None:
        """Test performance matrix has correct structure."""
        selector = ModelSelector()
        for lang_pair, scores in selector.PERFORMANCE_MATRIX.items():
            assert isinstance(lang_pair, tuple)
            assert len(lang_pair) == 2
            assert isinstance(scores, dict)
            for model, score in scores.items():
                assert isinstance(model, str)
                assert isinstance(score, float)
                assert 0 <= score <= 1

    def test_domain_preferences_structure(self) -> None:
        """Test domain preferences has correct structure."""
        selector = ModelSelector()
        expected_domains = ["legal", "medical", "general", "technical", "creative", "financial"]
        for domain in expected_domains:
            assert domain in selector.DOMAIN_PREFERENCES
            assert isinstance(selector.DOMAIN_PREFERENCES[domain], str)

    def test_model_costs_structure(self) -> None:
        """Test model costs has correct structure."""
        selector = ModelSelector()
        for model, cost in selector.MODEL_COSTS.items():
            assert isinstance(model, str)
            assert isinstance(cost, float)
            assert cost > 0

    def test_provider_map_structure(self) -> None:
        """Test provider map has correct structure."""
        selector = ModelSelector()
        for model, provider in selector.PROVIDER_MAP.items():
            assert isinstance(model, str)
            assert isinstance(provider, str)
