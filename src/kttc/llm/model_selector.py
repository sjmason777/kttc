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

"""Intelligent LLM Model Selection.

Selects optimal LLM model based on:
- Language pair
- Domain
- Task type (translation QA, translation generation)
- Cost vs quality tradeoffs

Based on Best LLMs for Translation 2025 research findings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModelSelector:
    """Intelligent LLM selection based on task characteristics.

    Performance matrix based on 2025 research findings:
    - Claude 3.5 Sonnet: Best general-purpose (78% good translations)
    - GPT-4.5/o1: Best for legal/medical domains
    - Different models optimal for different language pairs

    Example:
        >>> selector = ModelSelector()
        >>> model = selector.select_best_model(
        ...     source_lang="en",
        ...     target_lang="ru",
        ...     domain="legal"
        ... )
        >>> print(model)  # "gpt-4.5" (best for legal)
    """

    # Performance matrix from 2025 research
    # Format: (source_lang, target_lang): {model: score}
    PERFORMANCE_MATRIX = {
        # English to other languages
        ("en", "es"): {
            "claude-3.5-sonnet": 0.89,
            "gpt-4.5": 0.87,
            "gemini-2.0": 0.85,
        },
        ("en", "ru"): {
            "claude-3.5-sonnet": 0.86,
            "gpt-4.5": 0.84,
            "yandexgpt": 0.88,  # Best for Russian
            "gigachat": 0.87,  # Sber's model, strong for Russian
        },
        ("en", "zh"): {
            "gpt-4.5": 0.88,
            "claude-3.5-sonnet": 0.85,
            "gemini-2.0": 0.87,
        },
        ("en", "fr"): {
            "claude-3.5-sonnet": 0.90,
            "gpt-4.5": 0.88,
            "gemini-2.0": 0.86,
        },
        ("en", "de"): {
            "claude-3.5-sonnet": 0.89,
            "gpt-4.5": 0.87,
            "gemini-2.0": 0.85,
        },
        # Russian to English
        ("ru", "en"): {
            "yandexgpt": 0.87,
            "gigachat": 0.86,
            "claude-3.5-sonnet": 0.84,
            "gpt-4.5": 0.83,
        },
    }

    # Domain preferences (overrides language pair)
    DOMAIN_PREFERENCES = {
        "legal": "gpt-4.5",  # Best for legal domain
        "medical": "gpt-4.5",  # High accuracy needed
        "general": "claude-3.5-sonnet",  # Best overall
        "technical": "claude-3.5-sonnet",  # Technical accuracy
        "creative": "claude-3.5-sonnet",  # Better at nuance
        "financial": "gpt-4.5",  # Precision needed
    }

    # Cost tiers (relative cost per 1M tokens)
    MODEL_COSTS = {
        "claude-3.5-sonnet": 3.0,
        "gpt-4.5": 5.0,
        "gemini-2.0": 2.0,
        "yandexgpt": 1.5,
        "gigachat": 1.0,
    }

    # Provider availability mapping
    PROVIDER_MAP = {
        "claude-3.5-sonnet": "anthropic",
        "gpt-4.5": "openai",
        "gpt-4-turbo": "openai",
        "gemini-2.0": "google",
        "yandexgpt": "yandex",
        "gigachat": "gigachat",
    }

    def select_best_model(
        self,
        source_lang: str,
        target_lang: str,
        domain: str | None = None,
        task_type: str = "qa",  # "qa" or "translation"
        optimize_for: str = "quality",  # "quality" or "cost"
    ) -> str:
        """Select optimal model for given task.

        Args:
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)
            domain: Domain category (optional)
            task_type: Task type - "qa" (quality assurance) or "translation"
            optimize_for: Optimization target - "quality" or "cost"

        Returns:
            Model identifier (e.g., "claude-3.5-sonnet")

        Example:
            >>> selector = ModelSelector()
            >>> model = selector.select_best_model("en", "ru", domain="legal")
            >>> print(model)  # "gpt-4.5"
        """
        # 1. Check domain preference (highest priority)
        if domain and domain in self.DOMAIN_PREFERENCES:
            preferred = self.DOMAIN_PREFERENCES[domain]
            logger.info(f"Selected model '{preferred}' based on domain '{domain}'")
            return preferred

        # 2. Check language pair performance
        lang_pair = (source_lang, target_lang)
        if lang_pair in self.PERFORMANCE_MATRIX:
            scores = self.PERFORMANCE_MATRIX[lang_pair]

            if optimize_for == "cost":
                # Balance quality and cost
                best_model = self._select_cost_effective(scores)
            else:
                # Pure quality optimization
                best_model = max(scores.items(), key=lambda x: x[1])[0]

            logger.info(
                f"Selected model '{best_model}' for {source_lang}->{target_lang} "
                f"(optimize_for={optimize_for})"
            )
            return best_model

        # 3. Fallback to general-purpose best
        logger.info(f"Using fallback model for {source_lang}->{target_lang}")
        return "claude-3.5-sonnet"

    def _select_cost_effective(self, performance_scores: dict[str, float]) -> str:
        """Select most cost-effective model balancing quality and cost.

        Uses value score: performance / cost

        Args:
            performance_scores: Dict of model -> performance score

        Returns:
            Model identifier
        """
        value_scores = {}

        for model, perf_score in performance_scores.items():
            cost = self.MODEL_COSTS.get(model, 3.0)
            # Value = performance per unit cost
            value_scores[model] = perf_score / cost

        # Return model with best value
        best_model = max(value_scores.items(), key=lambda x: x[1])[0]
        logger.debug(f"Cost-effective selection: {best_model} (value scores: {value_scores})")
        return best_model

    def get_provider_for_model(self, model: str) -> str:
        """Get LLM provider name for model.

        Args:
            model: Model identifier

        Returns:
            Provider name (e.g., "anthropic", "openai")

        Raises:
            ValueError: If model not supported
        """
        provider = self.PROVIDER_MAP.get(model)

        if provider is None:
            raise ValueError(
                f"Unknown model: {model}. " f"Supported models: {list(self.PROVIDER_MAP.keys())}"
            )

        return provider

    def get_supported_language_pairs(self) -> list[tuple[str, str]]:
        """Get list of supported language pairs.

        Returns:
            List of (source_lang, target_lang) tuples
        """
        return list(self.PERFORMANCE_MATRIX.keys())

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get detailed information about a model.

        Args:
            model: Model identifier

        Returns:
            Dictionary with model information

        Example:
            >>> selector = ModelSelector()
            >>> info = selector.get_model_info("claude-3.5-sonnet")
            >>> print(info["cost"])  # 3.0
        """
        return {
            "model": model,
            "provider": self.PROVIDER_MAP.get(model, "unknown"),
            "cost": self.MODEL_COSTS.get(model, 0.0),
            "domains": [
                domain
                for domain, preferred_model in self.DOMAIN_PREFERENCES.items()
                if preferred_model == model
            ],
        }

    def recommend_models(
        self,
        source_lang: str,
        target_lang: str,
        domain: str | None = None,
        top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """Get top N recommended models with scores.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain category (optional)
            top_n: Number of recommendations

        Returns:
            List of (model, score) tuples sorted by score

        Example:
            >>> selector = ModelSelector()
            >>> models = selector.recommend_models("en", "ru", top_n=2)
            >>> for model, score in models:
            ...     print(f"{model}: {score:.2f}")
        """
        lang_pair = (source_lang, target_lang)

        if lang_pair not in self.PERFORMANCE_MATRIX:
            # Fallback recommendation
            return [("claude-3.5-sonnet", 0.85)]

        scores = self.PERFORMANCE_MATRIX[lang_pair]

        # Sort by score descending
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_models[:top_n]
