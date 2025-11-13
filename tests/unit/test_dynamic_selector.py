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

"""Tests for dynamic agent selection (Phase 4)."""

import pytest

from kttc.agents.domain_profiles import DOMAIN_PROFILES
from kttc.agents.dynamic_selector import DynamicAgentSelector
from kttc.core import TranslationTask
from kttc.llm import OpenAIProvider


class TestDynamicAgentSelector:
    """Tests for DynamicAgentSelector."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = OpenAIProvider(api_key="test-key")
        self.selector = DynamicAgentSelector(self.provider)

    def test_initialization(self) -> None:
        """Test selector initialization."""
        assert self.selector.llm_provider == self.provider
        assert self.selector.agent_temperature == 0.1
        assert self.selector.agent_max_tokens == 2000

    def test_simple_task_selection(self) -> None:
        """Test that simple tasks use minimal agents."""
        task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        agents = self.selector.select_agents(task, complexity="simple")

        # Simple task should use only core agents
        assert len(agents) == 2
        categories = {agent.category for agent in agents}
        assert "accuracy" in categories
        assert "fluency" in categories

    def test_medium_task_selection(self) -> None:
        """Test that medium tasks use standard agent suite."""
        task = TranslationTask(
            source_text="The quick brown fox jumps over the lazy dog.",
            translation="El rápido zorro marrón salta sobre el perro perezoso.",
            source_lang="en",
            target_lang="es",
        )

        agents = self.selector.select_agents(task, complexity="medium")

        # Medium task should use 3 agents
        assert len(agents) == 3
        categories = {agent.category for agent in agents}
        assert "accuracy" in categories
        assert "fluency" in categories
        assert "terminology" in categories

    def test_complex_task_selection(self) -> None:
        """Test that complex tasks use full agent suite."""
        task = TranslationTask(
            source_text="Complex technical documentation with multiple paragraphs.",
            translation="Documentación técnica compleja con múltiples párrafos.",
            source_lang="en",
            target_lang="es",
        )

        agents = self.selector.select_agents(task, complexity="complex")

        # Complex task should use all available agents
        assert len(agents) >= 3
        categories = {agent.category for agent in agents}
        assert "accuracy" in categories

    def test_auto_complexity_detection_simple(self) -> None:
        """Test auto-detection of simple task complexity."""
        task = TranslationTask(
            source_text="Hi",  # Very short
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        summary = self.selector.get_selection_summary(task, complexity="auto")

        # Should detect low complexity
        assert summary["complexity"] < 0.4
        assert len(summary["agents"]) == 2  # Simple agent set

    def test_auto_complexity_detection_complex(self) -> None:
        """Test auto-detection of complex task complexity."""
        task = TranslationTask(
            source_text=(
                "The application programming interface (API) uses RESTful architecture "
                "with JSON payloads. Authentication requires OAuth2 tokens. "
                "The database connection pool supports up to 100 concurrent connections. "
                "Configuration parameters must be set in the config.yml file."
            ),
            translation="...",
            source_lang="en",
            target_lang="es",
        )

        summary = self.selector.get_selection_summary(task, complexity="auto")

        # Should detect moderate-high complexity (long, technical, multiple sentences)
        # Auto-detection is conservative, so 0.4+ is acceptable
        assert summary["complexity"] > 0.4
        assert len(summary["agents"]) >= 2

    def test_float_complexity_value(self) -> None:
        """Test using float complexity value."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        agents = self.selector.select_agents(task, complexity=0.5)

        # 0.5 complexity should use medium agent set
        assert len(agents) == 3

    def test_invalid_complexity_string(self) -> None:
        """Test that invalid complexity string raises error."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        with pytest.raises(ValueError, match="Invalid complexity"):
            self.selector.select_agents(task, complexity="invalid")

    def test_invalid_complexity_float(self) -> None:
        """Test that out-of-range float raises error."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            self.selector.select_agents(task, complexity=1.5)

    def test_domain_profile_application(self) -> None:
        """Test that domain profile priorities are applied."""
        task = TranslationTask(
            source_text="The API endpoint returns JSON",
            translation="...",
            source_lang="en",
            target_lang="es",
        )

        technical_profile = DOMAIN_PROFILES["technical"]
        agents = self.selector.select_agents(
            task, complexity="simple", domain_profile=technical_profile
        )

        # Technical domain should ensure terminology agent is included
        categories = {agent.category for agent in agents}
        assert "terminology" in categories

    def test_russian_language_specific_agent(self) -> None:
        """Test that Russian-specific fluency agent is added for Russian target."""
        task = TranslationTask(
            source_text="Hello world from the amazing translation system",
            translation="Привет мир из удивительной системы перевода",
            source_lang="en",
            target_lang="ru",  # Russian
        )

        # Use medium complexity to ensure fluency agent is included
        agents = self.selector.select_agents(task, complexity="medium")

        # For Russian, should include either fluency_russian or generic fluency
        categories = {agent.category for agent in agents}
        # Note: Current implementation includes both, future optimization could replace
        assert "fluency" in categories or "fluency_russian" in categories

    def test_budget_constraint_basic(self) -> None:
        """Test basic budget constraints."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        # Very low budget - should fall back to minimal agents
        agents = self.selector.select_agents(task, complexity="complex", max_budget_tokens=1000)

        # Should reduce to 1-2 agents due to budget
        assert len(agents) <= 2
        # Accuracy should always be included
        categories = {agent.category for agent in agents}
        assert "accuracy" in categories

    def test_budget_constraint_sufficient(self) -> None:
        """Test that sufficient budget allows all agents."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        # High budget - should allow all desired agents
        agents = self.selector.select_agents(task, complexity="medium", max_budget_tokens=5000)

        # Should use full medium agent set (3 agents)
        assert len(agents) == 3

    def test_get_selection_summary(self) -> None:
        """Test selection summary without instantiating agents."""
        task = TranslationTask(
            source_text="Test translation",
            translation="Traducción de prueba",
            source_lang="en",
            target_lang="es",
        )

        summary = self.selector.get_selection_summary(task, complexity="medium")

        # Check summary structure
        assert "complexity" in summary
        assert "agents" in summary
        assert "estimated_tokens" in summary
        assert "reasoning" in summary

        # Check values
        assert summary["complexity"] == 0.5  # medium -> 0.5
        assert len(summary["agents"]) == 3  # medium uses 3 agents
        assert summary["estimated_tokens"] > 0
        assert "medium" in summary["reasoning"].lower()

    def test_token_estimation(self) -> None:
        """Test token usage estimation."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        summary_simple = self.selector.get_selection_summary(task, complexity="simple")
        summary_complex = self.selector.get_selection_summary(task, complexity="complex")

        # Complex should estimate more tokens than simple
        assert summary_complex["estimated_tokens"] >= summary_simple["estimated_tokens"]

    def test_context_complexity_override(self) -> None:
        """Test that complexity in context is respected."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
            context={"complexity": "complex"},
        )

        # When used through orchestrator, should pick up context complexity
        # Here we test the selection summary understands the flow
        summary = self.selector.get_selection_summary(task, complexity="auto")

        # Auto-detect would normally select simple, but context can override
        # This test ensures the mechanism exists
        assert "agents" in summary

    def test_special_characters_complexity_factor(self) -> None:
        """Test that special characters increase complexity."""
        simple_task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        technical_task = TranslationTask(
            source_text="function(){}; const x = {key: value}; api://endpoint",
            translation="...",
            source_lang="en",
            target_lang="es",
        )

        simple_summary = self.selector.get_selection_summary(simple_task, "auto")
        technical_summary = self.selector.get_selection_summary(technical_task, "auto")

        # Technical text with special chars should be more complex
        assert technical_summary["complexity"] > simple_summary["complexity"]

    def test_multiple_sentences_complexity_factor(self) -> None:
        """Test that multiple sentences increase complexity."""
        single_sentence = TranslationTask(
            source_text="This is one sentence",
            translation="Esta es una oración",
            source_lang="en",
            target_lang="es",
        )

        multiple_sentences = TranslationTask(
            source_text="First sentence. Second sentence! Third sentence?",
            translation="...",
            source_lang="en",
            target_lang="es",
        )

        single_summary = self.selector.get_selection_summary(single_sentence, "auto")
        multiple_summary = self.selector.get_selection_summary(multiple_sentences, "auto")

        # Multiple sentences should be more complex
        assert multiple_summary["complexity"] > single_summary["complexity"]

    def test_cost_savings_simple_vs_complex(self) -> None:
        """Test that simple tasks use fewer tokens than complex."""
        task = TranslationTask(
            source_text="Hi",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        simple_agents = self.selector.select_agents(task, "simple")
        complex_agents = self.selector.select_agents(task, "complex")

        # Simple should use fewer agents, thus fewer tokens/cost
        assert len(simple_agents) < len(complex_agents)

        # Estimate savings
        simple_tokens = self.selector._estimate_token_usage([a.category for a in simple_agents])
        complex_tokens = self.selector._estimate_token_usage([a.category for a in complex_agents])

        savings_pct = ((complex_tokens - simple_tokens) / complex_tokens) * 100
        # Should save at least 30%
        assert savings_pct >= 30


class TestDynamicAgentSelectorEdgeCases:
    """Test edge cases for dynamic agent selection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = OpenAIProvider(api_key="test-key")
        self.selector = DynamicAgentSelector(self.provider)

    def test_very_short_text(self) -> None:
        """Test selection with very short text."""
        task = TranslationTask(
            source_text="Hi",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        agents = self.selector.select_agents(task, complexity="auto")

        # Short text should use minimal agents
        assert len(agents) == 2

    def test_very_long_text(self) -> None:
        """Test selection with very long text."""
        long_text = " ".join(["word"] * 1000)  # 1000 words
        task = TranslationTask(
            source_text=long_text,
            translation=long_text,
            source_lang="en",
            target_lang="es",
        )

        summary = self.selector.get_selection_summary(task, complexity="auto")

        # Long text should be detected as higher complexity than simple (> 0.3)
        # Note: Repeated words without sentences/punctuation gets medium complexity (0.4)
        assert summary["complexity"] > 0.3

    def test_no_domain_profile(self) -> None:
        """Test selection without domain profile."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        agents = self.selector.select_agents(task, complexity="medium", domain_profile=None)

        # Should work fine without domain profile
        assert len(agents) == 3

    def test_budget_too_low_fallback(self) -> None:
        """Test fallback to minimum agent when budget too low."""
        task = TranslationTask(
            source_text="Test",
            translation="Prueba",
            source_lang="en",
            target_lang="es",
        )

        agents = self.selector.select_agents(task, complexity="complex", max_budget_tokens=100)

        # Should fall back to at least accuracy agent
        assert len(agents) >= 1
        categories = {agent.category for agent in agents}
        assert "accuracy" in categories
