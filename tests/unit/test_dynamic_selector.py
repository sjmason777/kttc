"""Unit tests for dynamic agent selector module.

Tests smart agent selection for budget-aware and complexity-adaptive QA.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before importing kttc modules
sys.modules["spacy"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["jieba"] = MagicMock()
sys.modules["hanlp"] = MagicMock()

import pytest  # noqa: E402

from kttc.agents.dynamic_selector import DynamicAgentSelector  # noqa: E402
from kttc.core.models import TranslationTask  # noqa: E402


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Create mock LLM provider."""
    return MagicMock()


@pytest.fixture
def simple_task() -> TranslationTask:
    """Create simple translation task (short text)."""
    return TranslationTask(
        source_text="Hello",
        translation="Привет",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def medium_task() -> TranslationTask:
    """Create medium complexity task."""
    return TranslationTask(
        source_text="The quick brown fox jumps over the lazy dog. This is a test sentence.",
        translation="Быстрая коричневая лиса прыгает через ленивую собаку. Это тестовое предложение.",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def complex_task() -> TranslationTask:
    """Create complex technical task."""
    return TranslationTask(
        source_text="""
        The API endpoint /api/v1/users/{id} accepts GET and POST requests.
        For authentication, use Bearer token in the Authorization header.
        Response format: JSON with status code 200 for success.
        Error codes: 400 (bad request), 401 (unauthorized), 404 (not found).
        Example: curl -X GET https://api.example.com/users/123 -H "Authorization: Bearer token"
        """,
        translation="API endpoint documentation in Russian...",
        source_lang="en",
        target_lang="ru",
        context={"type": "technical", "domain": "API documentation"},
    )


class TestDynamicAgentSelectorInit:
    """Tests for DynamicAgentSelector initialization."""

    def test_init_default_values(self, mock_llm_provider: MagicMock) -> None:
        """Test initialization with default values."""
        selector = DynamicAgentSelector(mock_llm_provider)

        assert selector.llm_provider == mock_llm_provider
        assert selector.agent_temperature == 0.1
        assert selector.agent_max_tokens == 2000

    def test_init_custom_values(self, mock_llm_provider: MagicMock) -> None:
        """Test initialization with custom values."""
        selector = DynamicAgentSelector(
            mock_llm_provider,
            agent_temperature=0.3,
            agent_max_tokens=3000,
        )

        assert selector.agent_temperature == 0.3
        assert selector.agent_max_tokens == 3000

    def test_agent_token_estimates_defined(self) -> None:
        """Test that token estimates are defined for all agents."""
        estimates = DynamicAgentSelector.AGENT_TOKEN_ESTIMATES

        assert "accuracy" in estimates
        assert "fluency" in estimates
        assert "terminology" in estimates
        assert "hallucination" in estimates
        assert "context" in estimates
        assert all(isinstance(v, int) for v in estimates.values())


class TestResolveComplexity:
    """Tests for _resolve_complexity method."""

    def test_resolve_float_complexity(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test resolving float complexity value."""
        selector = DynamicAgentSelector(mock_llm_provider)

        assert selector._resolve_complexity(simple_task, 0.5) == 0.5
        assert selector._resolve_complexity(simple_task, 0.0) == 0.0
        assert selector._resolve_complexity(simple_task, 1.0) == 1.0

    def test_resolve_float_out_of_range(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test resolving invalid float complexity raises error."""
        selector = DynamicAgentSelector(mock_llm_provider)

        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            selector._resolve_complexity(simple_task, 1.5)

        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            selector._resolve_complexity(simple_task, -0.1)

    def test_resolve_simple_string(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test resolving 'simple' complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        result = selector._resolve_complexity(simple_task, "simple")
        assert result == 0.2

    def test_resolve_medium_string(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test resolving 'medium' complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        result = selector._resolve_complexity(simple_task, "medium")
        assert result == 0.5

    def test_resolve_complex_string(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test resolving 'complex' complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        result = selector._resolve_complexity(simple_task, "complex")
        assert result == 0.9

    def test_resolve_invalid_string(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test resolving invalid complexity string raises error."""
        selector = DynamicAgentSelector(mock_llm_provider)

        with pytest.raises(ValueError, match="Invalid complexity"):
            selector._resolve_complexity(simple_task, "unknown")


class TestAutoDetectComplexity:
    """Tests for _auto_detect_complexity method."""

    def test_short_text_low_complexity(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test short text gets low complexity score."""
        selector = DynamicAgentSelector(mock_llm_provider)

        complexity = selector._auto_detect_complexity(simple_task)
        assert 0.0 <= complexity <= 0.5  # Should be relatively simple

    def test_long_technical_text_high_complexity(
        self, mock_llm_provider: MagicMock, complex_task: TranslationTask
    ) -> None:
        """Test long technical text gets high complexity score."""
        selector = DynamicAgentSelector(mock_llm_provider)

        complexity = selector._auto_detect_complexity(complex_task)
        assert complexity >= 0.5  # Should be relatively complex

    def test_text_with_context_increases_complexity(self, mock_llm_provider: MagicMock) -> None:
        """Test that providing context increases complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        task_no_context = TranslationTask(
            source_text="Hello world",
            translation="Привет мир",
            source_lang="en",
            target_lang="ru",
        )

        task_with_context = TranslationTask(
            source_text="Hello world",
            translation="Привет мир",
            source_lang="en",
            target_lang="ru",
            context={"type": "formal", "domain": "business"},
        )

        complexity_no_context = selector._auto_detect_complexity(task_no_context)
        complexity_with_context = selector._auto_detect_complexity(task_with_context)

        assert complexity_with_context > complexity_no_context

    def test_technical_content_high_complexity(self, mock_llm_provider: MagicMock) -> None:
        """Test text with special characters gets higher complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        technical_task = TranslationTask(
            source_text="function() { return x[0] + y[1]; }",
            translation="функция...",
            source_lang="en",
            target_lang="ru",
        )

        complexity = selector._auto_detect_complexity(technical_task)
        # Technical content should increase complexity
        assert complexity >= 0.3


class TestGetAgentSets:
    """Tests for agent set selection methods."""

    def test_get_simple_agents(self, mock_llm_provider: MagicMock) -> None:
        """Test simple agent set."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agents = selector._get_simple_agents()

        assert "accuracy" in agents
        assert "fluency" in agents
        assert len(agents) == 2

    def test_get_medium_agents(self, mock_llm_provider: MagicMock) -> None:
        """Test medium agent set."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agents = selector._get_medium_agents()

        assert "accuracy" in agents
        assert "fluency" in agents
        assert "terminology" in agents
        assert len(agents) == 3

    def test_get_complex_agents(self, mock_llm_provider: MagicMock) -> None:
        """Test complex agent set."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agents = selector._get_complex_agents()

        assert "accuracy" in agents
        assert "fluency" in agents
        assert "terminology" in agents
        assert "hallucination" in agents
        assert "context" in agents
        assert len(agents) == 5


class TestAddLanguageSpecificAgents:
    """Tests for language-specific agent selection."""

    def test_russian_target_adds_russian_fluency(self, mock_llm_provider: MagicMock) -> None:
        """Test Russian target adds Russian fluency agent."""
        selector = DynamicAgentSelector(mock_llm_provider)

        task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        agent_set = ["accuracy", "fluency"]
        result = selector._add_language_specific_agents(agent_set, task)

        # Should replace fluency with fluency_russian
        assert "fluency_russian" in result
        assert "fluency" not in result

    def test_chinese_target_adds_chinese_fluency(self, mock_llm_provider: MagicMock) -> None:
        """Test Chinese target adds Chinese fluency agent."""
        selector = DynamicAgentSelector(mock_llm_provider)

        task = TranslationTask(
            source_text="Hello",
            translation="你好",
            source_lang="en",
            target_lang="zh",
        )

        agent_set = ["accuracy", "fluency"]
        result = selector._add_language_specific_agents(agent_set, task)

        # Should replace fluency with fluency_chinese
        assert "fluency_chinese" in result
        assert "fluency" not in result

    def test_english_target_adds_english_fluency(self, mock_llm_provider: MagicMock) -> None:
        """Test English target adds English fluency agent."""
        selector = DynamicAgentSelector(mock_llm_provider)

        task = TranslationTask(
            source_text="Привет",
            translation="Hello",
            source_lang="ru",
            target_lang="en",
        )

        agent_set = ["accuracy", "fluency"]
        result = selector._add_language_specific_agents(agent_set, task)

        assert "fluency_english" in result

    def test_hindi_target_adds_hindi_fluency(self, mock_llm_provider: MagicMock) -> None:
        """Test Hindi target adds Hindi fluency agent."""
        selector = DynamicAgentSelector(mock_llm_provider)

        task = TranslationTask(
            source_text="Hello",
            translation="नमस्ते",
            source_lang="en",
            target_lang="hi",
        )

        agent_set = ["accuracy", "fluency"]
        result = selector._add_language_specific_agents(agent_set, task)

        assert "fluency_hindi" in result

    def test_persian_target_adds_persian_fluency(self, mock_llm_provider: MagicMock) -> None:
        """Test Persian/Farsi target adds Persian fluency agent."""
        selector = DynamicAgentSelector(mock_llm_provider)

        task = TranslationTask(
            source_text="Hello",
            translation="سلام",
            source_lang="en",
            target_lang="fa",
        )

        agent_set = ["accuracy", "fluency"]
        result = selector._add_language_specific_agents(agent_set, task)

        assert "fluency_persian" in result


class TestApplyDomainPriorities:
    """Tests for domain-specific agent prioritization."""

    def test_adds_priority_agents(self, mock_llm_provider: MagicMock) -> None:
        """Test domain profile adds priority agents."""
        selector = DynamicAgentSelector(mock_llm_provider)

        mock_profile = MagicMock()
        mock_profile.priority_agents = ["hallucination", "terminology"]

        agent_set = ["accuracy", "fluency"]
        result = selector._apply_domain_priorities(agent_set, mock_profile)

        assert "hallucination" in result
        assert "terminology" in result
        assert "accuracy" in result
        assert "fluency" in result

    def test_does_not_duplicate_agents(self, mock_llm_provider: MagicMock) -> None:
        """Test domain profile doesn't add duplicate agents."""
        selector = DynamicAgentSelector(mock_llm_provider)

        mock_profile = MagicMock()
        mock_profile.priority_agents = ["accuracy", "fluency"]  # Already in set

        agent_set = ["accuracy", "fluency"]
        result = selector._apply_domain_priorities(agent_set, mock_profile)

        # Should not have duplicates
        assert result.count("accuracy") == 1
        assert result.count("fluency") == 1


class TestApplyBudgetConstraints:
    """Tests for budget-constrained agent selection."""

    def test_applies_budget_limit(self, mock_llm_provider: MagicMock) -> None:
        """Test budget limits agent selection."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agent_set = ["accuracy", "fluency", "terminology", "hallucination", "context"]

        # Very low budget should reduce agents
        result = selector._apply_budget_constraints(agent_set, max_budget_tokens=1000)

        assert len(result) <= len(agent_set)
        assert "accuracy" in result  # Core agents should remain

    def test_budget_fits_returns_full_set(self, mock_llm_provider: MagicMock) -> None:
        """Test that full set is returned when budget is sufficient."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agent_set = ["accuracy", "fluency"]

        # Large budget should return all agents
        result = selector._apply_budget_constraints(agent_set, max_budget_tokens=10000)

        assert result == agent_set

    def test_very_low_budget_returns_accuracy(self, mock_llm_provider: MagicMock) -> None:
        """Test that very low budget returns at least accuracy."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agent_set = ["accuracy", "fluency", "terminology"]

        # Budget too low for any agent should still return accuracy
        result = selector._apply_budget_constraints(agent_set, max_budget_tokens=100)

        assert "accuracy" in result


class TestEstimateTokenUsage:
    """Tests for token usage estimation."""

    def test_estimates_token_usage(self, mock_llm_provider: MagicMock) -> None:
        """Test token usage estimation."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agent_set = ["accuracy", "fluency"]
        estimate = selector._estimate_token_usage(agent_set)

        expected = (
            DynamicAgentSelector.AGENT_TOKEN_ESTIMATES["accuracy"]
            + DynamicAgentSelector.AGENT_TOKEN_ESTIMATES["fluency"]
        )
        assert estimate == expected

    def test_estimates_zero_for_empty_set(self, mock_llm_provider: MagicMock) -> None:
        """Test zero estimate for empty agent set."""
        selector = DynamicAgentSelector(mock_llm_provider)

        estimate = selector._estimate_token_usage([])
        assert estimate == 0


class TestSelectAgentsIntegration:
    """Integration tests for select_agents method."""

    def test_select_agents_simple_complexity(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test agent selection with simple complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agents = selector.select_agents(simple_task, complexity="simple")

        assert len(agents) >= 2  # At least accuracy + fluency

    def test_select_agents_complex_complexity(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test agent selection with complex complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agents = selector.select_agents(simple_task, complexity="complex")

        # Complex should have more agents than simple
        assert len(agents) >= 4

    def test_select_agents_auto_complexity(
        self, mock_llm_provider: MagicMock, medium_task: TranslationTask
    ) -> None:
        """Test agent selection with auto complexity detection."""
        selector = DynamicAgentSelector(mock_llm_provider)

        agents = selector.select_agents(medium_task, complexity="auto")

        assert len(agents) >= 2
        # Should return agent instances
        assert all(hasattr(a, "category") for a in agents)

    def test_select_agents_with_budget(
        self, mock_llm_provider: MagicMock, complex_task: TranslationTask
    ) -> None:
        """Test agent selection with budget constraint."""
        selector = DynamicAgentSelector(mock_llm_provider)

        # Without budget - more agents
        agents_unlimited = selector.select_agents(complex_task, complexity="complex")

        # With tight budget - fewer agents
        agents_limited = selector.select_agents(
            complex_task, complexity="complex", max_budget_tokens=1500
        )

        assert len(agents_limited) <= len(agents_unlimited)

    def test_select_agents_with_domain_profile(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test agent selection with domain profile."""
        selector = DynamicAgentSelector(mock_llm_provider)

        mock_profile = MagicMock()
        mock_profile.priority_agents = ["terminology"]

        agents = selector.select_agents(
            simple_task, complexity="medium", domain_profile=mock_profile
        )

        # Domain priority agent should be included
        agent_categories = [a.category for a in agents]
        assert "terminology" in agent_categories


class TestGetSelectionSummary:
    """Tests for get_selection_summary method."""

    def test_get_summary_simple(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test getting selection summary for simple task."""
        selector = DynamicAgentSelector(mock_llm_provider)

        summary = selector.get_selection_summary(simple_task, complexity="simple")

        assert "complexity" in summary
        assert "agents" in summary
        assert "estimated_tokens" in summary
        assert "reasoning" in summary
        assert summary["complexity"] == 0.2
        assert "simple" in summary["reasoning"].lower() or "core" in summary["reasoning"].lower()

    def test_get_summary_medium(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test getting selection summary for medium complexity."""
        selector = DynamicAgentSelector(mock_llm_provider)

        summary = selector.get_selection_summary(simple_task, complexity="medium")

        assert summary["complexity"] == 0.5
        assert len(summary["agents"]) >= 3
        assert (
            "medium" in summary["reasoning"].lower() or "standard" in summary["reasoning"].lower()
        )

    def test_get_summary_complex(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test getting selection summary for complex task."""
        selector = DynamicAgentSelector(mock_llm_provider)

        summary = selector.get_selection_summary(simple_task, complexity="complex")

        assert summary["complexity"] == 0.9
        assert len(summary["agents"]) >= 4
        assert "complex" in summary["reasoning"].lower() or "full" in summary["reasoning"].lower()

    def test_get_summary_with_domain_profile(
        self, mock_llm_provider: MagicMock, simple_task: TranslationTask
    ) -> None:
        """Test summary includes domain profile info."""
        selector = DynamicAgentSelector(mock_llm_provider)

        mock_profile = MagicMock()
        mock_profile.priority_agents = ["hallucination"]
        mock_profile.domain_type = "medical"

        summary = selector.get_selection_summary(
            simple_task, complexity="simple", domain_profile=mock_profile
        )

        assert "domain" in summary["reasoning"].lower()
        assert "hallucination" in summary["agents"]

    def test_get_summary_includes_language_specific(self, mock_llm_provider: MagicMock) -> None:
        """Test summary includes language-specific agents."""
        selector = DynamicAgentSelector(mock_llm_provider)

        russian_task = TranslationTask(
            source_text="Hello",
            translation="Привет",
            source_lang="en",
            target_lang="ru",
        )

        summary = selector.get_selection_summary(russian_task, complexity="simple")

        assert "fluency_russian" in summary["agents"]
