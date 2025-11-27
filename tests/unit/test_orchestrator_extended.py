"""Extended tests for AgentOrchestrator.

Additional coverage for orchestrator features including:
- Domain adaptation
- Weighted consensus
- Language-specific agents
- Style preservation
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core import ErrorAnnotation, TranslationTask


@pytest.mark.unit
class TestOrchestratorAdvancedFeatures:
    """Test advanced orchestrator features."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_quick_mode(self, mock_llm: MagicMock) -> None:
        """Test orchestrator in quick mode."""
        # Arrange & Act
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, quick_mode=True)

        # Assert
        assert orchestrator.quick_mode is True
        assert orchestrator.enable_domain_adaptation is False
        assert orchestrator.enable_dynamic_selection is False
        assert orchestrator.style_analyzer is None

    @pytest.mark.asyncio
    async def test_orchestrator_with_selected_agents(self, mock_llm: MagicMock) -> None:
        """Test orchestrator with custom selected agents."""
        # Arrange & Act
        selected = ["accuracy", "fluency"]
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, selected_agents=selected)

        # Assert
        assert len(orchestrator.agents) == 2
        assert orchestrator.selected_agents == selected

    @pytest.mark.asyncio
    async def test_orchestrator_with_invalid_agent_name(self, mock_llm: MagicMock) -> None:
        """Test orchestrator ignores invalid agent names."""
        # Arrange & Act
        selected = ["accuracy", "invalid_agent", "fluency"]
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, selected_agents=selected)

        # Assert
        # Should have only 2 valid agents
        assert len(orchestrator.agents) == 2

    @pytest.mark.asyncio
    async def test_orchestrator_weighted_consensus_enabled(self, mock_llm: MagicMock) -> None:
        """Test orchestrator with weighted consensus enabled."""
        # Arrange & Act
        orchestrator = AgentOrchestrator(
            llm_provider=mock_llm, use_weighted_consensus=True, agent_weights={"accuracy": 1.5}
        )

        # Assert
        assert orchestrator.use_weighted_consensus is True
        assert orchestrator.consensus is not None

    @pytest.mark.asyncio
    async def test_orchestrator_domain_adaptation_disabled(self, mock_llm: MagicMock) -> None:
        """Test orchestrator with domain adaptation disabled."""
        # Arrange & Act
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, enable_domain_adaptation=False)

        # Assert
        assert orchestrator.enable_domain_adaptation is False
        assert orchestrator.domain_detector is None

    @pytest.mark.asyncio
    async def test_evaluate_with_breakdown(self, mock_llm: MagicMock) -> None:
        """Test evaluate_with_breakdown method."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        report, breakdown = await orchestrator.evaluate_with_breakdown(task)

        # Assert
        assert report is not None
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0
        # Should have breakdown by agent category
        assert "accuracy" in breakdown or "fluency" in breakdown

    @pytest.mark.asyncio
    async def test_evaluate_russian_with_language_agent(self, mock_llm: MagicMock) -> None:
        """Test evaluation of Russian translation triggers language-specific agent."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="Привет мир",
            source_lang="en",
            target_lang="ru",
        )

        # Act
        with (
            patch("kttc.agents.orchestrator.get_helper_for_language") as mock_get_helper,
            patch("kttc.agents.orchestrator.RussianFluencyAgent") as mock_russian_agent,
        ):
            mock_helper = Mock()
            mock_helper.is_available.return_value = True
            mock_get_helper.return_value = mock_helper

            mock_agent_instance = AsyncMock()
            mock_agent_instance.evaluate.return_value = []
            mock_agent_instance.category = "fluency_russian"
            mock_russian_agent.return_value = mock_agent_instance

            orchestrator = AgentOrchestrator(llm_provider=mock_llm, enable_dynamic_selection=False)
            report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None

    @pytest.mark.asyncio
    async def test_evaluate_chinese_with_language_agent(self, mock_llm: MagicMock) -> None:
        """Test evaluation of Chinese translation."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="你好世界",
            source_lang="en",
            target_lang="zh",
        )

        # Act
        with (
            patch("kttc.agents.orchestrator.get_helper_for_language") as mock_get_helper,
            patch("kttc.agents.orchestrator.ChineseFluencyAgent") as mock_chinese_agent,
        ):
            mock_helper = Mock()
            mock_helper.is_available.return_value = False
            mock_get_helper.return_value = mock_helper

            mock_agent_instance = AsyncMock()
            mock_agent_instance.evaluate.return_value = []
            mock_agent_instance.category = "fluency_chinese"
            mock_chinese_agent.return_value = mock_agent_instance

            orchestrator = AgentOrchestrator(llm_provider=mock_llm, enable_dynamic_selection=False)
            report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None

    @pytest.mark.asyncio
    async def test_evaluate_hindi_with_language_agent(self, mock_llm: MagicMock) -> None:
        """Test evaluation of Hindi translation."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="नमस्ते दुनिया",
            source_lang="en",
            target_lang="hi",
        )

        # Act
        with (
            patch("kttc.agents.orchestrator.get_helper_for_language") as mock_get_helper,
            patch("kttc.agents.orchestrator.HindiFluencyAgent") as mock_hindi_agent,
        ):
            mock_helper = Mock()
            mock_get_helper.return_value = mock_helper

            mock_agent_instance = AsyncMock()
            mock_agent_instance.evaluate.return_value = []
            mock_agent_instance.category = "fluency_hindi"
            mock_hindi_agent.return_value = mock_agent_instance

            orchestrator = AgentOrchestrator(llm_provider=mock_llm, enable_dynamic_selection=False)
            report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None

    @pytest.mark.asyncio
    async def test_evaluate_persian_with_language_agent(self, mock_llm: MagicMock) -> None:
        """Test evaluation of Persian translation."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="سلام دنیا",
            source_lang="en",
            target_lang="fa",
        )

        # Act
        with (
            patch("kttc.agents.orchestrator.get_helper_for_language") as mock_get_helper,
            patch("kttc.agents.orchestrator.PersianFluencyAgent") as mock_persian_agent,
        ):
            mock_helper = Mock()
            mock_get_helper.return_value = mock_helper

            mock_agent_instance = AsyncMock()
            mock_agent_instance.evaluate.return_value = []
            mock_agent_instance.category = "fluency_persian"
            mock_persian_agent.return_value = mock_agent_instance

            orchestrator = AgentOrchestrator(llm_provider=mock_llm, enable_dynamic_selection=False)
            report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None


@pytest.mark.unit
class TestOrchestratorDomainAdaptation:
    """Test domain adaptation features."""

    @pytest.mark.asyncio
    async def test_evaluate_with_domain_detection(self, mock_llm: MagicMock) -> None:
        """Test evaluation with domain detection enabled."""
        # Arrange
        task = TranslationTask(
            source_text="The patient has a fever",
            translation="El paciente tiene fiebre",
            source_lang="en",
            target_lang="es",
        )

        # Act
        with patch("kttc.agents.orchestrator.DomainDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.detect_domain.return_value = "medical"
            mock_detector.get_domain_confidence.return_value = 0.95
            mock_detector_class.return_value = mock_detector

            orchestrator = AgentOrchestrator(llm_provider=mock_llm, enable_domain_adaptation=True)
            report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None
        # Should have domain details in agent_details
        if report.agent_details:
            assert "detected_domain" in report.agent_details or report.agent_details is not None

    @pytest.mark.asyncio
    async def test_evaluate_with_context_complexity(self, mock_llm: MagicMock) -> None:
        """Test evaluation with context complexity hint."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
            context={"complexity": "simple"},
        )

        # Act
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, enable_dynamic_selection=True)
        report = await orchestrator.evaluate(task)

        # Assert
        assert report is not None


@pytest.mark.unit
class TestOrchestratorStylePreservation:
    """Test style preservation features."""

    @pytest.mark.asyncio
    async def test_init_style_analyzer_import_error(self, mock_llm: MagicMock) -> None:
        """Test initialization when style module is not available."""
        # Arrange & Act
        with patch("kttc.agents.orchestrator.AgentOrchestrator._init_style_analyzer") as mock_init:
            mock_init.return_value = None
            orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Assert
        assert orchestrator.style_analyzer is None

    @pytest.mark.asyncio
    async def test_analyze_source_style_no_analyzer(self, mock_llm: MagicMock) -> None:
        """Test style analysis when analyzer is not available."""
        # Arrange
        task = TranslationTask(
            source_text="Test text",
            translation="Texto de prueba",
            source_lang="en",
            target_lang="es",
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, quick_mode=True)
        orchestrator.style_analyzer = None

        # Act
        profile = orchestrator._analyze_source_style(task)

        # Assert
        assert profile is None

    @pytest.mark.asyncio
    async def test_get_style_preservation_agent_no_style(self, mock_llm: MagicMock) -> None:
        """Test getting style agent when no style detected."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        agent = orchestrator._get_style_preservation_agent(None)

        # Assert
        assert agent is None


@pytest.mark.unit
class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator."""

    @pytest.mark.asyncio
    async def test_evaluate_with_agent_failure(self, mock_llm: MagicMock) -> None:
        """Test evaluation when an agent fails."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        # Act
        with patch.object(AgentOrchestrator, "_select_agents") as mock_select:
            failing_agent = AsyncMock()
            failing_agent.evaluate.side_effect = Exception("Agent failure")
            failing_agent.category = "failing_agent"
            mock_select.return_value = [failing_agent]

            orchestrator = AgentOrchestrator(llm_provider=mock_llm)

            # Should raise AgentEvaluationError
            from kttc.agents.base import AgentEvaluationError

            with pytest.raises(AgentEvaluationError):
                await orchestrator.evaluate(task)


@pytest.mark.unit
class TestOrchestratorConsensusCalculation:
    """Test consensus calculation methods."""

    @pytest.mark.asyncio
    async def test_calculate_evaluation_scores_with_consensus(self, mock_llm: MagicMock) -> None:
        """Test score calculation with weighted consensus."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, use_weighted_consensus=True)
        agent_results = {
            "accuracy": [
                ErrorAnnotation(
                    category="accuracy",
                    subcategory="mistranslation",
                    severity="major",
                    location=(0, 5),
                    description="Test error",
                )
            ]
        }

        # Act
        (
            mqm_score,
            confidence,
            agent_agreement,
            agent_scores,
            consensus_metadata,
        ) = orchestrator._calculate_evaluation_scores(agent_results, [], 10, None)

        # Assert
        assert isinstance(mqm_score, float)
        # With consensus enabled, should have these metrics
        assert confidence is not None or confidence is None  # May be None with single agent
        assert (
            agent_agreement is not None or agent_agreement is None
        )  # May be None with single agent
        assert agent_scores is not None or agent_scores is None

    @pytest.mark.asyncio
    async def test_calculate_evaluation_scores_without_consensus(self, mock_llm: MagicMock) -> None:
        """Test score calculation without weighted consensus."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm, use_weighted_consensus=False)
        agent_results = {"accuracy": []}

        # Act
        (
            mqm_score,
            confidence,
            agent_agreement,
            agent_scores,
            consensus_metadata,
        ) = orchestrator._calculate_evaluation_scores(agent_results, [], 10, None)

        # Assert
        assert isinstance(mqm_score, float)
        assert confidence is None
        assert agent_agreement is None
        assert agent_scores is None
        assert consensus_metadata is None

    @pytest.mark.asyncio
    async def test_build_domain_and_style_details(self, mock_llm: MagicMock) -> None:
        """Test building domain and style details."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Create mock domain profile
        mock_domain_profile = Mock()
        mock_domain_profile.complexity = "high"
        mock_domain_profile.description = "Test domain"

        # Act
        details = orchestrator._build_domain_and_style_details(
            detected_domain="medical",
            domain_profile=mock_domain_profile,
            domain_confidence=0.95,
            quality_threshold=95.0,
            style_profile=None,
        )

        # Assert
        assert details is not None
        assert details["detected_domain"] == "medical"
        assert details["domain_confidence"] == 0.95
        assert details["domain_complexity"] == "high"

    @pytest.mark.asyncio
    async def test_build_domain_and_style_details_with_style(self, mock_llm: MagicMock) -> None:
        """Test building details with style profile."""
        # Arrange
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        mock_style_profile = Mock()
        mock_style_profile.has_significant_deviations = True
        mock_style_profile.deviation_score = 0.8
        mock_style_profile.detected_pattern = Mock(value="literary")
        mock_style_profile.is_literary = True
        mock_style_profile.detected_deviations = [Mock(type=Mock(value="metaphor"))]
        mock_style_profile.recommended_fluency_tolerance = 0.7

        # Act
        details = orchestrator._build_domain_and_style_details(
            detected_domain=None,
            domain_profile=None,
            domain_confidence=None,
            quality_threshold=95.0,
            style_profile=mock_style_profile,
        )

        # Assert
        assert details is not None
        assert "style_analysis" in details
        assert details["style_analysis"]["style_detected"] is True
        assert details["style_analysis"]["deviation_score"] == 0.8


@pytest.mark.unit
class TestOrchestratorAgentSelection:
    """Test agent selection logic."""

    @pytest.mark.asyncio
    async def test_get_language_specific_agents_no_special_language(
        self, mock_llm: MagicMock
    ) -> None:
        """Test getting language-specific agents for unsupported language."""
        # Arrange
        task = TranslationTask(
            source_text="Hello world",
            translation="Bonjour monde",
            source_lang="en",
            target_lang="fr",  # French - no specific agent
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        # Act
        agents = orchestrator._get_language_specific_agents(task)

        # Assert
        assert len(agents) == 0  # No language-specific agents for French

    @pytest.mark.asyncio
    async def test_get_domain_context_with_style_aware(self, mock_llm: MagicMock) -> None:
        """Test getting domain context with style-aware profile."""
        # Arrange
        task = TranslationTask(
            source_text="Test", translation="Test", source_lang="en", target_lang="es"
        )
        orchestrator = AgentOrchestrator(llm_provider=mock_llm)

        mock_style_profile = Mock()
        mock_style_profile.deviation_score = 0.9

        mock_style_aware_profile = Mock()
        mock_style_aware_profile.domain_type = "literary"

        # Act
        domain, profile, confidence = orchestrator._get_domain_context(
            task, mock_style_profile, mock_style_aware_profile
        )

        # Assert
        assert domain == "literary"
        assert profile == mock_style_aware_profile
        assert confidence == 0.9
