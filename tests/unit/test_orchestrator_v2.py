"""Tests for EnhancedAgentOrchestrator."""

from unittest.mock import AsyncMock, Mock

import pytest

from kttc.agents.orchestrator_v2 import EnhancedAgentOrchestrator
from kttc.core import QAReport, TranslationTask

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    llm = Mock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def orchestrator(mock_llm):
    """Create orchestrator with mock LLM."""
    return EnhancedAgentOrchestrator(
        llm_provider=mock_llm,
        quality_threshold=95.0,
        enable_hallucination_detection=True,
        enable_context_checking=True,
    )


@pytest.fixture
def translation_task():
    """Create sample translation task."""
    return TranslationTask(
        source_text="Hello world",
        translation="Hola mundo",
        source_lang="en",
        target_lang="es",
    )


class TestEnhancedAgentOrchestrator:
    """Test EnhancedAgentOrchestrator class."""

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.quality_threshold == 95.0
        assert len(orchestrator.agents) >= 4  # Accuracy, Fluency, Terminology, Hallucination
        assert orchestrator.context_agent is not None

    def test_initialization_without_optional_agents(self, mock_llm):
        """Test initialization with optional agents disabled."""
        orch = EnhancedAgentOrchestrator(
            llm_provider=mock_llm,
            enable_hallucination_detection=False,
            enable_context_checking=False,
        )
        assert len(orch.agents) == 3  # Only core agents
        assert orch.context_agent is None

    async def test_evaluate_basic(self, orchestrator, translation_task, mock_llm):
        """Test basic evaluation without optional features."""
        # Mock all agents returning empty errors
        mock_llm.complete.return_value = '{"errors": []}'

        report = await orchestrator.evaluate(
            translation_task,
            use_neural_metrics=False,
            use_translation_memory=False,
            use_terminology_base=False,
        )

        assert isinstance(report, QAReport)
        assert report.mqm_score >= 0
        assert report.mqm_score <= 100
        assert isinstance(report.errors, list)
        assert report.status in ["pass", "fail"]

    async def test_evaluate_with_errors(self, orchestrator, translation_task, mock_llm):
        """Test evaluation that finds errors."""
        # Mock agents returning errors
        mock_llm.complete.return_value = (
            '{"errors": ['
            '{"category": "accuracy", "subcategory": "mistranslation", '
            '"severity": "major", "location": [0, 5], "description": "Test error"}'
            "]}"
        )

        report = await orchestrator.evaluate(
            translation_task,
            use_neural_metrics=False,
            use_translation_memory=False,
            use_terminology_base=False,
        )

        assert len(report.errors) > 0
        assert report.mqm_score < 100

    async def test_evaluate_with_neural_metrics(self, orchestrator, translation_task, mock_llm):
        """Test evaluation with neural metrics."""
        mock_llm.complete.return_value = '{"errors": []}'

        # Mock neural metrics
        mock_neural = Mock()
        mock_neural.evaluate = AsyncMock()
        mock_neural.evaluate.return_value = Mock(
            comet_score=0.85,
            kiwi_score=0.80,
            quality_estimate="high",
        )
        orchestrator.neural_metrics = mock_neural

        report = await orchestrator.evaluate(
            translation_task,
            use_neural_metrics=True,
            use_translation_memory=False,
            use_terminology_base=False,
        )

        assert report.comet_score == 0.85
        assert report.kiwi_score == 0.80
        assert report.neural_quality_estimate == "high"
        assert report.composite_score is not None

    async def test_evaluate_neural_metrics_failure(self, orchestrator, translation_task, mock_llm):
        """Test handling of neural metrics failure."""
        mock_llm.complete.return_value = '{"errors": []}'

        # Mock neural metrics that raises exception
        mock_neural = Mock()
        mock_neural.evaluate = AsyncMock(side_effect=Exception("Neural metrics failed"))
        orchestrator.neural_metrics = mock_neural

        report = await orchestrator.evaluate(
            translation_task,
            use_neural_metrics=True,
            use_translation_memory=False,
            use_terminology_base=False,
        )

        # Should still return a report, with neural scores as None
        assert report.comet_score is None
        assert report.kiwi_score is None

    async def test_evaluate_with_terminology_base(self, orchestrator, translation_task, mock_llm):
        """Test evaluation with terminology base validation."""
        mock_llm.complete.return_value = '{"errors": []}'

        # Mock terminology base
        mock_termbase = Mock()
        mock_termbase.validate_translation = AsyncMock()
        mock_termbase.validate_translation.return_value = [
            Mock(
                source_term="Hello",
                expected_terms=["Hola"],
                found_in_translation=False,
                severity="major",
            )
        ]
        orchestrator.terminology_base = mock_termbase

        report = await orchestrator.evaluate(
            translation_task,
            use_neural_metrics=False,
            use_translation_memory=False,
            use_terminology_base=True,
        )

        assert len(report.errors) > 0
        assert any(
            e.category == "terminology" and e.subcategory == "glossary_violation"
            for e in report.errors
        )

    async def test_evaluate_with_translation_memory(self, orchestrator, translation_task, mock_llm):
        """Test evaluation with translation memory."""
        mock_llm.complete.return_value = '{"errors": []}'

        # Mock TM
        mock_tm = Mock()
        mock_tm.add_translation = AsyncMock()
        orchestrator.translation_memory = mock_tm

        report = await orchestrator.evaluate(
            translation_task,
            use_neural_metrics=False,
            use_translation_memory=True,
            use_terminology_base=False,
        )

        # High quality translations should be added to TM
        if report.mqm_score >= 90.0:
            mock_tm.add_translation.assert_called_once()

    async def test_evaluate_tm_add_failure(self, orchestrator, translation_task, mock_llm):
        """Test handling of TM add failure."""
        mock_llm.complete.return_value = '{"errors": []}'

        # Mock TM that fails on add
        mock_tm = Mock()
        mock_tm.add_translation = AsyncMock(side_effect=Exception("TM failed"))
        orchestrator.translation_memory = mock_tm

        # Should not raise exception
        report = await orchestrator.evaluate(
            translation_task,
            use_neural_metrics=False,
            use_translation_memory=True,
            use_terminology_base=False,
        )

        assert isinstance(report, QAReport)

    async def test_get_tm_suggestions(self, orchestrator, mock_llm):
        """Test getting TM suggestions."""
        # Mock TM
        mock_tm = Mock()
        mock_tm.search_similar = AsyncMock()
        mock_tm.search_similar.return_value = [
            Mock(similarity=0.95, segment=Mock(translation="Hola mundo"))
        ]
        orchestrator.translation_memory = mock_tm

        results = await orchestrator.get_tm_suggestions(
            source="Hello world",
            source_lang="en",
            target_lang="es",
            limit=3,
        )

        assert len(results) == 1
        assert results[0].similarity == 0.95

    async def test_get_tm_suggestions_no_tm(self, orchestrator):
        """Test getting TM suggestions when TM is not configured."""
        orchestrator.translation_memory = None

        results = await orchestrator.get_tm_suggestions(
            source="Hello world",
            source_lang="en",
            target_lang="es",
        )

        assert results == []

    async def test_get_tm_suggestions_failure(self, orchestrator):
        """Test handling of TM search failure."""
        mock_tm = Mock()
        mock_tm.search_similar = AsyncMock(side_effect=Exception("TM failed"))
        orchestrator.translation_memory = mock_tm

        results = await orchestrator.get_tm_suggestions(
            source="Hello world",
            source_lang="en",
            target_lang="es",
        )

        assert results == []

    def test_set_quality_threshold(self, orchestrator):
        """Test updating quality threshold."""
        orchestrator.set_quality_threshold(90.0)
        assert orchestrator.quality_threshold == 90.0

    def test_set_quality_threshold_invalid(self, orchestrator):
        """Test invalid quality threshold."""
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            orchestrator.set_quality_threshold(150.0)

        with pytest.raises(ValueError, match="must be between 0 and 100"):
            orchestrator.set_quality_threshold(-10.0)

    def test_set_document_context(self, orchestrator):
        """Test setting document context."""
        orchestrator.set_document_context("Full document text here")
        # Should not raise exception
        assert orchestrator.context_agent is not None

    def test_set_document_context_no_agent(self, mock_llm):
        """Test setting context when context agent disabled."""
        orch = EnhancedAgentOrchestrator(
            llm_provider=mock_llm,
            enable_context_checking=False,
        )
        # Should not raise exception even without context agent
        orch.set_document_context("Full document text")

    def test_add_segment_to_context(self, orchestrator):
        """Test adding segment to context."""
        orchestrator.add_segment_to_context(
            source="Hello",
            translation="Hola",
            segment_id="seg_1",
        )
        # Should not raise exception
        assert orchestrator.context_agent is not None

    def test_clear_context(self, orchestrator):
        """Test clearing context."""
        orchestrator.add_segment_to_context("Hello", "Hola")
        orchestrator.clear_context()
        # Should not raise exception

    def test_calculate_composite_score_all_metrics(self, orchestrator):
        """Test composite score with all metrics available."""
        score = orchestrator._calculate_composite_score(
            mqm_score=90.0,
            comet_score=0.85,  # 85%
            kiwi_score=0.80,  # 80%
        )
        # Weighted: 40% * 90 + 30% * 85 + 30% * 80 = 36 + 25.5 + 24 = 85.5
        assert score == pytest.approx(85.5, abs=0.1)

    def test_calculate_composite_score_mqm_only(self, orchestrator):
        """Test composite score with MQM only."""
        score = orchestrator._calculate_composite_score(
            mqm_score=90.0,
            comet_score=None,
            kiwi_score=None,
        )
        assert score == 90.0

    def test_calculate_composite_score_mqm_and_comet(self, orchestrator):
        """Test composite score with MQM and COMET."""
        score = orchestrator._calculate_composite_score(
            mqm_score=90.0,
            comet_score=0.80,
            kiwi_score=None,
        )
        # Weighted: (40% * 90 + 30% * 80) / 70% = (36 + 24) / 0.7 = 85.71
        assert score == pytest.approx(85.71, abs=0.1)
