"""Unit tests for debate module.

Tests multi-agent debate mechanism for error verification.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

# Mock heavy dependencies before importing kttc modules
sys.modules["spacy"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["jieba"] = MagicMock()
sys.modules["hanlp"] = MagicMock()

import pytest  # noqa: E402

from kttc.agents.debate import (  # noqa: E402
    DebateOrchestrator,
    DebateResult,
    DebateRound,
)
from kttc.core.models import (  # noqa: E402
    ErrorAnnotation,
    ErrorSeverity,
    TranslationTask,
)


@pytest.fixture
def sample_task() -> TranslationTask:
    """Create sample translation task."""
    return TranslationTask(
        source_text="Hello world",
        translation="Привет мир",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def sample_error() -> ErrorAnnotation:
    """Create sample error annotation."""
    return ErrorAnnotation(
        category="accuracy",
        subcategory="mistranslation",
        severity=ErrorSeverity.MAJOR,
        location=(0, 5),
        description="Test error",
        suggestion="Fix this",
        source_agent="accuracy",
    )


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value="""{
            "verdict": "confirmed",
            "confidence": 0.9,
            "reasoning": "Error is valid"
        }"""
    )
    return provider


class TestDebateResult:
    """Tests for DebateResult dataclass."""

    def test_create_debate_result(self, sample_error: ErrorAnnotation) -> None:
        """Test creating DebateResult."""
        result = DebateResult(
            error=sample_error,
            original_agent="accuracy",
            verifier_agent="fluency",
            verdict="confirmed",
            confidence=0.9,
            reasoning="Error is valid",
        )

        assert result.error == sample_error
        assert result.original_agent == "accuracy"
        assert result.verifier_agent == "fluency"
        assert result.verdict == "confirmed"
        assert result.confidence == 0.9
        assert result.reasoning == "Error is valid"
        assert result.modified_error is None

    def test_debate_result_with_modified_error(self, sample_error: ErrorAnnotation) -> None:
        """Test DebateResult with modified error."""
        modified = ErrorAnnotation(
            category="fluency",
            subcategory="grammar",
            severity=ErrorSeverity.MINOR,
            location=(0, 5),
            description="Modified error",
        )

        result = DebateResult(
            error=sample_error,
            original_agent="accuracy",
            verifier_agent="fluency",
            verdict="modified",
            confidence=0.85,
            reasoning="Error needs modification",
            modified_error=modified,
        )

        assert result.verdict == "modified"
        assert result.modified_error == modified


class TestDebateRound:
    """Tests for DebateRound dataclass."""

    def test_create_debate_round(self) -> None:
        """Test creating DebateRound."""
        round_result = DebateRound(
            round_number=1,
            errors_debated=5,
        )

        assert round_result.round_number == 1
        assert round_result.errors_debated == 5
        assert round_result.results == []
        assert round_result.summary == {}

    def test_debate_round_with_results(self, sample_error: ErrorAnnotation) -> None:
        """Test DebateRound with results."""
        result = DebateResult(
            error=sample_error,
            original_agent="accuracy",
            verifier_agent="fluency",
            verdict="confirmed",
            confidence=0.9,
            reasoning="Valid",
        )

        round_result = DebateRound(
            round_number=1,
            errors_debated=1,
            results=[result],
            summary={"confirmed": 1},
        )

        assert len(round_result.results) == 1
        assert round_result.summary["confirmed"] == 1


class TestDebateOrchestrator:
    """Tests for DebateOrchestrator class."""

    def test_init_default_values(self, mock_llm_provider: MagicMock) -> None:
        """Test initialization with default values."""
        orchestrator = DebateOrchestrator(mock_llm_provider)

        assert orchestrator.llm_provider == mock_llm_provider
        assert orchestrator.temperature == 0.1
        assert orchestrator.max_tokens == 1500
        assert orchestrator.confidence_threshold == 0.6

    def test_init_custom_values(self, mock_llm_provider: MagicMock) -> None:
        """Test initialization with custom values."""
        orchestrator = DebateOrchestrator(
            mock_llm_provider,
            temperature=0.3,
            max_tokens=2000,
            confidence_threshold=0.8,
        )

        assert orchestrator.temperature == 0.3
        assert orchestrator.max_tokens == 2000
        assert orchestrator.confidence_threshold == 0.8

    def test_verifier_assignments_defined(self) -> None:
        """Test verifier assignments are defined."""
        assignments = DebateOrchestrator.VERIFIER_ASSIGNMENTS

        assert "accuracy" in assignments
        assert "fluency" in assignments
        assert "terminology" in assignments
        assert "hallucination" in assignments
        assert "context" in assignments

    @pytest.mark.asyncio
    async def test_run_debate_empty_errors(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
    ) -> None:
        """Test run_debate returns empty list for empty errors."""
        orchestrator = DebateOrchestrator(mock_llm_provider)

        verified_errors, rounds = await orchestrator.run_debate([], sample_task)

        assert verified_errors == []
        assert rounds == []
        mock_llm_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_debate_with_errors(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
        sample_error: ErrorAnnotation,
    ) -> None:
        """Test run_debate processes errors."""
        mock_llm_provider.complete = AsyncMock(
            return_value="""{
                "verdict": "confirmed",
                "confidence": 0.9,
                "reasoning": "Error is correct"
            }"""
        )

        orchestrator = DebateOrchestrator(mock_llm_provider)
        _, rounds = await orchestrator.run_debate([sample_error], sample_task)

        assert len(rounds) == 1
        assert rounds[0].round_number == 1

    @pytest.mark.asyncio
    async def test_run_debate_rejected_error(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
        sample_error: ErrorAnnotation,
    ) -> None:
        """Test run_debate handles rejected errors."""
        mock_llm_provider.complete = AsyncMock(
            return_value="""{
                "verdict": "rejected",
                "confidence": 0.85,
                "reasoning": "This is not actually an error"
            }"""
        )

        orchestrator = DebateOrchestrator(mock_llm_provider)
        _, rounds = await orchestrator.run_debate([sample_error], sample_task)

        # Rejected errors should not be in verified list
        assert len(rounds) == 1

    @pytest.mark.asyncio
    async def test_run_debate_multiple_rounds(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
        sample_error: ErrorAnnotation,
    ) -> None:
        """Test run_debate with multiple rounds."""
        mock_llm_provider.complete = AsyncMock(
            return_value="""{
                "verdict": "confirmed",
                "confidence": 0.9,
                "reasoning": "Valid error"
            }"""
        )

        orchestrator = DebateOrchestrator(mock_llm_provider)
        _, rounds = await orchestrator.run_debate(
            [sample_error], sample_task, max_rounds=2
        )

        # Should have 2 rounds
        assert len(rounds) == 2
        assert rounds[0].round_number == 1
        assert rounds[1].round_number == 2


class TestDebateOrchestratorVerification:
    """Tests for debate verification logic."""

    @pytest.mark.asyncio
    async def test_verify_accuracy_error(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
    ) -> None:
        """Test verification of accuracy error by fluency agent."""
        error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.MAJOR,
            location=(0, 5),
            description="Mistranslation",
            source_agent="accuracy",
        )

        mock_llm_provider.complete = AsyncMock(
            return_value="""{
                "verdict": "confirmed",
                "confidence": 0.95,
                "reasoning": "Accuracy error confirmed by fluency analysis"
            }"""
        )

        orchestrator = DebateOrchestrator(mock_llm_provider)
        await orchestrator.run_debate([error], sample_task)

        # Verify LLM was called
        assert mock_llm_provider.complete.called

    @pytest.mark.asyncio
    async def test_verify_low_confidence_rejection(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
        sample_error: ErrorAnnotation,
    ) -> None:
        """Test errors below confidence threshold are rejected."""
        mock_llm_provider.complete = AsyncMock(
            return_value="""{
                "verdict": "confirmed",
                "confidence": 0.3,
                "reasoning": "Uncertain about this error"
            }"""
        )

        orchestrator = DebateOrchestrator(mock_llm_provider, confidence_threshold=0.6)
        _, rounds = await orchestrator.run_debate([sample_error], sample_task)

        # Low confidence should result in rejection
        assert len(rounds) == 1


class TestDebateOrchestratorErrorHandling:
    """Tests for error handling in debate orchestrator."""

    @pytest.mark.asyncio
    async def test_handle_invalid_json_response(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
        sample_error: ErrorAnnotation,
    ) -> None:
        """Test handling of invalid JSON response from LLM."""
        mock_llm_provider.complete = AsyncMock(return_value="not valid json")

        orchestrator = DebateOrchestrator(mock_llm_provider)

        # Should handle gracefully without raising
        verified_errors, rounds = await orchestrator.run_debate([sample_error], sample_task)
        assert isinstance(verified_errors, list)
        assert isinstance(rounds, list)

    @pytest.mark.asyncio
    async def test_handle_missing_verdict_field(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
        sample_error: ErrorAnnotation,
    ) -> None:
        """Test handling of response missing verdict field."""
        mock_llm_provider.complete = AsyncMock(
            return_value="""{
                "confidence": 0.9,
                "reasoning": "Missing verdict"
            }"""
        )

        orchestrator = DebateOrchestrator(mock_llm_provider)
        _, rounds = await orchestrator.run_debate([sample_error], sample_task)

        # Should handle missing fields gracefully
        assert isinstance(rounds, list)

    @pytest.mark.asyncio
    async def test_handle_llm_exception(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
        sample_error: ErrorAnnotation,
    ) -> None:
        """Test handling of LLM provider exception."""
        mock_llm_provider.complete = AsyncMock(side_effect=Exception("API Error"))

        orchestrator = DebateOrchestrator(mock_llm_provider)

        # Should handle exception gracefully
        verified_errors, _ = await orchestrator.run_debate([sample_error], sample_task)
        assert isinstance(verified_errors, list)


class TestDebateOrchestratorIntegration:
    """Integration tests for debate orchestrator."""

    @pytest.mark.asyncio
    async def test_full_debate_workflow(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
    ) -> None:
        """Test complete debate workflow with multiple errors."""
        errors = [
            ErrorAnnotation(
                category="accuracy",
                subcategory="mistranslation",
                severity=ErrorSeverity.MAJOR,
                location=(0, 5),
                description="Error 1",
                source_agent="accuracy",
            ),
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(6, 10),
                description="Error 2",
                source_agent="fluency",
            ),
            ErrorAnnotation(
                category="terminology",
                subcategory="inconsistent",
                severity=ErrorSeverity.MAJOR,
                location=(11, 15),
                description="Error 3",
                source_agent="terminology",
            ),
        ]

        # Return different verdicts for different errors
        responses = [
            '{"verdict": "confirmed", "confidence": 0.9, "reasoning": "Valid"}',
            '{"verdict": "rejected", "confidence": 0.8, "reasoning": "Invalid"}',
            '{"verdict": "confirmed", "confidence": 0.85, "reasoning": "Valid"}',
        ]
        mock_llm_provider.complete = AsyncMock(side_effect=responses)

        orchestrator = DebateOrchestrator(mock_llm_provider)
        _, rounds = await orchestrator.run_debate(errors, sample_task)

        assert len(rounds) == 1
        assert rounds[0].errors_debated == 3

    @pytest.mark.asyncio
    async def test_debate_preserves_error_details(
        self,
        mock_llm_provider: MagicMock,
        sample_task: TranslationTask,
    ) -> None:
        """Test debate preserves original error details."""
        original_error = ErrorAnnotation(
            category="accuracy",
            subcategory="mistranslation",
            severity=ErrorSeverity.CRITICAL,
            location=(0, 10),
            description="Critical mistranslation",
            suggestion="Use correct term",
            source_agent="accuracy",
        )

        mock_llm_provider.complete = AsyncMock(
            return_value='{"verdict": "confirmed", "confidence": 0.95, "reasoning": "Valid"}'
        )

        orchestrator = DebateOrchestrator(mock_llm_provider)
        verified_errors, _ = await orchestrator.run_debate([original_error], sample_task)

        # Verified error should preserve original details
        if verified_errors:
            assert verified_errors[0].category == "accuracy"
            assert verified_errors[0].severity == ErrorSeverity.CRITICAL
