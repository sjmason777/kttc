"""Unit tests for PersianFluencyAgent.

Tests agent logic with mocked LLM and language helper.
Focus: Fast, isolated tests that find real bugs.

Philosophy: "Tests must find errors, not tests for the sake of tests!"
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add tests directory to path to import conftest
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from conftest import MockLLMProvider  # noqa: E402

from kttc.agents.fluency_persian import PersianFluencyAgent  # noqa: E402
from kttc.core.models import ErrorAnnotation, ErrorSeverity, TranslationTask  # noqa: E402
from kttc.helpers.persian import PersianLanguageHelper  # noqa: E402

# ============================================================================
# Mock PersianLanguageHelper
# ============================================================================


class MockPersianHelper:
    """Mock Persian language helper to avoid loading heavy ML models."""

    def __init__(self, available: bool = True):
        """Initialize mock helper.

        Args:
            available: Whether helper is available
        """
        self.language_code = "fa"
        self._available = available
        self.tokenize_calls = 0
        self.spell_check_calls = 0
        self.verify_position_calls = 0
        self.verify_word_calls = 0

    def is_available(self) -> bool:
        """Return availability status."""
        return self._available

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Mock tokenization."""
        self.tokenize_calls += 1
        if not text:
            return []
        # Simple mock: split by spaces
        tokens = []
        start = 0
        for word in text.split():
            end = start + len(word)
            tokens.append((word, start, end))
            start = end + 1  # +1 for space
        return tokens

    def check_spelling(self, text: str) -> list[ErrorAnnotation]:
        """Mock spell checking."""
        self.spell_check_calls += 1
        # Return empty list by default
        return []

    def verify_error_position(self, error: ErrorAnnotation, text: str) -> bool:
        """Mock error position verification."""
        self.verify_position_calls += 1
        start, end = error.location
        # Basic validation: position must be within text bounds
        return 0 <= start < len(text) and start <= end <= len(text)

    def verify_word_exists(self, description: str, text: str) -> bool:
        """Mock word existence verification."""
        self.verify_word_calls += 1
        # Simple check: any word from description exists in text
        for word in description.split():
            if word in text:
                return True
        return False


@pytest.fixture
def mock_persian_helper() -> MockPersianHelper:
    """Provide mock Persian helper."""
    return MockPersianHelper(available=True)


@pytest.fixture
def mock_persian_helper_unavailable() -> MockPersianHelper:
    """Provide unavailable mock Persian helper."""
    return MockPersianHelper(available=False)


@pytest.fixture
def sample_persian_task() -> TranslationTask:
    """Provide a sample Persian translation task."""
    return TranslationTask(
        source_text="Hello world",
        translation="سلام دنیا",
        source_lang="en",
        target_lang="fa",
    )


# ============================================================================
# Basic Tests
# ============================================================================


@pytest.mark.unit
class TestPersianFluencyAgentBasics:
    """Test basic Persian fluency agent functionality."""

    def test_instantiation_with_helper(self, mock_llm: MockLLMProvider) -> None:
        """Test that PersianFluencyAgent can be instantiated with custom helper."""
        # Arrange
        helper = MockPersianHelper()

        # Act
        agent = PersianFluencyAgent(mock_llm, helper=helper)

        # Assert
        assert agent is not None
        assert agent.helper is helper
        assert agent.category == "fluency"

    def test_instantiation_without_helper(self, mock_llm: MockLLMProvider) -> None:
        """Test that PersianFluencyAgent creates helper if none provided."""
        # Act
        agent = PersianFluencyAgent(mock_llm)

        # Assert
        assert agent is not None
        assert agent.helper is not None
        assert isinstance(agent.helper, PersianLanguageHelper)

    def test_category_property(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test agent reports correct category."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act & Assert
        assert agent.category == "fluency"

    def test_get_base_prompt_includes_persian_specifics(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that base prompt includes Persian-specific instructions."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        prompt = agent.get_base_prompt()

        # Assert
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "PERSIAN-SPECIFIC" in prompt or "Persian" in prompt or "فارسی" in prompt


# ============================================================================
# Evaluation Tests
# ============================================================================


@pytest.mark.unit
class TestPersianFluencyEvaluation:
    """Test Persian fluency evaluation logic."""

    @pytest.mark.asyncio
    async def test_evaluate_persian_task_uses_hybrid_approach(
        self,
        mock_llm: MockLLMProvider,
        mock_persian_helper: MockPersianHelper,
        sample_persian_task: TranslationTask,
    ) -> None:
        """Test that Persian tasks use hybrid DadmaTools + LLM approach."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        errors = await agent.evaluate(sample_persian_task)

        # Assert
        assert isinstance(errors, list)
        # Should have called LLM at least once (base + Persian-specific)
        assert mock_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_evaluate_non_persian_task_falls_back_to_base(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that non-Persian tasks fall back to base fluency checks."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        spanish_task = TranslationTask(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
        )

        # Act
        errors = await agent.evaluate(spanish_task)

        # Assert
        assert isinstance(errors, list)
        # Should NOT call spell checking for non-Persian
        assert mock_persian_helper.spell_check_calls == 0

    @pytest.mark.asyncio
    async def test_evaluate_merges_dadmatools_and_llm_errors(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test that evaluation merges errors from both DadmaTools and LLM."""
        # Arrange
        # Mock LLM response with one error
        # Text "سلام دنیا" is ~9 chars, so use valid position
        llm_response = """{"errors": [
            {
                "subcategory": "grammar",
                "severity": "minor",
                "location": [0, 4],
                "description": "سلام grammar issue",
                "suggestion": "Fix"
            }
        ]}"""
        mock_llm = MockLLMProvider(response=llm_response)

        # Mock helper to return one spelling error
        mock_persian_helper.check_spelling = Mock(
            return_value=[
                ErrorAnnotation(
                    category="fluency",
                    subcategory="spelling",
                    severity=ErrorSeverity.MINOR,
                    location=(5, 9),
                    description="دنیا spelling",
                )
            ]
        )

        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        errors = await agent.evaluate(sample_persian_task)

        # Assert
        # Should have both errors (if they don't overlap)
        assert len(errors) >= 1  # At least LLM error
        categories = [e.subcategory for e in errors]
        # Check that we got errors from different sources
        assert any("persian_" in cat for cat in categories)  # LLM error


# ============================================================================
# DadmaTools Check Tests
# ============================================================================


@pytest.mark.unit
class TestDadmaToolsCheck:
    """Test DadmaTools spell checking integration."""

    @pytest.mark.asyncio
    async def test_dadmatools_check_when_helper_available(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test that DadmaTools check is called when helper is available."""
        # Arrange
        mock_llm = MockLLMProvider(response='{"errors": []}')
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        await agent._dadmatools_check(sample_persian_task)

        # Assert
        assert mock_persian_helper.spell_check_calls == 1

    @pytest.mark.asyncio
    async def test_dadmatools_check_when_helper_unavailable(
        self,
        mock_persian_helper_unavailable: MockPersianHelper,
        sample_persian_task: TranslationTask,
    ) -> None:
        """Test that DadmaTools check is skipped when helper unavailable."""
        # Arrange
        mock_llm = MockLLMProvider(response='{"errors": []}')
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper_unavailable)

        # Act
        errors = await agent._dadmatools_check(sample_persian_task)

        # Assert
        assert len(errors) == 0
        assert mock_persian_helper_unavailable.spell_check_calls == 0

    @pytest.mark.asyncio
    async def test_dadmatools_check_handles_exceptions(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test that DadmaTools check handles exceptions gracefully."""
        # Arrange
        mock_llm = MockLLMProvider(response='{"errors": []}')
        mock_persian_helper.check_spelling = Mock(side_effect=Exception("DadmaTools error"))
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        errors = await agent._dadmatools_check(sample_persian_task)

        # Assert
        assert isinstance(errors, list)
        assert len(errors) == 0  # Should return empty list on error


# ============================================================================
# LLM Check Tests
# ============================================================================


@pytest.mark.unit
class TestLLMCheck:
    """Test LLM-based Persian checking."""

    @pytest.mark.asyncio
    async def test_llm_check_sends_correct_prompt(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test that LLM check sends Persian-specific prompt."""
        # Arrange
        mock_llm = MockLLMProvider(response='{"errors": []}')
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        await agent._llm_check(sample_persian_task)

        # Assert
        assert mock_llm.last_prompt is not None
        assert "Persian" in mock_llm.last_prompt or "فارسی" in mock_llm.last_prompt
        assert sample_persian_task.translation in mock_llm.last_prompt

    @pytest.mark.asyncio
    async def test_llm_check_handles_exceptions(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test that LLM check handles exceptions gracefully."""
        # Arrange
        mock_llm = Mock()
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM error"))
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        errors = await agent._llm_check(sample_persian_task)

        # Assert
        assert isinstance(errors, list)
        assert len(errors) == 0  # Should return empty list on error


# ============================================================================
# Anti-Hallucination Verification Tests
# ============================================================================


@pytest.mark.unit
class TestLLMVerification:
    """Test LLM error verification (anti-hallucination)."""

    def test_verify_llm_errors_filters_invalid_positions(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that verification filters errors with invalid positions."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        text = "سلام دنیا"  # 9 characters

        # Create error with out-of-bounds position
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 100),  # Out of bounds!
                description="test error",
            )
        ]

        # Act
        verified = agent._verify_llm_errors(errors, text)

        # Assert
        assert len(verified) == 0  # Should be filtered out
        assert mock_persian_helper.verify_position_calls == 1

    def test_verify_llm_errors_filters_hallucinated_words(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that verification filters errors mentioning nonexistent words."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        text = "سلام دنیا"

        # Create error mentioning a word NOT in text
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 4),
                description="کتاب is wrong",  # کتاب not in text!
            )
        ]

        # Act
        verified = agent._verify_llm_errors(errors, text)

        # Assert
        assert len(verified) == 0  # Should be filtered out
        assert mock_persian_helper.verify_word_calls == 1

    def test_verify_llm_errors_keeps_valid_errors(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that verification keeps valid errors."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        text = "سلام دنیا"

        # Create valid error
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 4),
                description="سلام has issue",  # Word exists in text
            )
        ]

        # Act
        verified = agent._verify_llm_errors(errors, text)

        # Assert
        assert len(verified) == 1  # Should be kept
        assert verified[0] == errors[0]

    def test_verify_llm_errors_without_helper_returns_all(
        self, mock_llm: MockLLMProvider, mock_persian_helper_unavailable: MockPersianHelper
    ) -> None:
        """Test that without helper, all errors are returned (can't verify)."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper_unavailable)
        text = "test"

        # Create error with invalid position
        errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="test",
                severity=ErrorSeverity.MINOR,
                location=(0, 100),  # Out of bounds!
                description="test",
            )
        ]

        # Act
        verified = agent._verify_llm_errors(errors, text)

        # Assert
        assert len(verified) == 1  # Can't verify without helper


# ============================================================================
# Deduplication Tests
# ============================================================================


@pytest.mark.unit
class TestErrorDeduplication:
    """Test error deduplication logic."""

    def test_remove_duplicates_filters_overlapping_errors(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that overlapping DadmaTools errors are removed."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        dadma_errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="spelling",
                severity=ErrorSeverity.MINOR,
                location=(0, 4),
                description="DadmaTools error",
            )
        ]

        llm_errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(0, 4),  # Same position!
                description="LLM error",
            )
        ]

        # Act
        unique = agent._remove_duplicates(dadma_errors, llm_errors)

        # Assert
        assert len(unique) == 0  # DadmaTools error should be removed

    def test_remove_duplicates_keeps_non_overlapping_errors(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that non-overlapping errors are kept."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        dadma_errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="spelling",
                severity=ErrorSeverity.MINOR,
                location=(0, 4),
                description="DadmaTools error",
            )
        ]

        llm_errors = [
            ErrorAnnotation(
                category="fluency",
                subcategory="grammar",
                severity=ErrorSeverity.MINOR,
                location=(5, 9),  # Different position!
                description="LLM error",
            )
        ]

        # Act
        unique = agent._remove_duplicates(dadma_errors, llm_errors)

        # Assert
        assert len(unique) == 1  # DadmaTools error should be kept

    def test_errors_overlap_detects_overlap(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that overlap detection works correctly."""
        # Arrange
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 4),
            description="test",
        )

        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(2, 6),  # Overlaps with error1
            description="test",
        )

        # Act
        overlaps = PersianFluencyAgent._errors_overlap(error1, error2)

        # Assert
        assert overlaps is True

    def test_errors_overlap_detects_no_overlap(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that non-overlapping errors are detected."""
        # Arrange
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 4),
            description="test",
        )

        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(5, 9),  # No overlap
            description="test",
        )

        # Act
        overlaps = PersianFluencyAgent._errors_overlap(error1, error2)

        # Assert
        assert overlaps is False

    def test_errors_overlap_edge_case_adjacent(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test overlap detection for adjacent (but not overlapping) errors."""
        # Arrange
        error1 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(0, 4),
            description="test",
        )

        error2 = ErrorAnnotation(
            category="fluency",
            subcategory="test",
            severity=ErrorSeverity.MINOR,
            location=(4, 8),  # Starts where error1 ends
            description="test",
        )

        # Act
        overlaps = PersianFluencyAgent._errors_overlap(error1, error2)

        # Assert
        assert overlaps is False  # Adjacent, not overlapping


# ============================================================================
# JSON Parsing Tests
# ============================================================================


@pytest.mark.unit
class TestJSONParsing:
    """Test JSON response parsing."""

    def test_parse_json_response_valid_json(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test parsing valid JSON response."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        response = '{"errors": [{"subcategory": "test"}]}'

        # Act
        parsed = agent._parse_json_response(response)

        # Assert
        assert isinstance(parsed, dict)
        assert "errors" in parsed
        assert len(parsed["errors"]) == 1

    def test_parse_json_response_json_in_markdown(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        response = '```json\n{"errors": []}\n```'

        # Act
        parsed = agent._parse_json_response(response)

        # Assert
        assert isinstance(parsed, dict)
        assert "errors" in parsed

    def test_parse_json_response_invalid_json_returns_empty(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test that invalid JSON returns empty errors."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        response = "Not valid JSON at all"

        # Act
        parsed = agent._parse_json_response(response)

        # Assert
        assert isinstance(parsed, dict)
        assert "errors" in parsed
        assert len(parsed["errors"]) == 0

    def test_parse_json_response_json_embedded_in_text(
        self, mock_llm: MockLLMProvider, mock_persian_helper: MockPersianHelper
    ) -> None:
        """Test parsing JSON embedded in text."""
        # Arrange
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)
        response = 'Here is the result: {"errors": []} and that is all'

        # Act
        parsed = agent._parse_json_response(response)

        # Assert
        assert isinstance(parsed, dict)
        assert "errors" in parsed


# ============================================================================
# Integration Tests (with mocked components)
# ============================================================================


@pytest.mark.unit
class TestPersianFluencyIntegration:
    """Integration tests verifying components work together."""

    @pytest.mark.asyncio
    async def test_full_evaluation_workflow(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test complete evaluation workflow from start to finish."""
        # Arrange
        # Mock LLM with valid error response
        llm_response = """{"errors": [
            {
                "subcategory": "grammar",
                "severity": "minor",
                "location": [0, 4],
                "description": "سلام test",
                "suggestion": "Fix it"
            }
        ]}"""
        mock_llm = MockLLMProvider(response=llm_response)
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        errors = await agent.evaluate(sample_persian_task)

        # Assert
        assert isinstance(errors, list)
        # Should have at least one error from LLM
        assert len(errors) >= 1
        # Verify error structure
        for error in errors:
            assert hasattr(error, "category")
            assert hasattr(error, "location")
            assert hasattr(error, "description")

    @pytest.mark.asyncio
    async def test_parallel_execution_error_handling(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test that parallel execution handles errors gracefully."""
        # Arrange
        # Make DadmaTools fail
        mock_persian_helper.check_spelling = Mock(side_effect=Exception("DadmaTools failed"))

        # Make LLM succeed
        mock_llm = MockLLMProvider(response='{"errors": []}')
        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        errors = await agent.evaluate(sample_persian_task)

        # Assert
        # Should still return results despite DadmaTools failure
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_verify_and_deduplicate_workflow(
        self, mock_persian_helper: MockPersianHelper, sample_persian_task: TranslationTask
    ) -> None:
        """Test that verification and deduplication work together."""
        # Arrange
        # LLM returns 3 errors: 1 valid, 1 invalid position, 1 overlapping with DadmaTools
        llm_response = """{"errors": [
            {
                "subcategory": "grammar",
                "severity": "minor",
                "location": [0, 4],
                "description": "سلام valid error",
                "suggestion": "Fix"
            },
            {
                "subcategory": "grammar",
                "severity": "minor",
                "location": [0, 1000],
                "description": "Invalid position",
                "suggestion": "Fix"
            },
            {
                "subcategory": "grammar",
                "severity": "minor",
                "location": [5, 9],
                "description": "دنیا overlaps with DadmaTools",
                "suggestion": "Fix"
            }
        ]}"""
        mock_llm = MockLLMProvider(response=llm_response)

        # DadmaTools returns 1 error at same position as LLM error #3
        mock_persian_helper.check_spelling = Mock(
            return_value=[
                ErrorAnnotation(
                    category="fluency",
                    subcategory="spelling",
                    severity=ErrorSeverity.MINOR,
                    location=(5, 9),  # Overlaps with LLM error #3
                    description="DadmaTools error",
                )
            ]
        )

        agent = PersianFluencyAgent(mock_llm, helper=mock_persian_helper)

        # Act
        errors = await agent.evaluate(sample_persian_task)

        # Assert
        # Should have:
        # - LLM error #1 (valid)
        # - LLM error #3 (overlaps with DadmaTools, so DadmaTools is removed)
        # - NOT LLM error #2 (invalid position)
        # Total: 1-2 errors (depending on base fluency checks)
        assert isinstance(errors, list)
        # At least the valid LLM error
        assert len(errors) >= 1
