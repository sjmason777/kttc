"""Shared pytest fixtures for KTTC tests.

Provides mock LLM providers, test data, and common utilities.
"""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kttc.core.models import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask
from kttc.llm.base import BaseLLMProvider

# ============================================================================
# Mock LLM Provider Fixtures
# ============================================================================


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing without real API calls."""

    def __init__(self, response: str = '{"errors": []}', **kwargs: Any):
        """Initialize mock provider.

        Args:
            response: Default JSON response to return
            **kwargs: Additional arguments for customization
        """
        super().__init__()
        self.response = response
        self.call_count = 0
        self.last_prompt: str | None = None

    async def complete(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> str:
        """Return mock JSON response."""
        self.call_count += 1
        self.last_prompt = prompt
        return self.response

    async def stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Mock stream method."""
        self.call_count += 1
        self.last_prompt = prompt
        yield self.response


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Provide a basic mock LLM that returns no errors."""
    return MockLLMProvider(response='{"errors": []}')


@pytest.fixture
def mock_llm_class() -> type[MockLLMProvider]:
    """Provide MockLLMProvider class for tests that need custom instances.

    Use this fixture when you need to create MockLLMProvider with custom
    responses rather than using the default mock_llm fixture.

    Example:
        def test_with_custom_response(mock_llm_class):
            provider = mock_llm_class(response='{"errors": [...]}')
    """
    return MockLLMProvider


@pytest.fixture
def mock_llm_with_errors() -> MockLLMProvider:
    """Provide a mock LLM that returns translation errors in correct format."""
    error_response = """ERROR_START
CATEGORY: accuracy
SUBCATEGORY: mistranslation
SEVERITY: major
LOCATION: 0-5
DESCRIPTION: Test error description
SUGGESTION: Test suggestion
ERROR_END"""
    return MockLLMProvider(response=error_response)


@pytest.fixture
def mock_openai_provider(monkeypatch: pytest.MonkeyPatch) -> MockLLMProvider:
    """Mock OpenAI provider for testing."""
    mock_provider = MockLLMProvider()

    def mock_init(self: Any, api_key: str | None = None) -> None:
        self.api_key = api_key or "test-key"
        self.call_count = 0

    monkeypatch.setattr("kttc.llm.openai_provider.OpenAIProvider.__init__", mock_init)
    return mock_provider


@pytest.fixture
def mock_anthropic_provider(monkeypatch: pytest.MonkeyPatch) -> MockLLMProvider:
    """Mock Anthropic provider for testing."""
    mock_provider = MockLLMProvider()

    def mock_init(self: Any, api_key: str | None = None) -> None:
        self.api_key = api_key or "test-key"
        self.call_count = 0

    monkeypatch.setattr("kttc.llm.anthropic_provider.AnthropicProvider.__init__", mock_init)
    return mock_provider


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_translation_task() -> TranslationTask:
    """Provide a sample translation task for testing."""
    return TranslationTask(
        source_text="Hello world",
        translation="Привет мир",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def sample_translation_error() -> ErrorAnnotation:
    """Provide a sample translation error for testing."""
    return ErrorAnnotation(
        category="accuracy",
        subcategory="mistranslation",
        severity=ErrorSeverity.MAJOR,
        location=(0, 5),
        description="Incorrect translation of 'hello'",
        suggestion="Use 'hola' instead",
    )


@pytest.fixture
def sample_qa_report(
    sample_translation_task: TranslationTask, sample_translation_error: ErrorAnnotation
) -> QAReport:
    """Provide a sample QA report for testing."""
    return QAReport(
        task=sample_translation_task,
        mqm_score=85.0,
        errors=[sample_translation_error],
        status="pass",
    )


# ============================================================================
# Temporary File Fixtures
# ============================================================================


@pytest.fixture
def temp_text_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary source and translation files.

    Returns:
        Tuple of (source_file, translation_file) paths
    """
    source = tmp_path / "source.txt"
    translation = tmp_path / "translation.txt"

    source.write_text("Hello world\nHow are you?", encoding="utf-8")
    translation.write_text("Привет мир\nКак дела?", encoding="utf-8")

    return source, translation


# ============================================================================
# CLI Fixtures
# ============================================================================


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide Typer CLI test runner."""
    return CliRunner()


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest environment."""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection to auto-mark tests based on location."""
    for item in items:
        # Auto-mark based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run E2E tests (requires real API keys)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests based on markers and command line options."""
    # Skip E2E tests unless --run-e2e is specified
    if "e2e" in item.keywords and not item.config.getoption("--run-e2e"):
        pytest.skip("E2E tests skipped (use --run-e2e to run)")

    # Skip slow tests unless --run-slow is specified
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("Slow tests skipped (use --run-slow to run)")
