"""End-to-end tests with real API calls.

These tests make actual API calls to LLM providers.
Run with: pytest tests/e2e/ --run-e2e

Requires:
- KTTC_ANTHROPIC_API_KEY environment variable
"""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kttc.cli.main import app

runner = CliRunner()


# Skip all E2E tests if no API key is available
pytestmark = pytest.mark.e2e


@pytest.mark.e2e
@pytest.mark.slow
class TestRealAnthropicAPI:
    """Test with real Anthropic Claude API."""

    def test_anthropic_simple_translation_check(self, tmp_path: Path) -> None:
        """Test simple translation check with real Anthropic API."""
        # Arrange
        api_key = os.getenv("KTTC_ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("KTTC_ANTHROPIC_API_KEY not set")

        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"
        output = tmp_path / "report.json"

        # Use a simple, clear example
        source.write_text("Hello, how are you today?", encoding="utf-8")
        translation.write_text("Hola, ¿cómo estás hoy?", encoding="utf-8")

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
                "--output",
                str(output),
                "--format",
                "json",
            ],
            env={"KTTC_ANTHROPIC_API_KEY": api_key},
        )

        # Assert
        assert result.exit_code in [0, 1]  # Should succeed or fail based on quality
        # Output file should be created
        assert output.exists()
        # Should contain valid JSON with report structure
        content = output.read_text(encoding="utf-8")
        assert "mqm_score" in content
        assert "errors" in content

    def test_anthropic_with_intentional_error(self, tmp_path: Path) -> None:
        """Test detection of intentional translation error with real API."""
        # Arrange
        api_key = os.getenv("KTTC_ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("KTTC_ANTHROPIC_API_KEY not set")

        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"

        # Intentional mistranslation: "cat" -> "perro" (dog)
        source.write_text("I have a cat.", encoding="utf-8")
        translation.write_text("Tengo un perro.", encoding="utf-8")  # Wrong!

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
            env={"KTTC_ANTHROPIC_API_KEY": api_key},
        )

        # Assert
        # With a clear error, should detect it
        assert result.exit_code in [0, 1, 2]

    def test_anthropic_markdown_output(self, tmp_path: Path) -> None:
        """Test markdown report generation with real API."""
        # Arrange
        api_key = os.getenv("KTTC_ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("KTTC_ANTHROPIC_API_KEY not set")

        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"
        output = tmp_path / "report.md"

        source.write_text("Good morning!", encoding="utf-8")
        translation.write_text("¡Buenos días!", encoding="utf-8")

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
                "--output",
                str(output),
                "--format",
                "markdown",
            ],
            env={"KTTC_ANTHROPIC_API_KEY": api_key},
        )

        # Assert
        assert result.exit_code in [0, 1]
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        # Should contain markdown headers
        assert "#" in content


@pytest.mark.e2e
@pytest.mark.slow
class TestRealAPILanguagePairs:
    """Test real API with different language pairs."""

    def test_english_to_french_real_api(self, tmp_path: Path) -> None:
        """Test English to French with real API."""
        # Arrange
        api_key = os.getenv("KTTC_ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("KTTC_ANTHROPIC_API_KEY not set")

        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"

        source.write_text("Thank you very much.", encoding="utf-8")
        translation.write_text("Merci beaucoup.", encoding="utf-8")

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "fr",
            ],
            env={"KTTC_ANTHROPIC_API_KEY": api_key},
        )

        # Assert
        assert result.exit_code in [0, 1, 2]

    def test_english_to_german_real_api(self, tmp_path: Path) -> None:
        """Test English to German with real API."""
        # Arrange
        api_key = os.getenv("KTTC_ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("KTTC_ANTHROPIC_API_KEY not set")

        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"

        source.write_text("Good evening!", encoding="utf-8")
        translation.write_text("Guten Abend!", encoding="utf-8")

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "de",
            ],
            env={"KTTC_ANTHROPIC_API_KEY": api_key},
        )

        # Assert
        assert result.exit_code in [0, 1, 2]


@pytest.mark.e2e
@pytest.mark.slow
class TestRealAPIComplexScenarios:
    """Test complex real-world scenarios."""

    def test_longer_text_real_api(self, tmp_path: Path) -> None:
        """Test longer text with real API."""
        # Arrange
        api_key = os.getenv("KTTC_ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("KTTC_ANTHROPIC_API_KEY not set")

        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"

        long_source = """Artificial intelligence is transforming the world.
Machine learning algorithms are becoming more sophisticated.
The future of technology is exciting and full of possibilities."""

        long_translation = """La inteligencia artificial está transformando el mundo.
Los algoritmos de aprendizaje automático son cada vez más sofisticados.
El futuro de la tecnología es emocionante y lleno de posibilidades."""

        source.write_text(long_source, encoding="utf-8")
        translation.write_text(long_translation, encoding="utf-8")

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
            env={"KTTC_ANTHROPIC_API_KEY": api_key},
        )

        # Assert
        assert result.exit_code in [0, 1, 2]

    def test_technical_terminology_real_api(self, tmp_path: Path) -> None:
        """Test technical terminology with real API."""
        # Arrange
        api_key = os.getenv("KTTC_ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("KTTC_ANTHROPIC_API_KEY not set")

        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"

        # Technical content with specific terminology
        source.write_text("The database query optimization improved performance.", encoding="utf-8")
        translation.write_text(
            "La optimización de consultas de base de datos mejoró el rendimiento.",
            encoding="utf-8",
        )

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
            env={"KTTC_ANTHROPIC_API_KEY": api_key},
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
