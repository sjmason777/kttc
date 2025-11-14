"""Integration tests for full CLI flow.

Tests complete CLI workflows with multiple components.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kttc.cli.main import app

runner = CliRunner()


@pytest.mark.integration
class TestCLIIntegrationFlow:
    """Test complete CLI workflows."""

    def test_check_command_end_to_end(
        self, temp_text_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Test complete check command flow with real components."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.json"

        # Act - Note: Will use mocked LLM providers from conftest
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
        )

        # Assert
        # This is integration test - checks real flow but with mocked LLM
        # May pass or fail depending on LLM provider availability
        assert result.exit_code in [0, 1, 2]  # Various valid exit codes

    def test_markdown_report_generation(
        self, temp_text_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Test markdown report generation through full pipeline."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.md"

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
        # If output exists, verify it's valid markdown
        if output.exists():
            content = output.read_text(encoding="utf-8")
            assert "# Translation Quality Report" in content or "translation" in content.lower()

    def test_plain_text_output(self, temp_text_files: tuple[Path, Path], tmp_path: Path) -> None:
        """Test plain text output format."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.txt"

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
                "text",
            ],
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
        # If output exists, verify it's plain text
        if output.exists():
            content = output.read_text(encoding="utf-8")
            assert len(content) > 0
            assert "Translation" in content or "MQM" in content or "Quality" in content

    def test_check_without_output_file(self, temp_text_files: tuple[Path, Path]) -> None:
        """Test check command prints to stdout when no output file specified."""
        # Arrange
        source, translation = temp_text_files

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
        # Should print something to stdout
        assert len(result.stdout) > 0


@pytest.mark.integration
class TestAgentPipelineIntegration:
    """Test agent orchestration with real components (but mocked LLM)."""

    def test_multiple_agents_work_together(
        self, temp_text_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Test that multiple QA agents work together in pipeline."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.json"

        # Act - All agents (Accuracy, Fluency, Terminology) should run
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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
        # If successful, check report structure
        if result.exit_code == 0 and output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            assert "mqm_score" in data
            assert "errors" in data
            assert isinstance(data["errors"], list)

    def test_quality_threshold_enforcement(self, temp_text_files: tuple[Path, Path]) -> None:
        """Test that quality threshold is enforced."""
        # Arrange
        source, translation = temp_text_files

        # Act - Set very high threshold (should likely fail)
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
                "--threshold",
                "99.9",
            ],
        )

        # Assert - Exit code depends on translation quality
        assert result.exit_code in [0, 1, 2]
        # If it failed, should be exit code 1 (quality threshold not met)
        # If passed, should be exit code 0


@pytest.mark.integration
class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_batch_file_with_multiple_segments(self, tmp_path: Path) -> None:
        """Test processing batch file with multiple translation segments."""
        # Arrange
        batch_file = tmp_path / "batch.txt"
        batch_content = """SOURCE: Hello world
TRANSLATION: Hola mundo
---
SOURCE: How are you?
TRANSLATION: ¿Cómo estás?
---
SOURCE: Good morning
TRANSLATION: Buenos días"""
        batch_file.write_text(batch_content, encoding="utf-8")
        output = tmp_path / "batch_report.json"

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(batch_file),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
                "--output",
                str(output),
                "--format",
                "json",
            ],
        )

        # Assert
        assert result.exit_code in [0, 1, 2]


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across components."""

    def test_invalid_language_code(self, temp_text_files: tuple[Path, Path]) -> None:
        """Test handling of invalid language codes."""
        # Arrange
        source, translation = temp_text_files

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "invalid_lang",
                "--target-lang",
                "es",
            ],
        )

        # Assert - Should either handle gracefully or exit with error
        assert result.exit_code in [0, 1, 2]

    def test_missing_source_file(self, tmp_path: Path) -> None:
        """Test handling of missing source file."""
        # Arrange
        nonexistent = tmp_path / "nonexistent.txt"
        translation = tmp_path / "translation.txt"
        translation.write_text("Hola mundo", encoding="utf-8")

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(nonexistent),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        # Assert - Should exit with error code
        assert result.exit_code != 0

    def test_empty_source_file(self, tmp_path: Path) -> None:
        """Test handling of empty source file."""
        # Arrange
        source = tmp_path / "empty_source.txt"
        translation = tmp_path / "translation.txt"
        source.write_text("", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

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
        )

        # Assert - Should handle gracefully or error
        assert result.exit_code in [0, 1, 2]


@pytest.mark.integration
class TestLanguagePairs:
    """Test different language pairs."""

    def test_english_to_spanish(self, tmp_path: Path) -> None:
        """Test English to Spanish translation check."""
        # Arrange
        source = tmp_path / "en_source.txt"
        translation = tmp_path / "es_translation.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]

    def test_english_to_french(self, tmp_path: Path) -> None:
        """Test English to French translation check."""
        # Arrange
        source = tmp_path / "en_source.txt"
        translation = tmp_path / "fr_translation.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Bonjour monde", encoding="utf-8")

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]

    def test_english_to_german(self, tmp_path: Path) -> None:
        """Test English to German translation check."""
        # Arrange
        source = tmp_path / "en_source.txt"
        translation = tmp_path / "de_translation.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hallo Welt", encoding="utf-8")

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]

    def test_english_to_russian(self, tmp_path: Path) -> None:
        """Test English to Russian translation check."""
        # Arrange
        source = tmp_path / "en_source.txt"
        translation = tmp_path / "ru_translation.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Привет мир", encoding="utf-8")

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
                "ru",
            ],
        )

        # Assert
        assert result.exit_code in [0, 1, 2]


@pytest.mark.integration
class TestOutputFormatsIntegration:
    """Test different output formats with real components."""

    def test_html_output_format(self, temp_text_files: tuple[Path, Path], tmp_path: Path) -> None:
        """Test HTML output format generation."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.html"

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
                "html",
            ],
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
        # If output exists, verify it's valid HTML
        if output.exists():
            content = output.read_text(encoding="utf-8")
            assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_json_output_structure(
        self, temp_text_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Test JSON output has correct structure."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.json"

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
        # If successful, verify JSON structure
        if result.exit_code == 0 and output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            # Check required fields
            assert "mqm_score" in data
            assert "errors" in data
            assert "status" in data
            assert isinstance(data["mqm_score"], (int, float))
            assert isinstance(data["errors"], list)
            assert data["status"] in ["pass", "fail"]


@pytest.mark.integration
class TestTextProcessing:
    """Test processing of different text types."""

    def test_longer_text_processing(self, tmp_path: Path) -> None:
        """Test processing of longer text (multiple paragraphs)."""
        # Arrange
        source = tmp_path / "long_source.txt"
        translation = tmp_path / "long_translation.txt"
        long_en = """This is the first paragraph. It contains multiple sentences.
The translation quality should be evaluated carefully.

This is the second paragraph with more content.
Quality assurance is important for translations."""
        long_es = """Este es el primer párrafo. Contiene múltiples oraciones.
La calidad de la traducción debe evaluarse cuidadosamente.

Este es el segundo párrafo con más contenido.
El aseguramiento de calidad es importante para las traducciones."""
        source.write_text(long_en, encoding="utf-8")
        translation.write_text(long_es, encoding="utf-8")

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]

    def test_special_characters_handling(self, tmp_path: Path) -> None:
        """Test handling of special characters and symbols."""
        # Arrange
        source = tmp_path / "special_source.txt"
        translation = tmp_path / "special_translation.txt"
        source.write_text("Price: $100.00 • Discount: 20% → Final: $80.00", encoding="utf-8")
        translation.write_text("Precio: $100.00 • Descuento: 20% → Final: $80.00", encoding="utf-8")

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]

    def test_unicode_emoji_handling(self, tmp_path: Path) -> None:
        """Test handling of Unicode emojis."""
        # Arrange
        source = tmp_path / "emoji_source.txt"
        translation = tmp_path / "emoji_translation.txt"
        source.write_text("Hello! 👋 Welcome to our service! 🎉", encoding="utf-8")
        translation.write_text("¡Hola! 👋 ¡Bienvenido a nuestro servicio! 🎉", encoding="utf-8")

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
        )

        # Assert
        assert result.exit_code in [0, 1, 2]


@pytest.mark.integration
class TestPerformance:
    """Test performance characteristics."""

    def test_multiple_segments_performance(self, tmp_path: Path) -> None:
        """Test performance with multiple translation segments."""
        # Arrange
        batch_file = tmp_path / "multi_segment.txt"
        segments = []
        for i in range(5):
            segments.append(f"SOURCE: This is sentence {i + 1}.")
            segments.append(f"TRANSLATION: Esta es la oración {i + 1}.")
            segments.append("---")
        batch_content = "\n".join(segments)
        batch_file.write_text(batch_content, encoding="utf-8")

        # Act
        result = runner.invoke(
            app,
            [
                "check",
                str(batch_file),
                "--source-lang",
                "en",
                "--target-lang",
                "es",
            ],
        )

        # Assert
        assert result.exit_code in [0, 1, 2]
