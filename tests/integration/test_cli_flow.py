"""Integration tests for full CLI flow.

Tests complete CLI workflows with multiple components using demo mode.
Demo mode provides predictable responses without API calls, enabling
stricter assertions and reliable CI/CD integration.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kttc.cli.main import app

runner = CliRunner()


@pytest.mark.integration
class TestCLIIntegrationFlow:
    """Test complete CLI workflows using demo mode for predictable results."""

    def test_check_command_end_to_end(
        self, temp_text_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Test complete check command flow with demo provider."""
        # Arrange
        source, translation = temp_text_files
        output = tmp_path / "report.json"

        # Act - Use --demo flag for predictable responses without API calls
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
                "--demo",
            ],
        )

        # Assert - Demo mode should always succeed
        assert (
            result.exit_code == 0
        ), f"Expected success, got exit_code={result.exit_code}, output: {result.stdout}"
        assert output.exists(), "Output file should be created"

        # Verify JSON structure
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "mqm_score" in data, "Report should contain mqm_score"
        assert "errors" in data, "Report should contain errors list"
        assert isinstance(data["mqm_score"], int | float), "mqm_score should be numeric"

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
                "--demo",
            ],
        )

        # Assert - Demo mode should succeed
        assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
        assert output.exists(), "Markdown output file should be created"

        # Verify markdown content
        content = output.read_text(encoding="utf-8")
        assert len(content) > 0, "Markdown report should not be empty"
        assert (
            "translation" in content.lower() or "#" in content
        ), "Should contain markdown headers or translation content"

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
                "--demo",
            ],
        )

        # Assert - Demo mode should succeed
        assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
        assert output.exists(), "Text output file should be created"

        content = output.read_text(encoding="utf-8")
        assert len(content) > 0, "Text report should not be empty"
        # Check for expected content fields
        assert "mqm_score" in content or "status" in content or "errors" in content

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
                "--demo",
            ],
        )

        # Assert - Should succeed and print to stdout
        assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
        assert len(result.stdout) > 0, "Should print report to stdout"


@pytest.mark.integration
class TestAgentPipelineIntegration:
    """Test agent orchestration with demo provider for predictable results."""

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
                "--demo",
            ],
        )

        # Assert - Demo mode should succeed
        assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
        assert output.exists(), "Report file should be created"

        data = json.loads(output.read_text(encoding="utf-8"))
        assert "mqm_score" in data, "Should contain MQM score"
        assert "errors" in data, "Should contain errors list"
        assert isinstance(data["errors"], list), "Errors should be a list"

    def test_quality_threshold_enforcement_pass(self, temp_text_files: tuple[Path, Path]) -> None:
        """Test that quality threshold is enforced - pass scenario."""
        # Arrange
        source, translation = temp_text_files

        # Act - Set threshold below demo score (~95.0)
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
                "90.0",
                "--demo",
            ],
        )

        # Assert - Should pass with demo score of ~95
        assert result.exit_code == 0, f"Expected success with threshold 90.0, got: {result.stdout}"

    def test_quality_threshold_enforcement_fail(self, tmp_path: Path) -> None:
        """Test that quality threshold is enforced - fail scenario.

        Uses Russian target language which has rule-based checks that can
        detect real errors and produce lower scores.
        """
        # Arrange - Use text that will trigger Russian grammar checks
        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"
        source.write_text("Hello world", encoding="utf-8")
        # Missing comma in Russian greeting - should trigger error
        translation.write_text("Привет мир", encoding="utf-8")

        # Act - Set threshold at 95 (default), Russian check should fail
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
                "--threshold",
                "95.0",
                "--demo",
            ],
        )

        # Assert - Russian grammar check finds missing comma, score drops below 95
        assert (
            result.exit_code == 1
        ), f"Expected exit code 1 for threshold failure (Russian grammar), got: {result.exit_code}"


@pytest.mark.integration
class TestBatchProcessing:
    """Test batch processing capabilities with demo mode."""

    def test_batch_file_with_multiple_segments(self, tmp_path: Path) -> None:
        """Test processing multiple translation pairs."""
        # Arrange - Create source and translation files
        source = tmp_path / "source.txt"
        translation = tmp_path / "translation.txt"
        source.write_text("Hello world\nHow are you?\nGood morning", encoding="utf-8")
        translation.write_text("Hola mundo\n¿Cómo estás?\nBuenos días", encoding="utf-8")
        output = tmp_path / "batch_report.json"

        # Act - Use standard check command with two files
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
                "--demo",
            ],
        )

        # Assert - Demo mode should succeed
        assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
        assert output.exists(), "Output file should be created"


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
                "--demo",
            ],
        )

        # Assert - Should handle invalid language gracefully
        # Demo mode still processes but language detection may warn
        assert result.exit_code in [0, 1], f"Expected 0 or 1, got: {result.exit_code}"

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
                "--demo",
            ],
        )

        # Assert - Should exit with error code for missing file
        assert (
            result.exit_code == 1
        ), f"Expected error code 1 for missing file, got: {result.exit_code}"
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

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
                "--demo",
            ],
        )

        # Assert - Should handle empty file with error
        assert (
            result.exit_code == 1
        ), f"Expected error code 1 for empty file, got: {result.exit_code}"


@pytest.mark.integration
class TestLanguagePairs:
    """Test different language pairs with demo mode."""

    @pytest.mark.parametrize(
        "target_lang,translation_text",
        [
            ("es", "Hola mundo"),
            ("fr", "Bonjour monde"),
            ("de", "Hallo Welt"),
            ("ru", "Привет, мир"),  # With comma for correct Russian punctuation
        ],
        ids=["en-es", "en-fr", "en-de", "en-ru"],
    )
    def test_english_to_various_languages(
        self, tmp_path: Path, target_lang: str, translation_text: str
    ) -> None:
        """Test English to various language translation checks."""
        # Arrange
        source = tmp_path / "en_source.txt"
        translation = tmp_path / f"{target_lang}_translation.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text(translation_text, encoding="utf-8")

        # Act - Use lower threshold to allow for minor language-specific rules
        result = runner.invoke(
            app,
            [
                "check",
                str(source),
                str(translation),
                "--source-lang",
                "en",
                "--target-lang",
                target_lang,
                "--threshold",
                "60.0",  # Lower threshold to allow for language-specific checks
                "--demo",
            ],
        )

        # Assert - All language pairs should succeed with relaxed threshold
        assert (
            result.exit_code == 0
        ), f"Expected success for en->{target_lang}, got: {result.stdout}"


@pytest.mark.integration
class TestOutputFormatsIntegration:
    """Test different output formats with demo mode."""

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
                "--demo",
            ],
        )

        # Assert - Demo mode should succeed
        assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
        assert output.exists(), "HTML output file should be created"

        content = output.read_text(encoding="utf-8")
        assert "<html" in content.lower() or "<!doctype" in content.lower(), "Should be valid HTML"

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
                "--demo",
            ],
        )

        # Assert - Demo mode should succeed
        assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
        assert output.exists(), "JSON output file should be created"

        data = json.loads(output.read_text(encoding="utf-8"))
        # Check required fields
        assert "mqm_score" in data, "Should contain mqm_score"
        assert "errors" in data, "Should contain errors"
        assert "status" in data, "Should contain status"
        assert isinstance(data["mqm_score"], int | float), "mqm_score should be numeric"
        assert isinstance(data["errors"], list), "errors should be a list"
        assert data["status"] in ["pass", "fail"], "status should be pass or fail"


@pytest.mark.integration
class TestTextProcessing:
    """Test processing of different text types with demo mode."""

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
                "--demo",
            ],
        )

        # Assert - Demo mode should handle longer text
        assert result.exit_code == 0, f"Expected success for longer text, got: {result.stdout}"

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
                "--demo",
            ],
        )

        # Assert - Demo mode should handle special characters
        assert result.exit_code == 0, f"Expected success for special chars, got: {result.stdout}"

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
                "--demo",
            ],
        )

        # Assert - Demo mode should handle emojis
        assert result.exit_code == 0, f"Expected success for emojis, got: {result.stdout}"


@pytest.mark.integration
class TestPerformance:
    """Test performance characteristics with demo mode."""

    def test_multiple_segments_performance(self, tmp_path: Path) -> None:
        """Test performance with multiple translation lines."""
        # Arrange - Create source and translation files with multiple lines
        source = tmp_path / "multi_source.txt"
        translation = tmp_path / "multi_translation.txt"

        source_lines = [f"This is sentence {i + 1}." for i in range(5)]
        translation_lines = [f"Esta es la oración {i + 1}." for i in range(5)]

        source.write_text("\n".join(source_lines), encoding="utf-8")
        translation.write_text("\n".join(translation_lines), encoding="utf-8")

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
                "--demo",
            ],
        )

        # Assert - Demo mode should handle multiple lines
        assert result.exit_code == 0, f"Expected success for multi-segment, got: {result.stdout}"
