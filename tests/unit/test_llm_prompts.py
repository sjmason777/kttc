"""Unit tests for LLM prompt template system.

Tests prompt loading, formatting, and error handling.
"""

import pytest

from kttc.llm import PromptTemplate, PromptTemplateError


class TestPromptTemplate:
    """Test PromptTemplate loading and formatting."""

    def test_load_accuracy_template(self) -> None:
        """Test loading accuracy evaluation template."""
        template = PromptTemplate.load("accuracy")
        assert "MQM" in template.template
        assert "EVALUATION CRITERIA" in template.template
        assert "mistranslation" in template.template.lower()

    def test_load_fluency_template(self) -> None:
        """Test loading fluency evaluation template."""
        template = PromptTemplate.load("fluency")
        assert "fluency" in template.template.lower()
        assert "grammar" in template.template.lower()

    def test_load_terminology_template(self) -> None:
        """Test loading terminology evaluation template."""
        template = PromptTemplate.load("terminology")
        assert "terminology" in template.template.lower()
        assert "inconsistency" in template.template.lower()
        assert "misuse" in template.template.lower()

    def test_load_nonexistent_template(self) -> None:
        """Test error handling for nonexistent template."""
        with pytest.raises(PromptTemplateError, match="not found"):
            PromptTemplate.load("nonexistent_template")

    def test_format_basic(self) -> None:
        """Test basic template formatting."""
        template = PromptTemplate.load("accuracy")
        formatted = template.format(
            source_text="Hello, world!",
            translation="Hola, mundo!",
            source_lang="English",
            target_lang="Spanish",
        )
        assert "Hello, world!" in formatted
        assert "Hola, mundo!" in formatted
        assert "English" in formatted
        assert "Spanish" in formatted

    def test_format_with_context(self) -> None:
        """Test template formatting with glossary context."""
        template = PromptTemplate.load("terminology")
        glossary = {
            "patient": "paciente",
            "treatment": "tratamiento",
            "diagnosis": "diagnóstico",
        }
        formatted = template.format(
            source_text="The patient needs treatment.",
            translation="El paciente necesita tratamiento.",
            source_lang="English",
            target_lang="Spanish",
            context={"glossary": glossary, "domain": "medical"},
        )
        assert "patient" in formatted.lower()
        assert "paciente" in formatted.lower()
        assert "medical" in formatted.lower()

    def test_format_without_context(self) -> None:
        """Test terminology template without context (no glossary)."""
        template = PromptTemplate.load("terminology")
        # Should work even without context
        formatted = template.format(
            source_text="Hello",
            translation="Hola",
            source_lang="English",
            target_lang="Spanish",
        )
        assert "Hello" in formatted
        assert "Hola" in formatted

    def test_format_missing_required_field(self) -> None:
        """Test error when required field is missing."""
        template = PromptTemplate.load("accuracy")
        with pytest.raises(PromptTemplateError, match="Missing required variable"):
            template.format(
                source_text="Hello",
                # Missing translation, source_lang, target_lang
            )

    def test_template_immutability(self) -> None:
        """Test that loading the same template twice returns independent instances."""
        template1 = PromptTemplate.load("accuracy")
        template2 = PromptTemplate.load("accuracy")
        assert template1.template == template2.template
        # Formatting one shouldn't affect the other
        formatted1 = template1.format(
            source_text="Test1",
            translation="Prueba1",
            source_lang="en",
            target_lang="es",
        )
        formatted2 = template2.format(
            source_text="Test2",
            translation="Prueba2",
            source_lang="en",
            target_lang="es",
        )
        assert "Test1" in formatted1
        assert "Test2" in formatted2
        assert "Test2" not in formatted1

    def test_all_templates_loadable(self) -> None:
        """Test that all expected templates can be loaded."""
        expected_templates = ["accuracy", "fluency", "terminology"]
        for template_name in expected_templates:
            template = PromptTemplate.load(template_name)
            assert isinstance(template, PromptTemplate)
            assert len(template.template) > 0

    def test_create_from_text(self) -> None:
        """Test creating template from text directly."""
        custom_text = "Evaluate {source_text} -> {translation}"
        template = PromptTemplate(custom_text)
        formatted = template.format(source_text="Hello", translation="Hola")
        assert formatted == "Evaluate Hello -> Hola"

    def test_load_template_read_error(self) -> None:
        """Test error when template file cannot be read."""
        from pathlib import Path
        from unittest.mock import patch

        # Mock the read_text method to raise an exception
        with patch.object(Path, "read_text", side_effect=PermissionError("Access denied")):
            with pytest.raises(PromptTemplateError, match="Error reading template"):
                PromptTemplate.load("accuracy")

    def test_format_general_error(self) -> None:
        """Test general error during template formatting."""
        # Create a template with a format that will cause an error
        template = PromptTemplate("{invalid:d}")  # Trying to format string as int
        with pytest.raises(PromptTemplateError, match="Error formatting template"):
            template.format(invalid="not_an_int")

    def test_format_context_with_style_guide(self) -> None:
        """Test context formatting with style guide."""
        template = PromptTemplate.load("terminology")
        context = {
            "domain": "legal",
            "glossary": {"contract": "contrato", "clause": "cláusula"},
            "style_guide": "Use formal language and avoid contractions",
        }
        formatted = template.format(
            source_text="The contract clause is invalid.",
            translation="La cláusula del contrato es inválida.",
            source_lang="English",
            target_lang="Spanish",
            context=context,
        )
        assert "legal" in formatted.lower()
        assert "formal language" in formatted or "avoid contractions" in formatted
