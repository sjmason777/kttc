"""Unit tests for prompt template module.

Tests prompt template loading, formatting, and error handling.
"""

import pytest

from kttc.llm.prompts import PromptTemplate, PromptTemplateError


@pytest.mark.unit
class TestPromptTemplate:
    """Test PromptTemplate functionality."""

    def test_init_with_template_text(self) -> None:
        """Test initialization with template text."""
        template_text = "Source: {source_text}\nTranslation: {translation}"
        template = PromptTemplate(template_text)

        assert template.template == template_text

    def test_format_basic(self) -> None:
        """Test basic formatting with variables."""
        template = PromptTemplate("Source: {source_text}, Translation: {translation}")
        result = template.format(source_text="Hello", translation="Hola")

        assert result == "Source: Hello, Translation: Hola"

    def test_format_with_context_none(self) -> None:
        """Test formatting with context=None."""
        template = PromptTemplate("Text: {text}{context_section}")
        result = template.format(text="test", context=None)

        assert "test" in result
        # context_section should be empty when context is None
        assert result.strip() == "Text: test"

    def test_format_with_context_empty(self) -> None:
        """Test formatting with empty context."""
        template = PromptTemplate("Text: {text}{context_section}")
        result = template.format(text="test", context={})

        # Empty dict is falsy, so context_section should be empty
        assert result.strip() == "Text: test"

    def test_format_with_context_domain(self) -> None:
        """Test formatting with domain context."""
        template = PromptTemplate("Text: {text}\n{context_section}")
        result = template.format(text="test", context={"domain": "medical"})

        assert "CONTEXT INFORMATION" in result
        assert "Domain: medical" in result

    def test_format_with_context_glossary(self) -> None:
        """Test formatting with glossary context."""
        template = PromptTemplate("Text: {text}\n{context_section}")
        result = template.format(
            text="test",
            context={"glossary": {"hello": "hola", "world": "mundo"}},
        )

        assert "GLOSSARY" in result
        assert "hello → hola" in result
        assert "world → mundo" in result

    def test_format_with_context_style_guide(self) -> None:
        """Test formatting with style guide context."""
        template = PromptTemplate("Text: {text}\n{context_section}")
        result = template.format(text="test", context={"style_guide": "Formal tone"})

        assert "Style Guide: Formal tone" in result

    def test_format_with_full_context(self) -> None:
        """Test formatting with all context fields."""
        template = PromptTemplate("Text: {text}\n{context_section}")
        result = template.format(
            text="test",
            context={
                "domain": "legal",
                "glossary": {"contract": "contrato"},
                "style_guide": "Formal",
            },
        )

        assert "Domain: legal" in result
        assert "GLOSSARY" in result
        assert "contract → contrato" in result
        assert "Style Guide: Formal" in result

    def test_format_missing_variable_raises(self) -> None:
        """Test error when required variable is missing."""
        template = PromptTemplate("Source: {source_text}")

        with pytest.raises(PromptTemplateError, match="Missing required variable"):
            template.format()


@pytest.mark.unit
class TestPromptTemplateLoad:
    """Test PromptTemplate.load method."""

    def test_load_accuracy_template(self) -> None:
        """Test loading accuracy template."""
        template = PromptTemplate.load("accuracy")
        assert template is not None
        assert isinstance(template.template, str)
        assert len(template.template) > 0

    def test_load_fluency_template(self) -> None:
        """Test loading fluency template."""
        template = PromptTemplate.load("fluency")
        assert template is not None
        assert isinstance(template.template, str)

    def test_load_terminology_template(self) -> None:
        """Test loading terminology template."""
        template = PromptTemplate.load("terminology")
        assert template is not None
        assert isinstance(template.template, str)

    def test_load_nonexistent_template_raises(self) -> None:
        """Test error when template doesn't exist."""
        with pytest.raises(PromptTemplateError, match="Template not found"):
            PromptTemplate.load("nonexistent_agent_template")


@pytest.mark.unit
class TestPromptTemplateFormatContext:
    """Test _format_context method."""

    def test_format_context_domain_only(self) -> None:
        """Test formatting context with domain only."""
        template = PromptTemplate("")
        result = template._format_context({"domain": "technical"})

        assert "CONTEXT INFORMATION" in result
        assert "Domain: technical" in result

    def test_format_context_glossary_only(self) -> None:
        """Test formatting context with glossary only."""
        template = PromptTemplate("")
        result = template._format_context(
            {"glossary": {"term1": "translation1", "term2": "translation2"}}
        )

        assert "GLOSSARY" in result
        assert "term1 → translation1" in result
        assert "term2 → translation2" in result

    def test_format_context_style_only(self) -> None:
        """Test formatting context with style guide only."""
        template = PromptTemplate("")
        result = template._format_context({"style_guide": "Use passive voice"})

        assert "Style Guide: Use passive voice" in result

    def test_format_context_empty(self) -> None:
        """Test formatting empty context."""
        template = PromptTemplate("")
        result = template._format_context({})

        # Should still have header
        assert "CONTEXT INFORMATION" in result


@pytest.mark.unit
class TestPromptTemplateError:
    """Test PromptTemplateError exception."""

    def test_exception_message(self) -> None:
        """Test exception message is preserved."""
        error = PromptTemplateError("Test error message")
        assert str(error) == "Test error message"

    def test_exception_is_subclass_of_exception(self) -> None:
        """Test exception is proper subclass."""
        assert issubclass(PromptTemplateError, Exception)

    def test_exception_can_be_raised_and_caught(self) -> None:
        """Test exception can be properly caught."""
        with pytest.raises(PromptTemplateError):
            raise PromptTemplateError("Test")
