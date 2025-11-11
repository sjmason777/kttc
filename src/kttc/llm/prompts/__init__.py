# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt template management for KTTC agents.

Provides utilities for loading and formatting prompt templates
for different QA agents (accuracy, fluency, terminology).
"""

from pathlib import Path
from typing import Any


class PromptTemplateError(Exception):
    """Raised when there's an error loading or formatting a prompt template."""

    pass


class PromptTemplate:
    """Manages prompt templates for QA agents.

    Templates are stored as .txt files and support variable substitution.

    Example:
        >>> template = PromptTemplate.load("accuracy")
        >>> prompt = template.format(
        ...     source_text="Hello",
        ...     translation="Hola",
        ...     source_lang="en",
        ...     target_lang="es"
        ... )
    """

    def __init__(self, template_text: str):
        """Initialize with template text.

        Args:
            template_text: The raw template text with placeholders
        """
        self.template = template_text

    @classmethod
    def load(cls, agent_name: str) -> "PromptTemplate":
        """Load a prompt template for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "accuracy", "fluency", "terminology")

        Returns:
            PromptTemplate instance

        Raises:
            PromptTemplateError: If template file doesn't exist

        Example:
            >>> template = PromptTemplate.load("accuracy")
        """
        template_dir = Path(__file__).parent
        template_file = template_dir / f"{agent_name}.txt"

        if not template_file.exists():
            raise PromptTemplateError(
                f"Template not found: {agent_name} (expected: {template_file})"
            )

        try:
            template_text = template_file.read_text(encoding="utf-8")
            return cls(template_text)
        except Exception as e:
            raise PromptTemplateError(f"Error reading template {agent_name}: {e}") from e

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables.

        Args:
            **kwargs: Template variables (source_text, translation, etc.)

        Returns:
            Formatted prompt string

        Raises:
            PromptTemplateError: If required variables are missing

        Example:
            >>> template = PromptTemplate.load("accuracy")
            >>> prompt = template.format(
            ...     source_text="Hello",
            ...     translation="Hola",
            ...     source_lang="en",
            ...     target_lang="es"
            ... )
        """
        try:
            # Handle optional context section for terminology agent
            if "context" in kwargs and kwargs["context"]:
                context_info = self._format_context(kwargs.pop("context"))
                kwargs["context_section"] = context_info
            else:
                kwargs["context_section"] = ""

            return self.template.format(**kwargs)
        except KeyError as e:
            raise PromptTemplateError(f"Missing required variable: {e}") from e
        except Exception as e:
            raise PromptTemplateError(f"Error formatting template: {e}") from e

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context information for the prompt.

        Args:
            context: Context dictionary with domain, glossary, etc.

        Returns:
            Formatted context section

        Example:
            >>> context = {"domain": "medical", "glossary": {...}}
            >>> formatted = template._format_context(context)
        """
        lines = ["## CONTEXT INFORMATION:"]

        if "domain" in context:
            lines.append(f"Domain: {context['domain']}")

        if "glossary" in context:
            lines.append("\n## GLOSSARY:")
            glossary = context["glossary"]
            for source_term, target_term in glossary.items():
                lines.append(f"  - {source_term} → {target_term}")

        if "style_guide" in context:
            lines.append(f"\nStyle Guide: {context['style_guide']}")

        return "\n".join(lines)
