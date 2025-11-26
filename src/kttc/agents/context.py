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

"""Context Agent with RAG for document-level QA.

Provides document-level context awareness using RAG (Retrieval-Augmented Generation):
- Cross-reference validation
- Term consistency across document
- Coherence checking
- Context-aware error detection

Based on WMT 2025 SELF-RAMT framework.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider

from .base import AgentEvaluationError, AgentParsingError, BaseAgent

logger = logging.getLogger(__name__)


class ContextAgent(BaseAgent):
    """Context-aware QA agent with RAG support.

    Checks for document-level consistency:
    - Cross-reference preservation
    - Term consistency
    - Coherence across segments

    Example:
        >>> agent = ContextAgent(llm_provider)
        >>> # Optional: provide document context
        >>> agent.set_document_context("Full document text...")
        >>> errors = await agent.evaluate(task)
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """Initialize context agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        super().__init__(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.document_context: str | None = None
        self.document_segments: list[dict[str, Any]] = []

    @property
    def category(self) -> str:
        """Error category for this agent."""
        return "context"

    def get_base_prompt(self) -> str:
        """Get the base prompt for context-aware evaluation.

        Returns:
            Base prompt for document-level consistency checking
        """
        return """You are an expert at evaluating document-level consistency in translations.

Your task: Check for context-related issues:

1. **Cross-Reference Preservation**
   - Check if references to sections, figures, tables are preserved
   - Examples: "Section 3", "Figure 2", "Table 5"
   - Verify numbering consistency

2. **Term Consistency**
   - Track terminology usage across segments
   - Same source term should have same translation
   - Flag inconsistent translations of key terms

3. **Coherence Across Segments**
   - Check if segments flow naturally together
   - Verify pronoun references are clear
   - Check if context from previous segments is maintained

4. **Document Structure**
   - Preserve formatting cues
   - Maintain hierarchical structure
   - Keep list/enumeration consistency

Rules:
- Focus on multi-segment consistency
- Flag contradictions or inconsistencies
- Consider document-wide terminology
- Check if references make sense in context

Output JSON format with errors array."""

    def set_document_context(self, full_document: str) -> None:
        """Set full document context for RAG.

        Args:
            full_document: Full document text for context
        """
        self.document_context = full_document
        logger.info(f"Document context set ({len(full_document)} chars)")

    def add_segment(self, source: str, translation: str, segment_id: str | None = None) -> None:
        """Add translation segment to context.

        Builds up context for document-level consistency checking.

        Args:
            source: Source text segment
            translation: Translation segment
            segment_id: Optional segment identifier
        """
        self.document_segments.append(
            {
                "id": segment_id or f"seg_{len(self.document_segments)}",
                "source": source,
                "translation": translation,
            }
        )

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate translation for context-related issues.

        Args:
            task: Translation task

        Returns:
            List of context-related error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        errors: list[ErrorAnnotation] = []

        try:
            # Check 1: Cross-references
            cross_ref_errors = await self._check_cross_references(task)
            errors.extend(cross_ref_errors)

            # Check 2: Term consistency (if we have document context)
            if self.document_segments:
                consistency_errors = await self._check_term_consistency(task)
                errors.extend(consistency_errors)

            # Check 3: Coherence
            coherence_errors = await self._check_coherence(task)
            errors.extend(coherence_errors)

            logger.info(f"ContextAgent found {len(errors)} issues")
            return errors

        except Exception as e:
            logger.error(f"Context agent evaluation failed: {e}")
            raise AgentEvaluationError(f"Context checking failed: {e}") from e

    async def _check_cross_references(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Check if cross-references are preserved.

        Detects references like "Section 3.2", "Figure 1", "see above", etc.

        Args:
            task: Translation task

        Returns:
            List of errors for missing cross-references
        """
        # Extract references from source
        source_refs = self._extract_references(task.source_text)

        if not source_refs:
            return []

        errors = []

        # Check if all source references are preserved
        for ref in source_refs:
            # Simple check: is the reference mentioned in translation?
            # This is language-agnostic (numbers should be preserved)
            ref_numbers = re.findall(r"\d+", ref)

            if ref_numbers:
                # Check if numbers appear in translation
                found = all(num in task.translation for num in ref_numbers)

                if not found:
                    errors.append(
                        ErrorAnnotation(
                            category="context",
                            subcategory="cross_reference_missing",
                            severity=ErrorSeverity.MAJOR,
                            location=(0, min(len(task.translation), 20)),
                            description=f"Cross-reference '{ref}' from source not found in translation",
                            suggestion=f"Include reference: {ref}",
                        )
                    )

        return errors

    def _extract_references(self, text: str) -> list[str]:
        """Extract cross-references from text.

        Args:
            text: Text to extract references from

        Returns:
            List of reference strings
        """
        patterns = [
            r"Section\s+\d+\.?\d*",
            r"Figure\s+\d+",
            r"Table\s+\d+",
            r"Chapter\s+\d+",
            r"Appendix\s+[A-Z]",
            r"page\s+\d+",
            # Add more patterns as needed
        ]

        refs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.extend(matches)

        return list(set(refs))

    async def _check_term_consistency(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Check if terms are translated consistently across document.

        Args:
            task: Translation task

        Returns:
            List of errors for inconsistent terminology
        """
        if not self.document_segments:
            return []

        # Extract key terms from current translation
        current_terms = self._extract_technical_terms(task.translation)

        if not current_terms:
            return []

        # Check how same terms were translated in previous segments
        errors = []

        for term in current_terms:
            previous_translations = self._find_term_translations_in_context(term)

            if len(previous_translations) > 1:
                # Inconsistent translations found
                errors.append(
                    ErrorAnnotation(
                        category="context",
                        subcategory="term_inconsistency",
                        severity=ErrorSeverity.MINOR,
                        location=self._find_term_location(task.translation, term),
                        description=(
                            f"Term '{term}' translated inconsistently in document. "
                            f"Previous translations: {', '.join(set(previous_translations))}"
                        ),
                        suggestion=f"Use consistent translation: {previous_translations[0]}",
                    )
                )

        return errors

    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract potential technical terms from text.

        Simple heuristic: capitalized words, acronyms, compound terms.

        Args:
            text: Text to extract terms from

        Returns:
            List of terms
        """
        # Capitalized words
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", text)

        # Acronyms (2+ consecutive capital letters)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)

        # Compound terms (hyphenated or multi-word technical terms)
        compounds = re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", text)

        terms = list(set(capitalized + acronyms + compounds))

        # Filter out common words (simple heuristic)
        terms = [t for t in terms if len(t) > 2]

        return terms[:20]  # Limit to avoid too many checks

    def _find_term_translations_in_context(self, term: str) -> list[str]:
        """Find how term was translated in previous segments.

        Args:
            term: Term to search for

        Returns:
            List of translation variants found
        """
        translations = []

        for segment in self.document_segments:
            # Simple presence check
            if term.lower() in segment["translation"].lower():
                translations.append(term)  # Found same term

        return translations

    def _find_term_location(self, text: str, term: str) -> tuple[int, int]:
        """Find location of term in text.

        Args:
            text: Text to search
            term: Term to find

        Returns:
            Character position tuple (start, end)
        """
        index = text.lower().find(term.lower())

        if index != -1:
            return (index, index + len(term))

        # Fallback
        return (0, min(len(text), 10))

    async def _check_coherence(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Check translation coherence using LLM.

        Args:
            task: Translation task

        Returns:
            List of coherence errors
        """
        # Only check if we have context
        if not self.document_segments:
            return []

        # Build context from recent segments
        context_text = self._build_context_text()

        prompt = f"""You are an expert at evaluating translation coherence.

Your task: Check if the current translation is coherent with previous segments.

## PREVIOUS CONTEXT:
{context_text}

## CURRENT SOURCE ({task.source_lang}):
{task.source_text}

## CURRENT TRANSLATION ({task.target_lang}):
{task.translation}

Instructions:
1. Check if current translation is coherent with previous context
2. Look for:
   - Pronoun/reference ambiguity
   - Contradictions with previous segments
   - Sudden tone/style changes
   - Logical flow issues

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "coherence_issue|reference_ambiguity|style_inconsistency",
      "severity": "major|minor",
      "description": "Explanation of coherence issue"
    }}
  ]
}}

If translation is coherent, return empty errors array.

Output only valid JSON, no explanation."""

        try:
            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Parse response
            response_data = self._parse_json_response(response)
            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                errors.append(
                    ErrorAnnotation(
                        category="context",
                        subcategory=error_dict.get("subcategory", "coherence_issue"),
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=(0, len(task.translation)),
                        description=error_dict.get("description", "Coherence issue detected"),
                        suggestion=None,
                    )
                )

            return errors

        except Exception as e:
            logger.warning(f"Coherence check failed: {e}")
            return []

    def _build_context_text(self, max_segments: int = 3) -> str:
        """Build context text from recent segments.

        Args:
            max_segments: Maximum number of recent segments to include

        Returns:
            Context text
        """
        if not self.document_segments:
            return "(No previous context)"

        # Get last N segments
        recent = self.document_segments[-max_segments:]

        lines = []
        for seg in recent:
            lines.append(f"[{seg['id']}]")
            lines.append(f"  Translation: {seg['translation']}")

        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON dictionary

        Raises:
            AgentParsingError: If parsing fails
        """
        try:
            return cast(dict[str, Any], json.loads(response))
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(1)))
                except json.JSONDecodeError as e:
                    raise AgentParsingError(f"Failed to parse JSON: {e}") from e

            # Try to find JSON object
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(0)))
                except json.JSONDecodeError as e:
                    raise AgentParsingError(f"Failed to parse JSON: {e}") from e

            raise AgentParsingError(f"No valid JSON found in response: {response[:200]}")

    def clear_context(self) -> None:
        """Clear document context and segments."""
        self.document_context = None
        self.document_segments = []
        logger.info("Context cleared")
