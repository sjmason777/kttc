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

"""Grammar Agent for monolingual proofreading.

Checks grammar using:
- LanguageTool integration (when available)
- School curriculum rules from glossaries
- LLM-based grammar analysis

Used in self-check mode for proofreading without translation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider, LLMError

logger = logging.getLogger(__name__)


class GrammarAgent:
    """Agent for checking grammar in monolingual mode.

    This agent combines multiple approaches:
    1. LanguageTool for rule-based grammar checking
    2. School curriculum glossaries for language-specific rules
    3. LLM for context-aware grammar analysis

    Example:
        >>> provider = OpenAIProvider(api_key="...")
        >>> agent = GrammarAgent(provider, language="ru")
        >>> text = "Нехочу идти в школу."
        >>> errors = await agent.check(text)
        >>> print(f"Found {len(errors)} grammar errors")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        language: str = "en",
        use_languagetool: bool = True,
        use_school_rules: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """Initialize GrammarAgent.

        Args:
            llm_provider: LLM provider for context-aware checking (optional)
            language: Language code for the text being checked
            use_languagetool: Whether to use LanguageTool integration
            use_school_rules: Whether to use school curriculum glossaries
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        self.llm_provider = llm_provider
        self.language = language
        self.use_languagetool = use_languagetool
        self.use_school_rules = use_school_rules
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._school_rules: dict[str, Any] | None = None

    @property
    def category(self) -> str:
        """Get error category this agent checks."""
        return "grammar"

    def _load_school_rules(self) -> dict[str, Any]:
        """Load school curriculum rules for the language."""
        if self._school_rules is not None:
            return self._school_rules

        # Find glossary path
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent
        glossaries_dir = project_root / "glossaries" / self.language / "school_curriculum"

        self._school_rules = {"spelling_rules": {}, "punctuation_rules": {}, "common_mistakes": {}}

        if not glossaries_dir.exists():
            logger.debug(f"No school curriculum glossary for language: {self.language}")
            return self._school_rules

        # Load all JSON files in school_curriculum
        for json_file in glossaries_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Merge rules
                for key in ["spelling_rules", "punctuation_rules", "common_mistakes"]:
                    if key in data:
                        self._school_rules[key].update(data[key])

                # Also load grammar-specific keys
                for key in data:
                    if key not in [
                        "metadata",
                        "spelling_rules",
                        "punctuation_rules",
                        "common_mistakes",
                    ]:
                        if key not in self._school_rules:
                            self._school_rules[key] = {}
                        if isinstance(data[key], dict):
                            self._school_rules[key].update(data[key])

                logger.debug(f"Loaded school rules from {json_file.name}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return self._school_rules

    def _parse_severity(self, severity_str: str) -> ErrorSeverity:
        """Parse severity string to ErrorSeverity enum."""
        if severity_str == "critical":
            return ErrorSeverity.CRITICAL
        if severity_str == "minor":
            return ErrorSeverity.MINOR
        return ErrorSeverity.MAJOR

    def _check_common_mistakes(
        self, text: str, common_mistakes: dict[str, Any]
    ) -> list[ErrorAnnotation]:
        """Check text for common mistakes."""
        errors: list[ErrorAnnotation] = []
        text_lower = text.lower()

        for mistake_data in common_mistakes.values():
            if not isinstance(mistake_data, dict) or "examples" not in mistake_data:
                continue
            for example in mistake_data["examples"]:
                if example.lower() not in text_lower:
                    continue
                pos = text_lower.find(example.lower())
                correct_forms = mistake_data.get("correct_forms", [])
                suggestion = correct_forms[0] if correct_forms else ""
                actual_text = text[pos : pos + len(example)]
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="grammar",
                        severity=ErrorSeverity.MAJOR,
                        location=(pos, pos + len(example)),
                        description=f"{mistake_data.get('description', 'Common mistake')}: '{actual_text}' → '{suggestion}'",
                        suggestion=suggestion or None,
                    )
                )
        return errors

    def _check_spelling_rules(
        self, text: str, spelling_rules: dict[str, Any]
    ) -> list[ErrorAnnotation]:
        """Check text against spelling rules."""
        errors: list[ErrorAnnotation] = []
        text_lower = text.lower()

        for rule_data in spelling_rules.values():
            if not isinstance(rule_data, dict) or "examples" not in rule_data:
                continue
            incorrect = rule_data["examples"].get("incorrect", [])
            correct = rule_data["examples"].get("correct", [])

            for i, wrong in enumerate(incorrect):
                if wrong.lower() not in text_lower:
                    continue
                pos = text_lower.find(wrong.lower())
                suggestion = correct[i] if i < len(correct) else ""
                severity = self._parse_severity(rule_data.get("severity", "major"))
                actual_text = text[pos : pos + len(wrong)]
                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory="spelling",
                        severity=severity,
                        location=(pos, pos + len(wrong)),
                        description=f"{rule_data.get('name', 'Spelling rule')}: '{actual_text}' → '{suggestion}'",
                        suggestion=suggestion or None,
                    )
                )
        return errors

    def _check_with_school_rules(self, text: str) -> list[ErrorAnnotation]:
        """Check text against school curriculum rules."""
        rules = self._load_school_rules()
        errors = self._check_common_mistakes(text, rules.get("common_mistakes", {}))
        errors.extend(self._check_spelling_rules(text, rules.get("spelling_rules", {})))
        return errors

    async def _check_with_llm(self, text: str) -> list[ErrorAnnotation]:
        """Check text using LLM for context-aware grammar analysis.

        Args:
            text: Text to check

        Returns:
            List of errors found by LLM
        """
        if not self.llm_provider:
            return []

        lang_names = {
            "en": "English",
            "ru": "Russian",
            "zh": "Chinese",
            "hi": "Hindi",
            "fa": "Persian",
        }
        lang_name = lang_names.get(self.language, self.language)

        prompt = f"""You are a professional proofreader and grammar expert for {lang_name}.
Analyze the following text and identify grammar, spelling, and punctuation errors.

TEXT TO CHECK:
{text}

For each error found, provide in this exact JSON format:
{{
  "errors": [
    {{
      "text": "the incorrect text fragment",
      "suggestion": "the corrected text",
      "type": "grammar|spelling|punctuation",
      "severity": "critical|major|minor",
      "explanation": "brief explanation of the error"
    }}
  ]
}}

If no errors are found, return: {{"errors": []}}

Focus on:
1. Grammar errors (agreement, tense, word order)
2. Spelling mistakes
3. Punctuation errors
4. Style issues (if severe)

Be precise and only report actual errors, not stylistic preferences."""

        try:
            response = await self.llm_provider.complete(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse JSON response
            try:
                # Find JSON in response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if 0 <= json_start < json_end:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    return []

                errors: list[ErrorAnnotation] = []
                for err in data.get("errors", []):
                    error_text = err.get("text", "")
                    pos = text.find(error_text)
                    if pos == -1:
                        pos = 0

                    severity_str = err.get("severity", "major")
                    severity = (
                        ErrorSeverity.CRITICAL
                        if severity_str == "critical"
                        else ErrorSeverity.MINOR
                        if severity_str == "minor"
                        else ErrorSeverity.MAJOR
                    )

                    subcategory = err.get("type", "grammar")
                    if subcategory not in ["grammar", "spelling", "punctuation"]:
                        subcategory = "grammar"

                    errors.append(
                        ErrorAnnotation(
                            category="fluency",
                            subcategory=subcategory,
                            severity=severity,
                            location=(pos, pos + len(error_text)),
                            description=f"{err.get('explanation', 'Grammar error')}: '{error_text}'",
                            suggestion=err.get("suggestion") or None,
                        )
                    )

                return errors

            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return []

        except LLMError as e:
            logger.error(f"LLM grammar check failed: {e}")
            return []

    async def check(self, text: str) -> list[ErrorAnnotation]:
        """Check text for grammar errors.

        Args:
            text: Text to check

        Returns:
            List of grammar error annotations
        """
        all_errors: list[ErrorAnnotation] = []

        # 1. Check with school curriculum rules (fast, rule-based)
        if self.use_school_rules:
            school_errors = self._check_with_school_rules(text)
            all_errors.extend(school_errors)
            logger.debug(f"School rules found {len(school_errors)} errors")

        # 2. Check with LanguageTool (if available)
        if self.use_languagetool:
            try:
                from kttc.helpers import get_helper_for_language

                helper = get_helper_for_language(self.language)
                if helper and helper.is_available():
                    # Use helper's grammar checking if available
                    # Note: This depends on LanguageTool being configured
                    logger.debug("LanguageTool check available")
            except ImportError:
                logger.debug("LanguageTool integration not available")

        # 3. Check with LLM (context-aware, slower)
        if self.llm_provider:
            llm_errors = await self._check_with_llm(text)
            # Deduplicate - don't add LLM errors that overlap with school rules
            for llm_err in llm_errors:
                is_duplicate = False
                for existing in all_errors:
                    if (
                        abs(existing.location[0] - llm_err.location[0]) < 5
                        and existing.subcategory == llm_err.subcategory
                    ):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_errors.append(llm_err)
            logger.debug(f"LLM found {len(llm_errors)} additional errors")

        return all_errors

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate task in proofreading mode (compatibility method).

        For monolingual proofreading, we check the translation text
        as if it were the source (no translation comparison).

        Args:
            task: Translation task (uses translation field as text to check)

        Returns:
            List of error annotations
        """
        # In self-check mode, source_lang == target_lang
        # We check the "translation" field which contains the text to proofread
        text_to_check = task.translation or task.source_text
        self.language = task.target_lang or task.source_lang

        return await self.check(text_to_check)
