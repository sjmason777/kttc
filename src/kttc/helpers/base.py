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

"""Base abstract class for language-specific helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from kttc.core import ErrorAnnotation


@dataclass
class MorphologyInfo:
    """Morphological information about a word.

    Attributes:
        word: The word text
        pos: Part of speech (NOUN, VERB, ADJ, etc.)
        gender: Grammatical gender (masculine, feminine, neuter)
        case: Grammatical case (nominative, genitive, etc.)
        number: Singular or plural
        aspect: Verb aspect (perfective, imperfective)
        start: Start character position in text
        stop: End character position in text
        metadata: Additional language-specific data
    """

    word: str
    pos: str | None = None
    gender: str | None = None
    case: str | None = None
    number: str | None = None
    aspect: str | None = None
    start: int = 0
    stop: int = 0
    metadata: dict[str, Any] | None = None


class LanguageHelper(ABC):
    """Abstract base class for language-specific NLP helpers.

    Language helpers provide deterministic, rule-based checks to:
    1. Verify LLM error reports (anti-hallucination)
    2. Perform fast grammar checks
    3. Enrich LLM prompts with morphological data

    Each language should implement its own helper subclass.

    Example:
        >>> helper = RussianLanguageHelper()
        >>> exists = helper.verify_word_exists("лиса", text)
        >>> errors = helper.check_grammar(text)
    """

    @property
    @abstractmethod
    def language_code(self) -> str:
        """Get ISO 639-1 language code (e.g., 'ru', 'en', 'zh').

        Returns:
            Two-letter language code
        """

    @abstractmethod
    def verify_word_exists(self, word: str, text: str) -> bool:
        """Verify that a word exists in the text.

        Used to catch LLM hallucinations where it reports errors
        in words that don't actually appear in the text.

        Args:
            word: Word to search for
            text: Text to search in

        Returns:
            True if word exists, False if not found (hallucination)

        Example:
            >>> helper.verify_word_exists("используется", text)
            False  # Word not in text → LLM hallucination
        """

    @abstractmethod
    def verify_error_position(self, error: ErrorAnnotation, text: str) -> bool:
        """Verify that error position is valid and word exists there.

        Args:
            error: Error annotation with location
            text: Full text being checked

        Returns:
            True if position is valid and makes sense, False otherwise

        Example:
            >>> error = ErrorAnnotation(location=(10, 20), ...)
            >>> helper.verify_error_position(error, text)
            True  # Position valid
        """

    @abstractmethod
    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize text with accurate character positions.

        Args:
            text: Text to tokenize

        Returns:
            List of (word, start_pos, end_pos) tuples

        Example:
            >>> helper.tokenize("Быстрая лиса")
            [('Быстрая', 0, 7), ('лиса', 8, 12)]
        """

    @abstractmethod
    def analyze_morphology(self, text: str) -> list[MorphologyInfo]:
        """Analyze morphological properties of all words in text.

        Args:
            text: Text to analyze

        Returns:
            List of MorphologyInfo for each word

        Example:
            >>> info = helper.analyze_morphology("быстрая лиса")
            >>> info[0].pos  # 'ADJF'
            >>> info[0].gender  # 'femn'
        """

    @abstractmethod
    def check_grammar(self, text: str) -> list[ErrorAnnotation]:
        """Perform language-specific grammar checks.

        This is the main method for deterministic error detection.
        Should check for:
        - Agreement errors (gender, case, number)
        - Verb form issues
        - Other language-specific grammar rules

        Args:
            text: Text to check

        Returns:
            List of detected errors

        Example:
            >>> errors = helper.check_grammar("быстрый лиса")
            >>> errors[0].description  # "Gender mismatch: masc ≠ femn"
        """

    def is_available(self) -> bool:
        """Check if helper's dependencies are available.

        Returns:
            True if helper can be used, False if dependencies missing
        """
        return True  # Override in subclasses that have optional deps

    def get_enrichment_data(self, _text: str) -> dict[str, Any]:
        """Get morphological data for enriching LLM prompts.

        Optional method to provide additional context to LLM.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with morphological insights

        Example:
            >>> data = helper.get_enrichment_data(text)
            >>> data['verb_aspects']  # {'перепрыгивает': 'imperfective'}
        """
        return {}
