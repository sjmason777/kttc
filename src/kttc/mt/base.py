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

"""Base abstract class for Machine Translation providers.

Defines the interface that all MT providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result of a translation request.

    Attributes:
        text: Translated text
        source_lang: Detected or specified source language
        target_lang: Target language
        characters: Number of characters translated (for billing)
    """

    text: str
    source_lang: str
    target_lang: str
    characters: int = 0


class BaseMTProvider(ABC):
    """Abstract base class for Machine Translation providers.

    All MT providers (DeepL, Google Translate, etc.) must implement this interface.

    Attributes:
        total_characters: Total characters translated (for usage tracking)
    """

    def __init__(self) -> None:
        """Initialize provider with usage tracking."""
        self.total_characters = 0

    @abstractmethod
    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
    ) -> TranslationResult:
        """Translate text to target language.

        Args:
            text: Text to translate
            target_lang: Target language code (e.g., "EN", "DE", "RU")
            source_lang: Optional source language code (auto-detect if None)

        Returns:
            TranslationResult with translated text and metadata

        Raises:
            MTError: If translation fails
            MTQuotaExceededError: If API quota is exceeded

        Example:
            >>> provider = DeepLProvider(api_key="...")
            >>> result = await provider.translate("Hello", target_lang="DE")
            >>> print(result.text)
            'Hallo'
        """
        ...

    @abstractmethod
    async def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslationResult]:
        """Translate multiple texts to target language.

        Args:
            texts: List of texts to translate
            target_lang: Target language code
            source_lang: Optional source language code (auto-detect if None)

        Returns:
            List of TranslationResult objects

        Raises:
            MTError: If translation fails
            MTQuotaExceededError: If API quota is exceeded
        """
        ...

    @abstractmethod
    async def get_usage(self) -> dict[str, int]:
        """Get current API usage statistics.

        Returns:
            Dictionary with usage info (e.g., character_count, character_limit)
        """
        ...

    @abstractmethod
    async def get_supported_languages(self) -> list[dict[str, str]]:
        """Get list of supported languages.

        Returns:
            List of dictionaries with language info (code, name)
        """
        ...


class MTError(Exception):
    """Base exception for Machine Translation errors."""


class MTQuotaExceededError(MTError):
    """Raised when API quota is exceeded."""
