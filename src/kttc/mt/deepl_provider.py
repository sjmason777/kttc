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

"""DeepL Machine Translation provider implementation.

Uses DeepL REST API via aiohttp for lightweight, async operation.
Supports both free and pro API endpoints.

API Documentation: https://developers.deepl.com/docs

Supported languages (as of 2025):
- Source: AR, BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HU, ID, IT, JA, KO,
          LT, LV, NB, NL, PL, PT, RO, RU, SK, SL, SV, TR, UK, ZH
- Target: Same as source, plus regional variants like EN-GB, EN-US, PT-BR, PT-PT
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from .base import BaseMTProvider, MTError, MTQuotaExceededError, TranslationResult

logger = logging.getLogger(__name__)

# DeepL API endpoints
DEEPL_API_FREE = "https://api-free.deepl.com/v2"
DEEPL_API_PRO = "https://api.deepl.com/v2"


class DeepLProvider(BaseMTProvider):
    """DeepL Machine Translation provider.

    Uses DeepL REST API for high-quality translations.
    Supports both free (500k chars/month) and pro plans.

    Example:
        >>> provider = DeepLProvider(api_key="your-api-key")
        >>> result = await provider.translate("Hello world", target_lang="DE")
        >>> print(result.text)
        'Hallo Welt'
    """

    # Language code mapping (DeepL uses specific codes)
    LANGUAGE_MAP = {
        "en": "EN",
        "de": "DE",
        "fr": "FR",
        "es": "ES",
        "it": "IT",
        "nl": "NL",
        "pl": "PL",
        "ru": "RU",
        "ja": "JA",
        "zh": "ZH",
        "pt": "PT",
        "ko": "KO",
        "ar": "AR",
        "tr": "TR",
        "uk": "UK",
        "cs": "CS",
        "da": "DA",
        "fi": "FI",
        "hu": "HU",
        "id": "ID",
        "nb": "NB",
        "no": "NB",  # Norwegian maps to NB (Bokmål)
        "sv": "SV",
        "el": "EL",
        "ro": "RO",
        "sk": "SK",
        "bg": "BG",
        "et": "ET",
        "lt": "LT",
        "lv": "LV",
        "sl": "SL",
    }

    def __init__(
        self,
        api_key: str,
        use_free_api: bool = True,
        timeout: float = 30.0,
    ):
        """Initialize DeepL provider.

        Args:
            api_key: DeepL API authentication key
            use_free_api: Whether to use free API endpoint (default: True)
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = DEEPL_API_FREE if use_free_api else DEEPL_API_PRO
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    "Authorization": f"DeepL-Auth-Key {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _normalize_language(self, lang: str) -> str:
        """Normalize language code to DeepL format.

        Args:
            lang: Language code (e.g., "en", "EN", "en-US")

        Returns:
            DeepL-compatible language code
        """
        # Extract base language code
        base_lang = lang.lower().split("-")[0].split("_")[0]
        return self.LANGUAGE_MAP.get(base_lang, lang.upper())

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
    ) -> TranslationResult:
        """Translate text using DeepL API.

        Args:
            text: Text to translate
            target_lang: Target language code
            source_lang: Optional source language (auto-detect if None)

        Returns:
            TranslationResult with translated text

        Raises:
            MTError: If translation fails
            MTQuotaExceededError: If quota exceeded
        """
        results = await self.translate_batch([text], target_lang, source_lang)
        return results[0]

    async def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslationResult]:
        """Translate multiple texts using DeepL API.

        Args:
            texts: List of texts to translate
            target_lang: Target language code
            source_lang: Optional source language (auto-detect if None)

        Returns:
            List of TranslationResult objects

        Raises:
            MTError: If translation fails
            MTQuotaExceededError: If quota exceeded
        """
        if not texts:
            return []

        session = await self._get_session()
        url = f"{self.base_url}/translate"

        # Build request body
        body: dict[str, Any] = {
            "text": texts,
            "target_lang": self._normalize_language(target_lang),
        }

        if source_lang:
            body["source_lang"] = self._normalize_language(source_lang)

        try:
            async with session.post(url, json=body) as response:
                if response.status == 403:
                    raise MTError("DeepL authentication failed: Invalid API key")
                if response.status == 456:
                    raise MTQuotaExceededError("DeepL quota exceeded")
                if response.status >= 400:
                    error_text = await response.text()
                    raise MTError(f"DeepL API error ({response.status}): {error_text}")

                data = await response.json()

                results = []
                translations = data.get("translations", [])

                for i, trans in enumerate(translations):
                    translated_text = trans.get("text", "")
                    detected_lang = trans.get("detected_source_language", source_lang or "")
                    char_count = len(texts[i]) if i < len(texts) else 0

                    self.total_characters += char_count

                    results.append(
                        TranslationResult(
                            text=translated_text,
                            source_lang=detected_lang,
                            target_lang=target_lang,
                            characters=char_count,
                        )
                    )

                return results

        except aiohttp.ClientError as e:
            raise MTError(f"DeepL connection error: {e}") from e

    async def get_usage(self) -> dict[str, int]:
        """Get current API usage from DeepL.

        Returns:
            Dictionary with character_count and character_limit
        """
        session = await self._get_session()
        url = f"{self.base_url}/usage"

        try:
            async with session.get(url) as response:
                if response.status >= 400:
                    raise MTError(f"Failed to get usage: {response.status}")

                data = await response.json()
                return {
                    "character_count": data.get("character_count", 0),
                    "character_limit": data.get("character_limit", 0),
                }

        except aiohttp.ClientError as e:
            raise MTError(f"DeepL connection error: {e}") from e

    async def get_supported_languages(self) -> list[dict[str, str]]:
        """Get list of supported languages from DeepL.

        Returns:
            List of dictionaries with 'language' and 'name' keys
        """
        session = await self._get_session()
        url = f"{self.base_url}/languages"

        try:
            async with session.get(url) as response:
                if response.status >= 400:
                    raise MTError(f"Failed to get languages: {response.status}")

                data = await response.json()
                return [
                    {"code": lang.get("language", ""), "name": lang.get("name", "")}
                    for lang in data
                ]

        except aiohttp.ClientError as e:
            raise MTError(f"DeepL connection error: {e}") from e

    def __del__(self) -> None:
        """Cleanup: warn if session not properly closed."""
        if self._session and not self._session.closed:
            logger.warning(
                "DeepLProvider session not properly closed. Use 'await provider.close()'"
            )
