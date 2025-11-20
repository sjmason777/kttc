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

"""Language Support Registry.

Centralized registry for supported languages based on FLORES-200.
Provides language capabilities, resource levels, and model recommendations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """Centralized language support registry.

    Based on FLORES-200 (200 languages) with categorization by resource level:
    - High-resource: Well-supported with specialized agents
    - Medium-resource: Good support with general agents
    - Low-resource: Basic support, may have lower quality

    Example:
        >>> registry = LanguageRegistry()
        >>> caps = registry.get_language_capabilities("ru")
        >>> print(caps["resource_level"])  # "high"
        >>> print(caps["has_specialized_agents"])  # True
    """

    # Language definitions (subset of FLORES-200 for common languages)
    SUPPORTED_LANGUAGES = {
        # High-resource languages (Tier 1)
        "en": {
            "name": "English",
            "native_name": "English",
            "resource_level": "high",
            "specialization": False,  # Base language
            "iso639_3": "eng",
        },
        "es": {
            "name": "Spanish",
            "native_name": "Español",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "spa",
        },
        "ru": {
            "name": "Russian",
            "native_name": "Русский",
            "resource_level": "high",
            "specialization": True,  # Has RussianFluencyAgent
            "iso639_3": "rus",
        },
        "zh": {
            "name": "Chinese",
            "native_name": "中文",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "zho",
        },
        "fr": {
            "name": "French",
            "native_name": "Français",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "fra",
        },
        "de": {
            "name": "German",
            "native_name": "Deutsch",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "deu",
        },
        "ja": {
            "name": "Japanese",
            "native_name": "日本語",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "jpn",
        },
        "ar": {
            "name": "Arabic",
            "native_name": "العربية",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "ara",
        },
        "hi": {
            "name": "Hindi",
            "native_name": "हिन्दी",
            "resource_level": "high",
            "specialization": True,  # Has HindiFluencyAgent
            "iso639_3": "hin",
        },
        "fa": {
            "name": "Persian",
            "native_name": "فارسی",
            "resource_level": "high",
            "specialization": True,  # Has PersianFluencyAgent
            "iso639_3": "fas",
        },
        "pt": {
            "name": "Portuguese",
            "native_name": "Português",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "por",
        },
        "it": {
            "name": "Italian",
            "native_name": "Italiano",
            "resource_level": "high",
            "specialization": False,
            "iso639_3": "ita",
        },
        # Medium-resource languages (Tier 2)
        "uk": {
            "name": "Ukrainian",
            "native_name": "Українська",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "ukr",
        },
        "pl": {
            "name": "Polish",
            "native_name": "Polski",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "pol",
        },
        "nl": {
            "name": "Dutch",
            "native_name": "Nederlands",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "nld",
        },
        "tr": {
            "name": "Turkish",
            "native_name": "Türkçe",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "tur",
        },
        "ko": {
            "name": "Korean",
            "native_name": "한국어",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "kor",
        },
        "vi": {
            "name": "Vietnamese",
            "native_name": "Tiếng Việt",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "vie",
        },
        "th": {
            "name": "Thai",
            "native_name": "ไทย",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "tha",
        },
        "id": {
            "name": "Indonesian",
            "native_name": "Bahasa Indonesia",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "ind",
        },
        "cs": {
            "name": "Czech",
            "native_name": "Čeština",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "ces",
        },
        "ro": {
            "name": "Romanian",
            "native_name": "Română",
            "resource_level": "medium",
            "specialization": False,
            "iso639_3": "ron",
        },
        # Low-resource languages (Tier 3) - Examples from FLORES+
        "be": {
            "name": "Belarusian",
            "native_name": "Беларуская",
            "resource_level": "low",
            "specialization": False,
            "iso639_3": "bel",
        },
        "ka": {
            "name": "Georgian",
            "native_name": "ქართული",
            "resource_level": "low",
            "specialization": False,
            "iso639_3": "kat",
        },
        "hy": {
            "name": "Armenian",
            "native_name": "Հայերեն",
            "resource_level": "low",
            "specialization": False,
            "iso639_3": "hye",
        },
    }

    def get_language_capabilities(self, lang_code: str) -> dict[str, Any]:
        """Get capabilities and information for a language.

        Args:
            lang_code: ISO 639-1 language code (e.g., "ru", "en")

        Returns:
            Dictionary with language capabilities

        Example:
            >>> registry = LanguageRegistry()
            >>> caps = registry.get_language_capabilities("ru")
            >>> print(caps["name"])  # "Russian"
            >>> print(caps["supported"])  # True
        """
        if lang_code not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{lang_code}' not in registry")
            return {
                "supported": False,
                "lang_code": lang_code,
                "message": f"Language '{lang_code}' not in FLORES-200 registry. "
                "Quality may be lower for unsupported languages.",
            }

        lang = self.SUPPORTED_LANGUAGES[lang_code]
        resource_level = str(lang["resource_level"])

        return {
            "supported": True,
            "lang_code": lang_code,
            "name": lang["name"],
            "native_name": lang["native_name"],
            "resource_level": resource_level,
            "has_specialized_agents": lang["specialization"],
            "recommended_model": self._recommend_model(lang_code, resource_level),
            "iso639_3": lang["iso639_3"],
        }

    def _recommend_model(self, lang_code: str, resource_level: str) -> str:
        """Recommend best LLM model for language.

        Args:
            lang_code: Language code
            resource_level: Resource level (high/medium/low)

        Returns:
            Recommended model identifier
        """
        # Special cases
        if lang_code == "ru":
            return "yandexgpt"  # Best for Russian
        elif lang_code == "zh":
            return "gpt-4.5"  # Strong Chinese support
        elif lang_code == "ja":
            return "claude-3.5-sonnet"  # Good Japanese support
        elif lang_code == "hi":
            return "claude-3.5-sonnet"  # Good Hindi support
        elif lang_code == "fa":
            return "claude-3.5-sonnet"  # Good Persian support

        # General recommendations by resource level
        if resource_level == "low":
            # Low-resource: Use models with better multilingual coverage
            return "gemini-2.0"  # Google's model has broad language support
        else:
            # High/medium-resource: Use best overall model
            return "claude-3.5-sonnet"

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if language is supported.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            True if language is in registry
        """
        return lang_code in self.SUPPORTED_LANGUAGES

    def get_all_languages(self) -> list[dict[str, Any]]:
        """Get list of all supported languages.

        Returns:
            List of language information dictionaries

        Example:
            >>> registry = LanguageRegistry()
            >>> languages = registry.get_all_languages()
            >>> high_resource = [l for l in languages if l["resource_level"] == "high"]
            >>> print(f"High-resource languages: {len(high_resource)}")
        """
        return [
            {"lang_code": code, **lang_info} for code, lang_info in self.SUPPORTED_LANGUAGES.items()
        ]

    def get_languages_by_resource_level(self, level: str) -> list[str]:
        """Get languages filtered by resource level.

        Args:
            level: Resource level - "high", "medium", or "low"

        Returns:
            List of language codes

        Example:
            >>> registry = LanguageRegistry()
            >>> high_langs = registry.get_languages_by_resource_level("high")
            >>> print(high_langs)  # ['en', 'es', 'ru', ...]
        """
        return [
            code
            for code, info in self.SUPPORTED_LANGUAGES.items()
            if info["resource_level"] == level
        ]

    def get_language_pairs(self) -> list[tuple[str, str]]:
        """Get all possible language pairs from registry.

        Returns:
            List of (source, target) language code tuples

        Note:
            This generates all combinations. In practice, not all pairs
            have equal quality. Use ModelSelector for quality information.
        """
        lang_codes = list(self.SUPPORTED_LANGUAGES.keys())
        pairs = []

        for source in lang_codes:
            for target in lang_codes:
                if source != target:
                    pairs.append((source, target))

        return pairs

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with statistics

        Example:
            >>> registry = LanguageRegistry()
            >>> stats = registry.get_statistics()
            >>> print(f"Total languages: {stats['total_languages']}")
        """
        by_level = {}
        for level in ["high", "medium", "low"]:
            by_level[level] = len(self.get_languages_by_resource_level(level))

        specialized = sum(1 for info in self.SUPPORTED_LANGUAGES.values() if info["specialization"])

        return {
            "total_languages": len(self.SUPPORTED_LANGUAGES),
            "by_resource_level": by_level,
            "with_specialization": specialized,
            "total_possible_pairs": len(self.get_language_pairs()),
        }


# Global registry instance
_registry: LanguageRegistry | None = None


def get_language_registry() -> LanguageRegistry:
    """Get global LanguageRegistry instance.

    Returns:
        Singleton LanguageRegistry instance

    Example:
        >>> registry = get_language_registry()
        >>> caps = registry.get_language_capabilities("en")
    """
    global _registry
    if _registry is None:
        _registry = LanguageRegistry()
    return _registry
