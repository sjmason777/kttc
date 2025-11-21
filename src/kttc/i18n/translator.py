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

"""Translation loading and lookup for CLI localization.

Loads translations from JSON files and provides lookup functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kttc.i18n.detector import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    detect_system_language,
)
from kttc.i18n.rtl import is_rtl_language, process_rtl_text

logger = logging.getLogger(__name__)

# Path to locale files
LOCALES_DIR = Path(__file__).parent / "locales"


class Translator:
    """Translation manager for CLI localization.

    Loads translations from JSON files and provides string lookup.
    Thread-safe singleton pattern.
    """

    _instance: Translator | None = None
    _initialized: bool = False

    def __new__(cls) -> Translator:
        """Singleton pattern - return existing instance or create new one."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize translator (only once due to singleton)."""
        if Translator._initialized:
            return

        self._language: str = DEFAULT_LANGUAGE
        self._translations: dict[str, dict[str, Any]] = {}
        self._fallback: dict[str, Any] = {}

        # Load English as fallback
        self._load_locale(DEFAULT_LANGUAGE)
        self._fallback = self._translations.get(DEFAULT_LANGUAGE, {})

        # Auto-detect and set language
        detected = detect_system_language()
        self.set_language(detected)

        Translator._initialized = True

    @property
    def language(self) -> str:
        """Get current language code."""
        return self._language

    @property
    def is_rtl(self) -> bool:
        """Check if current language is RTL."""
        return is_rtl_language(self._language)

    @property
    def language_name(self) -> str:
        """Get current language name."""
        translations = self._translations.get(self._language, {})
        meta = translations.get("_meta", {})
        name = meta.get("name", self._language.upper())
        return str(name)

    def set_language(self, lang: str) -> None:
        """Set current language.

        Args:
            lang: Language code (en, ru, zh, fa, hi) or "auto" for auto-detect
        """
        if lang == "auto":
            lang = detect_system_language()

        # Normalize and validate
        lang = lang.lower().split("_")[0].split("-")[0]

        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Language '%s' not supported, falling back to '%s'",
                lang,
                DEFAULT_LANGUAGE,
            )
            lang = DEFAULT_LANGUAGE

        # Load locale if not already loaded
        if lang not in self._translations:
            self._load_locale(lang)

        self._language = lang
        logger.debug("Language set to: %s", lang)

    def get(self, key: str, **kwargs: Any) -> str:
        """Get translated string by key.

        Args:
            key: Translation key (e.g., "check_header", "error_generic")
            **kwargs: Format arguments for string interpolation

        Returns:
            Translated string (with RTL processing if needed)
        """
        # Get translation from current language
        translations = self._translations.get(self._language, {})
        text = translations.get(key)

        # Fallback to English if not found
        if text is None:
            text = self._fallback.get(key)

        # If still not found, return key
        if text is None:
            logger.warning("Translation key not found: %s", key)
            return f"[{key}]"

        # Apply format arguments if provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError as e:
                logger.warning("Missing format argument %s for key '%s'", e, key)

        # Apply RTL processing if needed
        if self.is_rtl:
            text = process_rtl_text(text, self._language)

        return text

    def get_ai_language_instruction(self) -> str:
        """Get instruction for AI to respond in current language.

        Returns:
            Instruction string to append to AI prompts
        """
        return self.get("ai_response_language")

    def _load_locale(self, lang: str) -> None:
        """Load locale file for language.

        Args:
            lang: Language code
        """
        locale_file = LOCALES_DIR / f"{lang}.json"

        if not locale_file.exists():
            logger.warning("Locale file not found: %s", locale_file)
            return

        try:
            with open(locale_file, encoding="utf-8") as f:
                self._translations[lang] = json.load(f)
            logger.debug("Loaded locale: %s", lang)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse locale file %s: %s", locale_file, e)
        except OSError as e:
            logger.error("Failed to read locale file %s: %s", locale_file, e)

    def get_all_keys(self) -> set[str]:
        """Get all translation keys from fallback (English).

        Returns:
            Set of all translation keys
        """
        return {k for k in self._fallback if not k.startswith("_")}

    def validate_translations(self, lang: str) -> list[str]:
        """Validate that all keys are translated for a language.

        Args:
            lang: Language code to validate

        Returns:
            List of missing translation keys
        """
        if lang not in self._translations:
            self._load_locale(lang)

        if lang not in self._translations:
            return list(self.get_all_keys())

        translations = self._translations[lang]
        all_keys = self.get_all_keys()

        return [k for k in all_keys if k not in translations]


# Global translator instance
_translator: Translator | None = None


def get_translator() -> Translator:
    """Get global translator instance.

    Returns:
        Translator singleton instance
    """
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator
