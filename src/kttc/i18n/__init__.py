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

"""Internationalization (i18n) module for KTTC CLI.

Provides multilingual support for CLI output in English, Russian,
Chinese, Persian, and Hindi.

Usage:
    from kttc.i18n import _, set_language, get_language

    # Get translated string
    print(_("check_header"))  # "Translation Quality Check" or localized

    # With format arguments
    print(_("error_count").format(count=5))

    # Set language explicitly
    set_language("ru")  # Russian
    set_language("auto")  # Auto-detect from system

    # Get current language
    lang = get_language()  # "en", "ru", "zh", "fa", "hi"
"""

from __future__ import annotations

from typing import Any

from kttc.i18n.detector import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    detect_system_language,
    get_supported_languages,
    is_supported_language,
)
from kttc.i18n.rtl import is_rtl_language, process_rtl_text
from kttc.i18n.translator import Translator, get_translator


def _(key: str, **kwargs: Any) -> str:
    """Get translated string by key.

    This is the main function for getting localized strings.
    Use `.format()` for string interpolation.

    Args:
        key: Translation key (e.g., "check_header", "error_generic")
        **kwargs: Format arguments for string interpolation

    Returns:
        Translated string

    Examples:
        >>> _("check_header")
        "Translation Quality Check"

        >>> _("error_count", count=5)
        "Errors: 5"
    """
    return get_translator().get(key, **kwargs)


def set_language(lang: str) -> None:
    """Set UI language for CLI output.

    Args:
        lang: Language code or "auto" for auto-detection
            - "en" - English (default)
            - "ru" - Russian
            - "zh" - Chinese (Simplified)
            - "fa" - Persian/Farsi
            - "hi" - Hindi
            - "auto" - Auto-detect from system
    """
    get_translator().set_language(lang)


def get_language() -> str:
    """Get current UI language code.

    Returns:
        Two-letter language code (en, ru, zh, fa, hi)
    """
    return get_translator().language


def get_language_name() -> str:
    """Get current language name in native script.

    Returns:
        Language name (e.g., "English", "Русский", "中文")
    """
    return get_translator().language_name


def is_rtl() -> bool:
    """Check if current language is right-to-left.

    Returns:
        True if current language is RTL (Persian)
    """
    return get_translator().is_rtl


def get_ai_language_instruction() -> str:
    """Get instruction for AI to respond in current language.

    Returns:
        Instruction string to append to AI prompts
    """
    return get_translator().get_ai_language_instruction()


__all__ = [
    # Main translation function
    "_",
    # Language management
    "set_language",
    "get_language",
    "get_language_name",
    "is_rtl",
    # AI integration
    "get_ai_language_instruction",
    # Detection utilities
    "detect_system_language",
    "get_supported_languages",
    "is_supported_language",
    # RTL utilities
    "is_rtl_language",
    "process_rtl_text",
    # Constants
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    # Classes
    "Translator",
    "get_translator",
]
