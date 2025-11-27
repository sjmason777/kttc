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

"""System language detection for cross-platform i18n support.

Detects system language on Windows, Linux, and macOS.
"""

from __future__ import annotations

import locale
import os

# Supported languages
SUPPORTED_LANGUAGES = {"en", "ru", "zh", "fa", "hi"}
DEFAULT_LANGUAGE = "en"

# Windows LCID to language code mapping (common codes)
WINDOWS_LCID_MAP = {
    1033: "en",  # English - United States
    2057: "en",  # English - United Kingdom
    1049: "ru",  # Russian
    2052: "zh",  # Chinese - Simplified
    1028: "zh",  # Chinese - Traditional
    1065: "fa",  # Persian/Farsi
    1081: "hi",  # Hindi
}


def _get_supported_language(lang_string: str | None) -> str | None:
    """Normalize language string and return if supported.

    Args:
        lang_string: Language string to normalize, or None

    Returns:
        Supported language code or None
    """
    if not lang_string:
        return None
    lang = _normalize_language_code(lang_string)
    return lang if lang in SUPPORTED_LANGUAGES else None


def _detect_from_env_vars() -> str | None:
    """Try to detect language from environment variables.

    Returns:
        Supported language code or None
    """
    # Check KTTC-specific env var first
    result = _get_supported_language(os.environ.get("KTTC_UI_LANG"))
    if result:
        return result

    # Check standard locale environment variables
    for env_var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        result = _get_supported_language(os.environ.get(env_var))
        if result:
            return result

    return None


def _detect_from_locale() -> str | None:
    """Try to detect language from Python locale module.

    Returns:
        Supported language code or None
    """
    try:
        lang_locale, _ = locale.getlocale()
        return _get_supported_language(lang_locale)
    except (ValueError, TypeError):
        return None


def _detect_from_windows_api() -> str | None:
    """Try to detect language from Windows API.

    Returns:
        Supported language code or None
    """
    if os.name != "nt":
        return None
    try:
        win_lang = _detect_windows_language()
        return win_lang if win_lang in SUPPORTED_LANGUAGES else None
    except OSError:
        return None


def detect_system_language() -> str:
    """Detect system language (cross-platform).

    Priority:
    1. KTTC_UI_LANG environment variable
    2. LC_ALL, LC_MESSAGES, LANG environment variables
    3. locale.getdefaultlocale()
    4. Windows API (on Windows)
    5. Default to English

    Returns:
        Two-letter language code (en, ru, zh, fa, hi)
    """
    return (
        _detect_from_env_vars()
        or _detect_from_locale()
        or _detect_from_windows_api()
        or DEFAULT_LANGUAGE
    )


def _normalize_language_code(lang_string: str) -> str:
    """Normalize language string to two-letter code.

    Examples:
        "ru_RU.UTF-8" -> "ru"
        "en-US" -> "en"
        "zh_CN" -> "zh"
        "fa_IR" -> "fa"
        "hi_IN" -> "hi"

    Args:
        lang_string: Language string in various formats

    Returns:
        Two-letter language code
    """
    if not lang_string:
        return DEFAULT_LANGUAGE

    # Remove encoding suffix (e.g., ".UTF-8")
    lang = lang_string.split(".")[0]

    # Get first part before underscore or hyphen
    lang = lang.replace("-", "_").split("_")[0].lower()

    # Handle Chinese variants
    if lang in ("zh", "chinese"):
        return "zh"

    return lang


def _detect_windows_language() -> str | None:
    """Detect language using Windows API.

    Returns:
        Language code or None if detection failed
    """
    try:
        import ctypes

        windll = ctypes.windll.kernel32  # type: ignore[attr-defined]
        lang_id = windll.GetUserDefaultUILanguage()

        # Check direct mapping
        if lang_id in WINDOWS_LCID_MAP:
            return WINDOWS_LCID_MAP[lang_id]

        # Extract primary language ID (lower 10 bits)
        primary_lang_id = lang_id & 0x3FF

        # Common primary language IDs
        primary_lang_map = {
            0x09: "en",  # English
            0x19: "ru",  # Russian
            0x04: "zh",  # Chinese
            0x29: "fa",  # Persian
            0x39: "hi",  # Hindi
        }

        return primary_lang_map.get(primary_lang_id)

    except Exception:
        return None


def get_supported_languages() -> list[str]:
    """Get list of supported languages.

    Returns:
        List of supported language codes
    """
    return sorted(SUPPORTED_LANGUAGES)


def is_supported_language(lang: str) -> bool:
    """Check if language is supported.

    Args:
        lang: Language code to check

    Returns:
        True if language is supported
    """
    return _normalize_language_code(lang) in SUPPORTED_LANGUAGES
