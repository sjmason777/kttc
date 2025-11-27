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

"""Language detection and helper selection."""

from __future__ import annotations

import logging
import re

from .base import LanguageHelper

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language of text using character-based heuristics.

    This is a simple, fast detection method that works without external libraries.
    For production, could be replaced with langdetect or similar.

    Args:
        text: Text to analyze

    Returns:
        ISO 639-1 language code ('ru', 'en', 'zh', etc.)

    Example:
        >>> detect_language("Привет, мир!")
        'ru'
        >>> detect_language("Hello, world!")
        'en'
    """
    if not text or not text.strip():
        return "en"  # Default fallback

    # Character range detection
    cyrillic_chars = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    arabic_chars = len(re.findall(r"[\u0600-\u06ff]", text))
    devanagari_chars = len(re.findall(r"[\u0900-\u097f]", text))  # Hindi/Sanskrit
    persian_chars = len(re.findall(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufefc]", text))

    total_chars = max(
        cyrillic_chars
        + latin_chars
        + chinese_chars
        + arabic_chars
        + devanagari_chars
        + persian_chars,
        1,
    )

    # Determine dominant script
    if cyrillic_chars / total_chars > 0.3:
        return "ru"  # Russian
    if chinese_chars / total_chars > 0.3:
        return "zh"  # Chinese
    if devanagari_chars / total_chars > 0.3:
        return "hi"  # Hindi
    if persian_chars / total_chars > 0.3:
        # Distinguish Persian from Arabic by common Persian characters
        if len(re.findall(r"[پچژگ]", text)) > 0:  # Persian-specific letters
            return "fa"  # Persian
        return "ar"  # Arabic
    if arabic_chars / total_chars > 0.3:
        return "ar"  # Arabic
    if latin_chars / total_chars > 0.3:
        # Could be English, Spanish, French, etc.
        # For now, default to English
        return "en"
    return "en"  # Default fallback


def _try_get_helper(
    helper_class: type[LanguageHelper],
    helper_name: str,
    language_code: str,
    install_hint: str,
) -> LanguageHelper | None:
    """Try to instantiate and validate a language helper.

    Args:
        helper_class: The helper class to instantiate
        helper_name: Name for logging
        language_code: Language code for logging
        install_hint: Installation instructions if unavailable

    Returns:
        Helper instance if available, None otherwise
    """
    helper: LanguageHelper = helper_class()
    if helper.is_available():
        logger.info(f"Using {helper_name} for language: {language_code}")
        return helper
    logger.warning(f"{helper_name} dependencies not available. {install_hint}")
    return None


# Language helper configuration: (module, class_name, install_hint)
_HELPER_CONFIG: dict[str, tuple[str, str, str]] = {
    "ru": (
        ".russian",
        "RussianLanguageHelper",
        "Install with: pip install mawo-pymorphy3 mawo-razdel mawo-natasha",
    ),
    "en": (
        ".english",
        "EnglishLanguageHelper",
        "Install with: pip install spacy && python -m spacy download en_core_web_sm",
    ),
    "zh": (
        ".chinese",
        "ChineseLanguageHelper",
        "Install with: pip install jieba spacy && python -m spacy download zh_core_web_sm",
    ),
    "hi": (
        ".hindi",
        "HindiLanguageHelper",
        'Install with: pip install "kttc[hindi]" or pip install indic-nlp-library stanza spello',
    ),
    "fa": (
        ".persian",
        "PersianLanguageHelper",
        'Install with: pip install "kttc[persian]" or pip install "dadmatools[full]"',
    ),
}


def get_helper_for_language(language_code: str) -> LanguageHelper | None:
    """Get appropriate language helper for given language code.

    Args:
        language_code: ISO 639-1 code ('ru', 'en', 'zh', etc.)

    Returns:
        LanguageHelper instance if available, None otherwise

    Example:
        >>> helper = get_helper_for_language('ru')
        >>> if helper and helper.is_available():
        ...     errors = helper.check_grammar(text)
    """
    import importlib

    language_code = language_code.lower()

    if language_code not in _HELPER_CONFIG:
        logger.debug(f"No language helper available for: {language_code}")
        return None

    module_name, class_name, install_hint = _HELPER_CONFIG[language_code]
    module = importlib.import_module(module_name, package="kttc.helpers")
    helper_class = getattr(module, class_name)

    return _try_get_helper(helper_class, class_name, language_code, install_hint)


def get_helper_from_text(text: str) -> LanguageHelper | None:
    """Auto-detect language and get appropriate helper.

    Args:
        text: Text to detect language from

    Returns:
        LanguageHelper if available, None otherwise

    Example:
        >>> helper = get_helper_from_text("Привет, мир!")
        >>> helper.language_code
        'ru'
    """
    lang_code = detect_language(text)
    return get_helper_for_language(lang_code)
