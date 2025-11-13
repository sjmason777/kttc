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

    total_chars = max(cyrillic_chars + latin_chars + chinese_chars + arabic_chars, 1)

    # Determine dominant script
    if cyrillic_chars / total_chars > 0.3:
        return "ru"  # Russian
    elif chinese_chars / total_chars > 0.3:
        return "zh"  # Chinese
    elif arabic_chars / total_chars > 0.3:
        return "ar"  # Arabic
    elif latin_chars / total_chars > 0.3:
        # Could be English, Spanish, French, etc.
        # For now, default to English
        return "en"
    else:
        return "en"  # Default fallback


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
    language_code = language_code.lower()

    if language_code == "ru":
        from .russian import RussianLanguageHelper

        helper: LanguageHelper | None = RussianLanguageHelper()
        if helper and helper.is_available():
            logger.info(f"Using RussianLanguageHelper for language: {language_code}")
            return helper
        else:
            logger.warning(
                "RussianLanguageHelper dependencies not available. "
                "Install with: pip install mawo-pymorphy3 mawo-razdel mawo-natasha"
            )
            return None

    elif language_code == "en":
        from .english import EnglishLanguageHelper

        helper = EnglishLanguageHelper()
        if helper.is_available():
            logger.info(f"Using EnglishLanguageHelper for language: {language_code}")
            return helper
        else:
            logger.warning(
                "EnglishLanguageHelper dependencies not available. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
            return None

    elif language_code == "zh":
        from .chinese import ChineseLanguageHelper

        helper = ChineseLanguageHelper()
        if helper.is_available():
            logger.info(f"Using ChineseLanguageHelper for language: {language_code}")
            return helper
        else:
            logger.warning(
                "ChineseLanguageHelper dependencies not available. "
                "Install with: pip install jieba spacy && python -m spacy download zh_core_web_sm"
            )
            return None

    logger.debug(f"No language helper available for: {language_code}")
    return None


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
