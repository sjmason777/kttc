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

"""Right-to-Left (RTL) text processing for Persian language.

Uses python-bidi and arabic-reshaper for proper RTL display in terminals.
"""

from __future__ import annotations

# RTL languages
RTL_LANGUAGES = {"fa", "ar", "he"}


def is_rtl_language(lang: str) -> bool:
    """Check if language is RTL.

    Args:
        lang: Two-letter language code

    Returns:
        True if language is RTL
    """
    return lang.lower() in RTL_LANGUAGES


def process_rtl_text(text: str, lang: str) -> str:
    """Process text for RTL display in terminal.

    For RTL languages (Persian, Arabic, Hebrew), this function:
    1. Reshapes Arabic/Persian characters to correct form
    2. Applies Unicode Bidirectional Algorithm for proper display

    Args:
        text: Text to process
        lang: Language code

    Returns:
        Processed text suitable for terminal display
    """
    if not is_rtl_language(lang):
        return text

    if not text:
        return text

    try:
        # Try to use arabic-reshaper for proper character shaping
        try:
            import arabic_reshaper  # type: ignore[import-not-found]

            text = arabic_reshaper.reshape(text)
        except ImportError:
            # Silently ignore missing arabic-reshaper dependency
            pass

        # Apply Unicode Bidirectional Algorithm
        try:
            from bidi.algorithm import get_display  # type: ignore[import-not-found]

            text = get_display(text)
        except ImportError:
            # Silently ignore missing python-bidi dependency
            pass

        return text

    except Exception:
        # If processing fails, return original text
        return text


def get_rtl_marker() -> str:
    """Get Unicode RTL marker for embedding RTL text.

    Returns:
        Unicode Right-to-Left Isolate character
    """
    return "\u2067"  # RIGHT-TO-LEFT ISOLATE


def get_ltr_marker() -> str:
    """Get Unicode LTR marker for embedding LTR text.

    Returns:
        Unicode Left-to-Right Isolate character
    """
    return "\u2066"  # LEFT-TO-RIGHT ISOLATE


def get_pop_marker() -> str:
    """Get Unicode Pop Directional Isolate marker.

    Returns:
        Unicode Pop Directional Isolate character
    """
    return "\u2069"  # POP DIRECTIONAL ISOLATE


def wrap_rtl(text: str) -> str:
    """Wrap text with RTL markers.

    Args:
        text: Text to wrap

    Returns:
        Text wrapped with RTL isolate markers
    """
    return f"{get_rtl_marker()}{text}{get_pop_marker()}"


def wrap_ltr(text: str) -> str:
    """Wrap text with LTR markers.

    Args:
        text: Text to wrap

    Returns:
        Text wrapped with LTR isolate markers
    """
    return f"{get_ltr_marker()}{text}{get_pop_marker()}"
