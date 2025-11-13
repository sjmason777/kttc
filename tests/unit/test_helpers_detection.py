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

"""Comprehensive tests for language detection."""

import pytest

from kttc.helpers.detection import (
    detect_language,
    get_helper_for_language,
    get_helper_from_text,
)


class TestDetectLanguage:
    """Test suite for language detection."""

    def test_detect_russian(self):
        """Test Russian language detection."""
        assert detect_language("Привет, мир!") == "ru"
        assert detect_language("Быстрая коричневая лиса") == "ru"
        assert detect_language("Москва - столица России") == "ru"

    def test_detect_english(self):
        """Test English language detection."""
        assert detect_language("Hello, world!") == "en"
        assert detect_language("The quick brown fox") == "en"
        assert detect_language("New York is a city") == "en"

    def test_detect_chinese(self):
        """Test Chinese language detection."""
        assert detect_language("你好世界") == "zh"
        assert detect_language("我爱中文") == "zh"
        assert detect_language("苹果公司首席执行官") == "zh"

    def test_detect_arabic(self):
        """Test Arabic language detection."""
        assert detect_language("مرحبا بالعالم") == "ar"
        assert detect_language("السلام عليكم") == "ar"

    def test_detect_empty_string(self):
        """Test detection with empty string."""
        assert detect_language("") == "en"  # Default fallback

    def test_detect_whitespace_only(self):
        """Test detection with whitespace only."""
        assert detect_language("   \n\t  ") == "en"  # Default fallback

    def test_detect_mixed_russian_english(self):
        """Test detection with mixed Russian and English."""
        # Majority Russian (Привет = 6 cyrillic, "from Moscow" = 11 latin, 6/17 = 35% > 30%)
        assert detect_language("Привет from Moscow") == "ru"
        # Close to threshold - Москва = 6 cyrillic, "Hello from " = 11 latin, 6/17 = 35% > 30%
        # So this is actually Russian, not English
        assert detect_language("Hello from Москва") == "ru"
        # Clearly majority English (привет = 6 cyrillic, "Hello world everyone" = 19 latin, 6/25 = 24% < 30%)
        assert detect_language("Hello world everyone привет") == "en"

    def test_detect_mixed_chinese_english(self):
        """Test detection with mixed Chinese and English."""
        # Majority Chinese (我爱 = 2 chinese, "Python programming" = 18 latin, 2/20 = 10% < 30%)
        # So this is actually English
        assert detect_language("我爱 Python programming") == "en"
        # Majority English (是很好 = 3 chinese, "Python programming " = 19 latin, 3/22 = 13% < 30%)
        assert detect_language("Python programming 是很好") == "en"
        # Clearly majority Chinese
        assert detect_language("我爱中文编程 Python") == "zh"

    def test_detect_numbers_only(self):
        """Test detection with numbers only."""
        assert detect_language("12345") == "en"  # Default fallback

    def test_detect_punctuation_only(self):
        """Test detection with punctuation only."""
        assert detect_language(".,!?;:") == "en"  # Default fallback

    def test_detect_with_special_characters(self):
        """Test detection with special characters."""
        assert detect_language("Привет @user #tag") == "ru"
        assert detect_language("Hello @user #tag") == "en"

    def test_detect_threshold_edge_cases(self):
        """Test detection at threshold boundaries."""
        # Exactly 30% Cyrillic
        text_30_percent = "абв" + "x" * 7  # 3 cyrillic, 7 latin = 30%
        result = detect_language(text_30_percent)
        assert result in ["ru", "en"]  # Either is acceptable at boundary

    def test_detect_case_insensitive(self):
        """Test that detection works with different cases."""
        assert detect_language("ПРИВЕТ МИР") == "ru"
        assert detect_language("HELLO WORLD") == "en"
        assert detect_language("ПрИвЕт мИр") == "ru"

    def test_detect_with_yo_letter(self):
        """Test Russian detection with ё letter."""
        assert detect_language("Ёлка и ежик") == "ru"

    def test_detect_traditional_chinese(self):
        """Test Chinese detection with traditional characters."""
        assert detect_language("我愛繁體中文") == "zh"

    def test_detect_very_short_text(self):
        """Test detection with very short text."""
        assert detect_language("Я") == "ru"
        assert detect_language("I") == "en"
        assert detect_language("我") == "zh"

    def test_detect_very_long_text(self):
        """Test detection with very long text."""
        long_russian = "Привет, мир! " * 100
        assert detect_language(long_russian) == "ru"

        long_english = "Hello, world! " * 100
        assert detect_language(long_english) == "en"


class TestGetHelperForLanguage:
    """Test suite for get_helper_for_language."""

    def test_get_russian_helper(self):
        """Test getting Russian helper."""
        helper = get_helper_for_language("ru")

        if helper is None:
            pytest.skip("Russian helper dependencies not available")

        assert helper.language_code == "ru"
        assert helper.is_available()

    def test_get_english_helper(self):
        """Test getting English helper."""
        helper = get_helper_for_language("en")

        if helper is None:
            pytest.skip("English helper dependencies not available")

        assert helper.language_code == "en"
        assert helper.is_available()

    def test_get_chinese_helper(self):
        """Test getting Chinese helper."""
        helper = get_helper_for_language("zh")

        if helper is None:
            pytest.skip("Chinese helper dependencies not available")

        assert helper.language_code == "zh"
        assert helper.is_available()

    def test_get_helper_uppercase_code(self):
        """Test that language code is case-insensitive."""
        helper_lower = get_helper_for_language("ru")
        helper_upper = get_helper_for_language("RU")
        helper_mixed = get_helper_for_language("Ru")

        # All should return same type of helper or all None
        if helper_lower:
            assert helper_upper is not None
            assert helper_mixed is not None
            assert helper_lower.language_code == helper_upper.language_code
            assert helper_lower.language_code == helper_mixed.language_code

    def test_get_helper_unsupported_language(self):
        """Test getting helper for unsupported language."""
        helper = get_helper_for_language("fr")  # French not supported yet
        assert helper is None

        helper = get_helper_for_language("de")  # German not supported yet
        assert helper is None

        helper = get_helper_for_language("xx")  # Invalid code
        assert helper is None

    def test_get_helper_empty_code(self):
        """Test getting helper with empty code."""
        helper = get_helper_for_language("")
        assert helper is None

    def test_get_helper_returns_different_instances(self):
        """Test that each call returns a new instance."""
        helper1 = get_helper_for_language("en")
        helper2 = get_helper_for_language("en")

        if helper1 and helper2:
            # Should be different instances
            assert helper1 is not helper2

    def test_get_helper_availability_consistent(self):
        """Test that helper availability is consistent."""
        helper1 = get_helper_for_language("ru")
        helper2 = get_helper_for_language("ru")

        # Both should have same availability
        if helper1:
            assert helper2 is not None
            assert helper1.is_available() == helper2.is_available()


class TestGetHelperFromText:
    """Test suite for get_helper_from_text."""

    def test_get_helper_from_russian_text(self):
        """Test getting helper from Russian text."""
        helper = get_helper_from_text("Привет, мир!")

        if helper is None:
            pytest.skip("Russian helper dependencies not available")

        assert helper.language_code == "ru"

    def test_get_helper_from_english_text(self):
        """Test getting helper from English text."""
        helper = get_helper_from_text("Hello, world!")

        if helper is None:
            pytest.skip("English helper dependencies not available")

        assert helper.language_code == "en"

    def test_get_helper_from_chinese_text(self):
        """Test getting helper from Chinese text."""
        helper = get_helper_from_text("你好世界")

        if helper is None:
            pytest.skip("Chinese helper dependencies not available")

        assert helper.language_code == "zh"

    def test_get_helper_from_empty_text(self):
        """Test getting helper from empty text."""
        helper = get_helper_from_text("")

        # Should default to English
        if helper:
            assert helper.language_code == "en"

    def test_get_helper_from_mixed_text(self):
        """Test getting helper from mixed language text."""
        # Majority Russian (Привет = 6 cyrillic, "from Moscow" = 11 latin, 6/17 = 35% > 30%)
        helper = get_helper_from_text("Привет from Moscow")
        if helper:
            assert helper.language_code == "ru"

        # Also Russian (Москва = 6 cyrillic, "Hello from " = 11 latin, 6/17 = 35% > 30%)
        helper = get_helper_from_text("Hello from Москва")
        if helper:
            assert helper.language_code == "ru"

        # Clearly English (привет = 6 cyrillic, "Hello world everyone" = 19 latin, 6/25 = 24% < 30%)
        helper = get_helper_from_text("Hello world everyone привет")
        if helper:
            assert helper.language_code == "en"

    def test_get_helper_integration(self):
        """Test full integration: detect language and use helper."""
        texts = {
            "ru": "Быстрая лиса прыгает",
            "en": "The quick fox jumps",
            "zh": "我爱中文编程",
        }

        for expected_lang, text in texts.items():
            # Detect language
            detected_lang = detect_language(text)
            assert detected_lang == expected_lang

            # Get helper
            helper = get_helper_for_language(detected_lang)

            if helper:
                # Use helper
                tokens = helper.tokenize(text)
                assert len(tokens) > 0

                # Verify all positions are correct
                for word, start, end in tokens:
                    assert text[start:end] == word

    def test_all_supported_languages_have_helpers(self):
        """Test that all detected languages have corresponding helpers."""
        test_texts = {
            "Привет, мир!": "ru",
            "Hello, world!": "en",
            "你好世界": "zh",
        }

        for text, expected_lang in test_texts.items():
            detected = detect_language(text)
            assert detected == expected_lang

            # Should be able to get helper (or None if not installed)
            helper = get_helper_for_language(detected)
            # Just verify it doesn't crash
            assert helper is None or helper.language_code == expected_lang

    def test_helper_can_process_detected_language(self):
        """Test that helper can actually process text it was selected for."""
        texts = [
            "Привет, мир!",
            "Hello, world!",
            "你好世界",
        ]

        for text in texts:
            helper = get_helper_from_text(text)

            if helper and helper.is_available():
                # Helper should be able to tokenize its own language
                tokens = helper.tokenize(text)
                assert isinstance(tokens, list)

                # Verify positions
                for word, start, end in tokens:
                    assert text[start:end] == word, f"Position error in {helper.language_code}"

    def test_detection_stability(self):
        """Test that detection is stable (same result for same input)."""
        text = "Привет, мир!"

        results = [detect_language(text) for _ in range(5)]

        # All results should be identical
        assert len(set(results)) == 1, "Detection should be deterministic"

    def test_helper_selection_stability(self):
        """Test that helper selection is stable."""
        text = "Hello, world!"

        helpers = [get_helper_from_text(text) for _ in range(5)]

        # All should return same type of helper (or all None)
        if helpers[0]:
            assert all(h is not None for h in helpers)
            assert all(h.language_code == helpers[0].language_code for h in helpers)
        else:
            assert all(h is None for h in helpers)
