"""Tests for Language Registry.

Comprehensive tests for language support registry functionality.
"""

import pytest

from kttc.utils.languages import LanguageRegistry, get_language_registry


class TestLanguageRegistry:
    """Test LanguageRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create LanguageRegistry instance."""
        return LanguageRegistry()

    def test_get_supported_language_russian(self, registry):
        """Test getting capabilities for supported language (Russian)."""
        caps = registry.get_language_capabilities("ru")

        assert caps["supported"] is True
        assert caps["lang_code"] == "ru"
        assert caps["name"] == "Russian"
        assert caps["native_name"] == "Русский"
        assert caps["resource_level"] == "high"
        assert caps["has_specialized_agents"] is True
        assert caps["recommended_model"] == "yandexgpt"
        assert caps["iso639_3"] == "rus"

    def test_get_supported_language_english(self, registry):
        """Test getting capabilities for English."""
        caps = registry.get_language_capabilities("en")

        assert caps["supported"] is True
        assert caps["lang_code"] == "en"
        assert caps["name"] == "English"
        assert caps["resource_level"] == "high"
        assert caps["has_specialized_agents"] is False

    def test_get_unsupported_language(self, registry):
        """Test getting capabilities for unsupported language."""
        caps = registry.get_language_capabilities("xx")

        assert caps["supported"] is False
        assert caps["lang_code"] == "xx"
        assert "message" in caps
        assert "not in FLORES-200 registry" in caps["message"]

    def test_is_language_supported_true(self, registry):
        """Test language support check for supported language."""
        assert registry.is_language_supported("en") is True
        assert registry.is_language_supported("ru") is True
        assert registry.is_language_supported("es") is True

    def test_is_language_supported_false(self, registry):
        """Test language support check for unsupported language."""
        assert registry.is_language_supported("xx") is False
        assert registry.is_language_supported("zz") is False

    def test_get_all_languages(self, registry):
        """Test getting all supported languages."""
        languages = registry.get_all_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert all("lang_code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)
        assert all("resource_level" in lang for lang in languages)

    def test_get_languages_by_resource_level_high(self, registry):
        """Test filtering languages by high resource level."""
        high_langs = registry.get_languages_by_resource_level("high")

        assert isinstance(high_langs, list)
        assert "en" in high_langs
        assert "ru" in high_langs
        assert "es" in high_langs
        # Should not contain medium/low resource languages
        for lang in high_langs:
            assert registry.SUPPORTED_LANGUAGES[lang]["resource_level"] == "high"

    def test_get_languages_by_resource_level_medium(self, registry):
        """Test filtering languages by medium resource level."""
        medium_langs = registry.get_languages_by_resource_level("medium")

        assert isinstance(medium_langs, list)
        assert len(medium_langs) > 0
        for lang in medium_langs:
            assert registry.SUPPORTED_LANGUAGES[lang]["resource_level"] == "medium"

    def test_get_languages_by_resource_level_low(self, registry):
        """Test filtering languages by low resource level."""
        low_langs = registry.get_languages_by_resource_level("low")

        assert isinstance(low_langs, list)
        assert len(low_langs) > 0
        for lang in low_langs:
            assert registry.SUPPORTED_LANGUAGES[lang]["resource_level"] == "low"

    def test_get_language_pairs(self, registry):
        """Test getting all language pairs."""
        pairs = registry.get_language_pairs()

        assert isinstance(pairs, list)
        assert len(pairs) > 0
        # Each pair should be a tuple of two different language codes
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs)
        assert all(pair[0] != pair[1] for pair in pairs)
        # Should have combinations (not including self-pairs)
        total_langs = len(registry.SUPPORTED_LANGUAGES)
        expected_pairs = total_langs * (total_langs - 1)
        assert len(pairs) == expected_pairs

    def test_get_statistics(self, registry):
        """Test getting registry statistics."""
        stats = registry.get_statistics()

        assert "total_languages" in stats
        assert "by_resource_level" in stats
        assert "with_specialization" in stats
        assert "total_possible_pairs" in stats

        assert stats["total_languages"] > 0
        assert isinstance(stats["by_resource_level"], dict)
        assert "high" in stats["by_resource_level"]
        assert "medium" in stats["by_resource_level"]
        assert "low" in stats["by_resource_level"]
        assert stats["with_specialization"] >= 0

    def test_recommend_model_russian(self, registry):
        """Test model recommendation for Russian."""
        model = registry._recommend_model("ru", "high")
        assert model == "yandexgpt"

    def test_recommend_model_chinese(self, registry):
        """Test model recommendation for Chinese."""
        model = registry._recommend_model("zh", "high")
        assert model == "gpt-4.5"

    def test_recommend_model_japanese(self, registry):
        """Test model recommendation for Japanese."""
        model = registry._recommend_model("ja", "high")
        assert model == "claude-3.5-sonnet"

    def test_recommend_model_low_resource(self, registry):
        """Test model recommendation for low-resource language."""
        model = registry._recommend_model("be", "low")
        assert model == "gemini-2.0"

    def test_recommend_model_high_resource_generic(self, registry):
        """Test model recommendation for generic high-resource language."""
        model = registry._recommend_model("es", "high")
        assert model == "claude-3.5-sonnet"


class TestLanguageRegistrySingleton:
    """Test singleton pattern for language registry."""

    def test_get_language_registry_returns_instance(self):
        """Test that get_language_registry returns LanguageRegistry instance."""
        registry = get_language_registry()
        assert isinstance(registry, LanguageRegistry)

    def test_get_language_registry_singleton(self):
        """Test that get_language_registry returns same instance."""
        registry1 = get_language_registry()
        registry2 = get_language_registry()
        assert registry1 is registry2


class TestLanguageData:
    """Test language data integrity."""

    @pytest.fixture
    def registry(self):
        """Create LanguageRegistry instance."""
        return LanguageRegistry()

    def test_all_languages_have_required_fields(self, registry):
        """Test that all language definitions have required fields."""
        required_fields = {"name", "native_name", "resource_level", "specialization", "iso639_3"}

        for lang_code, lang_info in registry.SUPPORTED_LANGUAGES.items():
            assert isinstance(lang_code, str)
            assert len(lang_code) == 2  # ISO 639-1 codes are 2 characters
            for field in required_fields:
                assert field in lang_info, f"Language {lang_code} missing field {field}"

    def test_resource_levels_are_valid(self, registry):
        """Test that all resource levels are valid."""
        valid_levels = {"high", "medium", "low"}

        for lang_code, lang_info in registry.SUPPORTED_LANGUAGES.items():
            assert lang_info["resource_level"] in valid_levels

    def test_specialization_is_boolean(self, registry):
        """Test that specialization field is boolean."""
        for lang_code, lang_info in registry.SUPPORTED_LANGUAGES.items():
            assert isinstance(lang_info["specialization"], bool)

    def test_russian_has_specialization(self, registry):
        """Test that Russian is marked with specialization (RussianFluencyAgent)."""
        assert registry.SUPPORTED_LANGUAGES["ru"]["specialization"] is True

    def test_iso639_3_codes_are_valid(self, registry):
        """Test that all ISO 639-3 codes are 3 characters."""
        for lang_code, lang_info in registry.SUPPORTED_LANGUAGES.items():
            assert len(lang_info["iso639_3"]) == 3
