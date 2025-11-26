"""Unit tests for language registry module.

Tests language support, capabilities, and model recommendations.
"""

import pytest

from kttc.utils.languages import LanguageRegistry, get_language_registry


@pytest.mark.unit
class TestLanguageRegistry:
    """Test LanguageRegistry functionality."""

    @pytest.fixture
    def registry(self) -> LanguageRegistry:
        """Provide a fresh LanguageRegistry instance."""
        return LanguageRegistry()

    def test_supported_languages_defined(self, registry: LanguageRegistry) -> None:
        """Test that supported languages are defined."""
        assert len(registry.SUPPORTED_LANGUAGES) > 0
        assert "en" in registry.SUPPORTED_LANGUAGES
        assert "ru" in registry.SUPPORTED_LANGUAGES

    def test_get_language_capabilities_supported(self, registry: LanguageRegistry) -> None:
        """Test getting capabilities for supported language."""
        caps = registry.get_language_capabilities("en")

        assert caps["supported"] is True
        assert caps["lang_code"] == "en"
        assert caps["name"] == "English"
        assert caps["resource_level"] == "high"
        assert "recommended_model" in caps

    def test_get_language_capabilities_russian(self, registry: LanguageRegistry) -> None:
        """Test Russian language with specialization."""
        caps = registry.get_language_capabilities("ru")

        assert caps["supported"] is True
        assert caps["name"] == "Russian"
        assert caps["has_specialized_agents"] is True
        assert caps["recommended_model"] == "yandexgpt"

    def test_get_language_capabilities_unsupported(self, registry: LanguageRegistry) -> None:
        """Test getting capabilities for unsupported language."""
        caps = registry.get_language_capabilities("xyz")

        assert caps["supported"] is False
        assert "message" in caps

    def test_is_language_supported(self, registry: LanguageRegistry) -> None:
        """Test language support check."""
        assert registry.is_language_supported("en") is True
        assert registry.is_language_supported("ru") is True
        assert registry.is_language_supported("xyz") is False

    def test_get_all_languages(self, registry: LanguageRegistry) -> None:
        """Test getting all languages."""
        languages = registry.get_all_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert all("lang_code" in lang for lang in languages)
        assert all("name" in lang for lang in languages)

    def test_get_languages_by_resource_level_high(self, registry: LanguageRegistry) -> None:
        """Test getting high-resource languages."""
        high_langs = registry.get_languages_by_resource_level("high")

        assert isinstance(high_langs, list)
        assert "en" in high_langs
        assert "ru" in high_langs
        assert "es" in high_langs

    def test_get_languages_by_resource_level_medium(self, registry: LanguageRegistry) -> None:
        """Test getting medium-resource languages."""
        medium_langs = registry.get_languages_by_resource_level("medium")

        assert isinstance(medium_langs, list)
        assert "uk" in medium_langs
        assert "pl" in medium_langs

    def test_get_languages_by_resource_level_low(self, registry: LanguageRegistry) -> None:
        """Test getting low-resource languages."""
        low_langs = registry.get_languages_by_resource_level("low")

        assert isinstance(low_langs, list)
        assert "be" in low_langs
        assert "ka" in low_langs

    def test_get_language_pairs(self, registry: LanguageRegistry) -> None:
        """Test getting language pairs."""
        pairs = registry.get_language_pairs()

        assert isinstance(pairs, list)
        assert len(pairs) > 0
        assert ("en", "ru") in pairs
        assert ("ru", "en") in pairs
        # No same-language pairs
        assert ("en", "en") not in pairs

    def test_get_statistics(self, registry: LanguageRegistry) -> None:
        """Test getting registry statistics."""
        stats = registry.get_statistics()

        assert "total_languages" in stats
        assert "by_resource_level" in stats
        assert "with_specialization" in stats
        assert "total_possible_pairs" in stats
        assert stats["total_languages"] > 0
        assert stats["with_specialization"] >= 3  # ru, hi, fa have specialization


@pytest.mark.unit
class TestModelRecommendations:
    """Test model recommendations for different languages."""

    @pytest.fixture
    def registry(self) -> LanguageRegistry:
        """Provide a fresh LanguageRegistry instance."""
        return LanguageRegistry()

    def test_russian_gets_yandexgpt(self, registry: LanguageRegistry) -> None:
        """Test Russian gets YandexGPT recommendation."""
        caps = registry.get_language_capabilities("ru")
        assert caps["recommended_model"] == "yandexgpt"

    def test_chinese_gets_gpt4(self, registry: LanguageRegistry) -> None:
        """Test Chinese gets GPT-4 recommendation."""
        caps = registry.get_language_capabilities("zh")
        assert "gpt" in caps["recommended_model"].lower()

    def test_japanese_gets_claude(self, registry: LanguageRegistry) -> None:
        """Test Japanese gets Claude recommendation."""
        caps = registry.get_language_capabilities("ja")
        assert "claude" in caps["recommended_model"].lower()

    def test_hindi_gets_claude(self, registry: LanguageRegistry) -> None:
        """Test Hindi gets Claude recommendation."""
        caps = registry.get_language_capabilities("hi")
        assert "claude" in caps["recommended_model"].lower()

    def test_persian_gets_claude(self, registry: LanguageRegistry) -> None:
        """Test Persian gets Claude recommendation."""
        caps = registry.get_language_capabilities("fa")
        assert "claude" in caps["recommended_model"].lower()

    def test_low_resource_gets_gemini(self, registry: LanguageRegistry) -> None:
        """Test low-resource language gets Gemini recommendation."""
        caps = registry.get_language_capabilities("be")
        assert "gemini" in caps["recommended_model"].lower()

    def test_high_resource_default_gets_claude(self, registry: LanguageRegistry) -> None:
        """Test high-resource default gets Claude recommendation."""
        caps = registry.get_language_capabilities("es")
        assert "claude" in caps["recommended_model"].lower()


@pytest.mark.unit
class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_get_language_registry_returns_instance(self) -> None:
        """Test that get_language_registry returns instance."""
        registry = get_language_registry()

        assert isinstance(registry, LanguageRegistry)

    def test_get_language_registry_is_singleton(self) -> None:
        """Test that get_language_registry returns same instance."""
        registry1 = get_language_registry()
        registry2 = get_language_registry()

        assert registry1 is registry2


@pytest.mark.unit
class TestLanguageInfo:
    """Test specific language information."""

    @pytest.fixture
    def registry(self) -> LanguageRegistry:
        """Provide a fresh LanguageRegistry instance."""
        return LanguageRegistry()

    def test_english_info(self, registry: LanguageRegistry) -> None:
        """Test English language info."""
        lang = registry.SUPPORTED_LANGUAGES["en"]

        assert lang["name"] == "English"
        assert lang["native_name"] == "English"
        assert lang["resource_level"] == "high"
        assert lang["iso639_3"] == "eng"

    def test_russian_info(self, registry: LanguageRegistry) -> None:
        """Test Russian language info."""
        lang = registry.SUPPORTED_LANGUAGES["ru"]

        assert lang["name"] == "Russian"
        assert lang["native_name"] == "Русский"
        assert lang["resource_level"] == "high"
        assert lang["specialization"] is True
        assert lang["iso639_3"] == "rus"

    def test_hindi_info(self, registry: LanguageRegistry) -> None:
        """Test Hindi language info."""
        lang = registry.SUPPORTED_LANGUAGES["hi"]

        assert lang["name"] == "Hindi"
        assert lang["native_name"] == "हिन्दी"
        assert lang["resource_level"] == "high"
        assert lang["specialization"] is True

    def test_persian_info(self, registry: LanguageRegistry) -> None:
        """Test Persian language info."""
        lang = registry.SUPPORTED_LANGUAGES["fa"]

        assert lang["name"] == "Persian"
        assert lang["native_name"] == "فارسی"
        assert lang["specialization"] is True
