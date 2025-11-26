"""Unit tests for configuration module.

Tests Settings and credential management.
"""

import pytest

from kttc.utils.config import Settings, get_settings


@pytest.mark.unit
class TestSettings:
    """Test Settings functionality."""

    def test_settings_has_default_values(self) -> None:
        """Test settings has expected attributes with defaults."""
        settings = Settings()

        # Check that settings has expected attributes
        # Values may vary based on environment variables
        assert hasattr(settings, "default_llm_provider")
        assert hasattr(settings, "default_model")
        assert hasattr(settings, "default_temperature")
        assert hasattr(settings, "default_max_tokens")
        assert hasattr(settings, "request_timeout")
        assert hasattr(settings, "max_retries")
        assert hasattr(settings, "mqm_pass_threshold")
        assert hasattr(settings, "log_level")
        assert hasattr(settings, "gigachat_scope")

        # Check types
        assert isinstance(settings.default_llm_provider, str)
        assert isinstance(settings.default_model, str)
        assert isinstance(settings.default_temperature, float)
        assert isinstance(settings.default_max_tokens, int)
        assert isinstance(settings.request_timeout, float)
        assert isinstance(settings.max_retries, int)
        assert isinstance(settings.mqm_pass_threshold, float)
        assert isinstance(settings.log_level, str)
        assert isinstance(settings.gigachat_scope, str)

    def test_settings_api_keys_type(self) -> None:
        """Test API keys have correct type."""
        settings = Settings()

        # These may be None or str depending on environment
        assert isinstance(settings.openai_api_key, (str, type(None)))
        assert isinstance(settings.anthropic_api_key, (str, type(None)))
        assert isinstance(settings.gigachat_client_id, (str, type(None)))
        assert isinstance(settings.gigachat_client_secret, (str, type(None)))
        assert isinstance(settings.yandex_api_key, (str, type(None)))
        assert isinstance(settings.yandex_folder_id, (str, type(None)))


@pytest.mark.unit
class TestGetLLMProviderCredentials:
    """Test get_llm_provider_credentials method."""

    def test_get_openai_credentials(self) -> None:
        """Test getting OpenAI credentials."""
        settings = Settings()
        # Use object.__setattr__ to bypass pydantic validation
        object.__setattr__(settings, "openai_api_key", "test-openai-key")
        creds = settings.get_llm_provider_credentials("openai")

        assert creds == {"api_key": "test-openai-key"}

    def test_get_anthropic_credentials(self) -> None:
        """Test getting Anthropic credentials."""
        settings = Settings()
        object.__setattr__(settings, "anthropic_api_key", "test-anthropic-key")
        creds = settings.get_llm_provider_credentials("anthropic")

        assert creds == {"api_key": "test-anthropic-key"}

    def test_get_gigachat_credentials(self) -> None:
        """Test getting GigaChat credentials."""
        settings = Settings()
        object.__setattr__(settings, "gigachat_client_id", "test-client-id")
        object.__setattr__(settings, "gigachat_client_secret", "test-client-secret")
        object.__setattr__(settings, "gigachat_scope", "GIGACHAT_API_PERS")
        creds = settings.get_llm_provider_credentials("gigachat")

        assert creds == {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "scope": "GIGACHAT_API_PERS",
        }

    def test_get_yandex_credentials(self) -> None:
        """Test getting Yandex credentials."""
        settings = Settings()
        object.__setattr__(settings, "yandex_api_key", "test-yandex-key")
        object.__setattr__(settings, "yandex_folder_id", "test-folder-id")
        creds = settings.get_llm_provider_credentials("yandex")

        assert creds == {
            "api_key": "test-yandex-key",
            "folder_id": "test-folder-id",
        }

    def test_get_credentials_uses_default_provider(self) -> None:
        """Test default provider is used when none specified."""
        settings = Settings()
        object.__setattr__(settings, "openai_api_key", "default-key")
        object.__setattr__(settings, "default_llm_provider", "openai")

        creds = settings.get_llm_provider_credentials(None)
        assert creds == {"api_key": "default-key"}

    def test_get_openai_credentials_missing_raises(self) -> None:
        """Test error when OpenAI credentials missing."""
        settings = Settings()
        object.__setattr__(settings, "openai_api_key", None)

        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            settings.get_llm_provider_credentials("openai")

    def test_get_anthropic_credentials_missing_raises(self) -> None:
        """Test error when Anthropic credentials missing."""
        settings = Settings()
        object.__setattr__(settings, "anthropic_api_key", None)

        with pytest.raises(ValueError, match="Anthropic API key not configured"):
            settings.get_llm_provider_credentials("anthropic")

    def test_get_gigachat_credentials_missing_raises(self) -> None:
        """Test error when GigaChat credentials missing."""
        settings = Settings()
        object.__setattr__(settings, "gigachat_client_id", None)
        object.__setattr__(settings, "gigachat_client_secret", None)

        with pytest.raises(ValueError, match="GigaChat credentials not configured"):
            settings.get_llm_provider_credentials("gigachat")

    def test_get_yandex_credentials_missing_raises(self) -> None:
        """Test error when Yandex credentials missing."""
        settings = Settings()
        object.__setattr__(settings, "yandex_api_key", None)
        object.__setattr__(settings, "yandex_folder_id", None)

        with pytest.raises(ValueError, match="Yandex GPT credentials not configured"):
            settings.get_llm_provider_credentials("yandex")

    def test_unknown_provider_raises(self) -> None:
        """Test error for unknown provider."""
        settings = Settings()

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            settings.get_llm_provider_credentials("unknown_provider")


@pytest.mark.unit
class TestGetLLMProviderKey:
    """Test get_llm_provider_key method (legacy)."""

    def test_get_provider_key_openai(self) -> None:
        """Test getting OpenAI key via legacy method."""
        settings = Settings()
        object.__setattr__(settings, "openai_api_key", "test-key")

        key = settings.get_llm_provider_key("openai")
        assert key == "test-key"

    def test_get_provider_key_anthropic(self) -> None:
        """Test getting Anthropic key via legacy method."""
        settings = Settings()
        object.__setattr__(settings, "anthropic_api_key", "test-key")

        key = settings.get_llm_provider_key("anthropic")
        assert key == "test-key"

    def test_get_provider_key_gigachat_raises(self) -> None:
        """Test GigaChat raises with legacy method (no simple key)."""
        settings = Settings()
        object.__setattr__(settings, "gigachat_client_id", "id")
        object.__setattr__(settings, "gigachat_client_secret", "secret")

        with pytest.raises(ValueError, match="does not use simple API key"):
            settings.get_llm_provider_key("gigachat")

    def test_get_provider_key_uses_default(self) -> None:
        """Test default provider is used."""
        settings = Settings()
        object.__setattr__(settings, "openai_api_key", "default-key")
        object.__setattr__(settings, "default_llm_provider", "openai")

        key = settings.get_llm_provider_key(None)
        assert key == "default-key"


@pytest.mark.unit
class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_returns_settings(self) -> None:
        """Test get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_returns_same_instance(self) -> None:
        """Test get_settings returns singleton."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


@pytest.mark.unit
class TestSettingsValidation:
    """Test Settings validation."""

    def test_temperature_range(self) -> None:
        """Test temperature validation."""
        settings = Settings()
        assert 0.0 <= settings.default_temperature <= 2.0

    def test_max_tokens_positive(self) -> None:
        """Test max_tokens is positive."""
        settings = Settings()
        assert settings.default_max_tokens > 0

    def test_request_timeout_positive(self) -> None:
        """Test request_timeout is positive."""
        settings = Settings()
        assert settings.request_timeout > 0

    def test_max_retries_non_negative(self) -> None:
        """Test max_retries is non-negative."""
        settings = Settings()
        assert settings.max_retries >= 0

    def test_mqm_threshold_range(self) -> None:
        """Test MQM threshold is in valid range."""
        settings = Settings()
        assert 0.0 <= settings.mqm_pass_threshold <= 100.0
