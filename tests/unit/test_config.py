"""Unit tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from kttc.utils.config import Settings, get_settings


class TestSettings:
    """Test Settings configuration class."""

    def test_default_settings(self) -> None:
        """Test configuration values (from .env or defaults)."""
        settings = Settings()
        # Check that settings are loaded (either from .env or defaults)
        assert settings.default_llm_provider in ["openai", "gigachat", "anthropic", "yandex"]
        # Default model varies based on environment, just check it's a string
        assert isinstance(settings.default_model, str)
        assert len(settings.default_model) > 0
        assert settings.default_temperature == 0.1
        # default_max_tokens can be 1000 (.env) or 2000 (config.py default)
        assert settings.default_max_tokens in [1000, 2000]
        assert settings.request_timeout == 30.0
        assert settings.max_retries == 3
        assert settings.mqm_pass_threshold == 95.0
        assert settings.log_level == "INFO"

    def test_settings_from_env_vars(self) -> None:
        """Test loading settings from environment variables."""
        with patch.dict(
            os.environ,
            {
                "KTTC_OPENAI_API_KEY": "test-openai-key",
                "KTTC_ANTHROPIC_API_KEY": "test-anthropic-key",
                "KTTC_DEFAULT_LLM_PROVIDER": "anthropic",
                "KTTC_DEFAULT_MODEL": "claude-3-5-sonnet-20241022",
                "KTTC_DEFAULT_TEMPERATURE": "0.5",
                "KTTC_DEFAULT_MAX_TOKENS": "1000",
                "KTTC_REQUEST_TIMEOUT": "60.0",
                "KTTC_MAX_RETRIES": "5",
                "KTTC_MQM_PASS_THRESHOLD": "90.0",
                "KTTC_LOG_LEVEL": "DEBUG",
            },
            clear=False,
        ):
            settings = Settings()
            assert settings.openai_api_key == "test-openai-key"
            assert settings.anthropic_api_key == "test-anthropic-key"
            assert settings.default_llm_provider == "anthropic"
            assert settings.default_model == "claude-3-5-sonnet-20241022"
            assert settings.default_temperature == 0.5
            assert settings.default_max_tokens == 1000
            assert settings.request_timeout == 60.0
            assert settings.max_retries == 5
            assert settings.mqm_pass_threshold == 90.0
            assert settings.log_level == "DEBUG"

    def test_get_llm_provider_key_openai(self) -> None:
        """Test getting OpenAI API key."""
        with patch.dict(
            os.environ,
            {"KTTC_OPENAI_API_KEY": "test-openai-key"},
            clear=False,
        ):
            settings = Settings()
            key = settings.get_llm_provider_key("openai")
            assert key == "test-openai-key"

    def test_get_llm_provider_key_anthropic(self) -> None:
        """Test getting Anthropic API key."""
        with patch.dict(
            os.environ,
            {"KTTC_ANTHROPIC_API_KEY": "test-anthropic-key"},
            clear=False,
        ):
            settings = Settings()
            key = settings.get_llm_provider_key("anthropic")
            assert key == "test-anthropic-key"

    def test_get_llm_provider_key_default_provider(self) -> None:
        """Test getting API key for default provider."""
        with patch.dict(
            os.environ,
            {
                "KTTC_OPENAI_API_KEY": "test-openai-key",
                "KTTC_DEFAULT_LLM_PROVIDER": "openai",
            },
            clear=False,
        ):
            settings = Settings()
            key = settings.get_llm_provider_key()  # No provider specified
            assert key == "test-openai-key"

    def test_get_llm_provider_key_unknown_provider(self) -> None:
        """Test error for unknown provider."""
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            settings.get_llm_provider_key("unknown_provider")

    def test_temperature_validation(self) -> None:
        """Test temperature value validation."""
        with patch.dict(os.environ, {"KTTC_DEFAULT_TEMPERATURE": "0.0"}, clear=False):
            settings = Settings()
            assert settings.default_temperature == 0.0

        with patch.dict(os.environ, {"KTTC_DEFAULT_TEMPERATURE": "2.0"}, clear=False):
            settings = Settings()
            assert settings.default_temperature == 2.0

        # Test invalid values
        with patch.dict(os.environ, {"KTTC_DEFAULT_TEMPERATURE": "-0.1"}, clear=False):
            with pytest.raises(Exception):  # pydantic validation error
                Settings()

        with patch.dict(os.environ, {"KTTC_DEFAULT_TEMPERATURE": "2.1"}, clear=False):
            with pytest.raises(Exception):  # pydantic validation error
                Settings()

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens value validation."""
        with patch.dict(os.environ, {"KTTC_DEFAULT_MAX_TOKENS": "100"}, clear=False):
            settings = Settings()
            assert settings.default_max_tokens == 100

        # Test invalid value (must be > 0)
        with patch.dict(os.environ, {"KTTC_DEFAULT_MAX_TOKENS": "0"}, clear=False):
            with pytest.raises(Exception):  # pydantic validation error
                Settings()

    def test_mqm_threshold_validation(self) -> None:
        """Test MQM threshold validation."""
        with patch.dict(os.environ, {"KTTC_MQM_PASS_THRESHOLD": "0.0"}, clear=False):
            settings = Settings()
            assert settings.mqm_pass_threshold == 0.0

        with patch.dict(os.environ, {"KTTC_MQM_PASS_THRESHOLD": "100.0"}, clear=False):
            settings = Settings()
            assert settings.mqm_pass_threshold == 100.0

        # Test invalid values
        with patch.dict(os.environ, {"KTTC_MQM_PASS_THRESHOLD": "-1.0"}, clear=False):
            with pytest.raises(Exception):  # pydantic validation error
                Settings()

        with patch.dict(os.environ, {"KTTC_MQM_PASS_THRESHOLD": "101.0"}, clear=False):
            with pytest.raises(Exception):  # pydantic validation error
                Settings()

    def test_max_retries_validation(self) -> None:
        """Test max_retries validation."""
        with patch.dict(os.environ, {"KTTC_MAX_RETRIES": "0"}, clear=False):
            settings = Settings()
            assert settings.max_retries == 0

        with patch.dict(os.environ, {"KTTC_MAX_RETRIES": "10"}, clear=False):
            settings = Settings()
            assert settings.max_retries == 10

        # Test invalid value (must be >= 0)
        with patch.dict(os.environ, {"KTTC_MAX_RETRIES": "-1"}, clear=False):
            with pytest.raises(Exception):  # pydantic validation error
                Settings()

    def test_request_timeout_validation(self) -> None:
        """Test request_timeout validation."""
        with patch.dict(os.environ, {"KTTC_REQUEST_TIMEOUT": "10.5"}, clear=False):
            settings = Settings()
            assert settings.request_timeout == 10.5

        # Test invalid value (must be > 0)
        with patch.dict(os.environ, {"KTTC_REQUEST_TIMEOUT": "0.0"}, clear=False):
            with pytest.raises(Exception):  # pydantic validation error
                Settings()

    def test_case_insensitive_env_vars(self) -> None:
        """Test that environment variables are case insensitive."""
        with patch.dict(
            os.environ,
            {"kttc_log_level": "warning"},  # lowercase
            clear=False,
        ):
            settings = Settings()
            # Should work due to case_sensitive=False
            assert settings.log_level.upper() == "WARNING"

    def test_get_llm_provider_credentials_gigachat(self) -> None:
        """Test getting GigaChat credentials."""
        with patch.dict(
            os.environ,
            {
                "KTTC_GIGACHAT_CLIENT_ID": "test-client-id",
                "KTTC_GIGACHAT_CLIENT_SECRET": "test-client-secret",
                "KTTC_GIGACHAT_SCOPE": "GIGACHAT_API_CORP",
            },
            clear=False,
        ):
            settings = Settings()
            creds = settings.get_llm_provider_credentials("gigachat")
            assert creds["client_id"] == "test-client-id"
            assert creds["client_secret"] == "test-client-secret"
            assert creds["scope"] == "GIGACHAT_API_CORP"

    def test_get_llm_provider_credentials_gigachat_missing(self) -> None:
        """Test error when GigaChat credentials are not configured."""
        # Override to ensure credentials are None
        with patch.dict(
            os.environ,
            {
                "KTTC_GIGACHAT_CLIENT_ID": "",
                "KTTC_GIGACHAT_CLIENT_SECRET": "",
            },
            clear=True,
        ):
            settings = Settings()
            with pytest.raises(ValueError, match="GigaChat credentials not configured"):
                settings.get_llm_provider_credentials("gigachat")

    def test_get_llm_provider_credentials_yandex(self) -> None:
        """Test getting Yandex GPT credentials."""
        with patch.dict(
            os.environ,
            {
                "KTTC_YANDEX_API_KEY": "test-yandex-key",
                "KTTC_YANDEX_FOLDER_ID": "test-folder-id",
            },
            clear=False,
        ):
            settings = Settings()
            creds = settings.get_llm_provider_credentials("yandex")
            assert creds["api_key"] == "test-yandex-key"
            assert creds["folder_id"] == "test-folder-id"

    def test_get_llm_provider_credentials_yandex_missing(self) -> None:
        """Test error when Yandex GPT credentials are not configured."""
        # Override to ensure credentials are None
        with patch.dict(
            os.environ,
            {
                "KTTC_YANDEX_API_KEY": "",
                "KTTC_YANDEX_FOLDER_ID": "",
            },
            clear=True,
        ):
            settings = Settings()
            with pytest.raises(ValueError, match="Yandex GPT credentials not configured"):
                settings.get_llm_provider_credentials("yandex")

    def test_get_llm_provider_key_gigachat_error(self) -> None:
        """Test error when trying to get API key for GigaChat (uses different auth)."""
        with patch.dict(
            os.environ,
            {
                "KTTC_GIGACHAT_CLIENT_ID": "test-id",
                "KTTC_GIGACHAT_CLIENT_SECRET": "test-secret",
            },
            clear=False,
        ):
            settings = Settings()
            with pytest.raises(ValueError, match="does not use simple API key authentication"):
                settings.get_llm_provider_key("gigachat")

    def test_get_llm_provider_key_yandex_has_api_key(self) -> None:
        """Test getting API key for Yandex (which does support simple API key)."""
        with patch.dict(
            os.environ,
            {
                "KTTC_YANDEX_API_KEY": "test-yandex-key",
                "KTTC_YANDEX_FOLDER_ID": "test-folder",
            },
            clear=False,
        ):
            settings = Settings()
            key = settings.get_llm_provider_key("yandex")
            assert key == "test-yandex-key"


class TestGetSettings:
    """Test get_settings singleton function."""

    def test_get_settings_returns_singleton(self) -> None:
        """Test that get_settings returns the same instance."""
        # Reset the global singleton
        import kttc.utils.config

        kttc.utils.config._settings = None

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_caches_instance(self) -> None:
        """Test that get_settings caches the Settings instance."""
        import kttc.utils.config

        kttc.utils.config._settings = None

        with patch.dict(
            os.environ,
            {"KTTC_DEFAULT_MODEL": "test-model-1"},
            clear=False,
        ):
            settings1 = get_settings()
            assert settings1.default_model == "test-model-1"

        # Change env var but should still return cached instance
        with patch.dict(
            os.environ,
            {"KTTC_DEFAULT_MODEL": "test-model-2"},
            clear=False,
        ):
            settings2 = get_settings()
            # Should still be the same cached instance with old value
            assert settings2 is settings1
            assert settings2.default_model == "test-model-1"

    def test_get_settings_respects_global_reset(self) -> None:
        """Test that resetting global settings works."""
        import kttc.utils.config

        with patch.dict(
            os.environ,
            {"KTTC_DEFAULT_MODEL": "model-1"},
            clear=False,
        ):
            kttc.utils.config._settings = None
            settings1 = get_settings()
            assert settings1.default_model == "model-1"

        with patch.dict(
            os.environ,
            {"KTTC_DEFAULT_MODEL": "model-2"},
            clear=False,
        ):
            # Reset global
            kttc.utils.config._settings = None
            settings2 = get_settings()
            assert settings2.default_model == "model-2"
            assert settings1 is not settings2
