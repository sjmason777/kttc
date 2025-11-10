"""Configuration management for KTTC.

Handles API keys, model settings, and other configuration using Pydantic Settings.
Supports environment variables and .env files.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Loads configuration from environment variables or .env file.
    All settings can be overridden via environment variables with KTTC_ prefix.

    Example .env file:
        KTTC_OPENAI_API_KEY=sk-...
        KTTC_ANTHROPIC_API_KEY=sk-ant-...
        KTTC_DEFAULT_LLM_PROVIDER=openai
        KTTC_DEFAULT_MODEL=gpt-4-turbo

    Example usage:
        >>> settings = Settings()
        >>> print(settings.openai_api_key)
        sk-...
    """

    # LLM API Keys
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
        json_schema_extra={"env": "KTTC_OPENAI_API_KEY"},
    )

    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
        json_schema_extra={"env": "KTTC_ANTHROPIC_API_KEY"},
    )

    # GigaChat (Sber) credentials
    gigachat_client_id: str | None = Field(
        default=None,
        description="GigaChat Client ID",
        json_schema_extra={"env": "KTTC_GIGACHAT_CLIENT_ID"},
    )

    gigachat_client_secret: str | None = Field(
        default=None,
        description="GigaChat Client Secret",
        json_schema_extra={"env": "KTTC_GIGACHAT_CLIENT_SECRET"},
    )

    gigachat_scope: str = Field(
        default="GIGACHAT_API_PERS",
        description="GigaChat API scope",
        json_schema_extra={"env": "KTTC_GIGACHAT_SCOPE"},
    )

    # Yandex GPT credentials
    yandex_api_key: str | None = Field(
        default=None,
        description="Yandex GPT API key",
        json_schema_extra={"env": "KTTC_YANDEX_API_KEY"},
    )

    yandex_folder_id: str | None = Field(
        default=None,
        description="Yandex Cloud Folder ID",
        json_schema_extra={"env": "KTTC_YANDEX_FOLDER_ID"},
    )

    # Default LLM Configuration
    default_llm_provider: str = Field(
        default="openai",
        description="Default LLM provider (openai, anthropic, gigachat, yandex)",
        json_schema_extra={"env": "KTTC_DEFAULT_LLM_PROVIDER"},
    )

    default_model: str = Field(
        default="gpt-4-turbo",
        description="Default model name",
        json_schema_extra={"env": "KTTC_DEFAULT_MODEL"},
    )

    default_temperature: float = Field(
        default=0.1,
        description="Default temperature for LLM calls",
        ge=0.0,
        le=2.0,
        json_schema_extra={"env": "KTTC_DEFAULT_TEMPERATURE"},
    )

    default_max_tokens: int = Field(
        default=2000,
        description="Default max tokens for LLM calls",
        gt=0,
        json_schema_extra={"env": "KTTC_DEFAULT_MAX_TOKENS"},
    )

    # Request Settings
    request_timeout: float = Field(
        default=30.0,
        description="Default timeout for API requests (seconds)",
        gt=0,
        json_schema_extra={"env": "KTTC_REQUEST_TIMEOUT"},
    )

    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed API calls",
        ge=0,
        json_schema_extra={"env": "KTTC_MAX_RETRIES"},
    )

    # Quality Thresholds
    mqm_pass_threshold: float = Field(
        default=95.0,
        description="Minimum MQM score to pass",
        ge=0.0,
        le=100.0,
        json_schema_extra={"env": "KTTC_MQM_PASS_THRESHOLD"},
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        json_schema_extra={"env": "KTTC_LOG_LEVEL"},
    )

    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KTTC_",
        case_sensitive=False,
        extra="ignore",
    )

    def get_llm_provider_credentials(self, provider: str | None = None) -> dict[str, str]:
        """Get credentials for specified LLM provider.

        Args:
            provider: Provider name (openai, anthropic, gigachat, yandex).
                     Uses default if None.

        Returns:
            Dictionary with provider credentials

        Raises:
            ValueError: If credentials not configured

        Example:
            >>> settings = Settings()
            >>> creds = settings.get_llm_provider_credentials("openai")
            >>> # Returns: {"api_key": "sk-..."}
            >>> creds = settings.get_llm_provider_credentials("gigachat")
            >>> # Returns: {"client_id": "...", "client_secret": "...", "scope": "..."}
        """
        provider = provider or self.default_llm_provider

        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured. Set KTTC_OPENAI_API_KEY")
            return {"api_key": self.openai_api_key}

        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured. Set KTTC_ANTHROPIC_API_KEY")
            return {"api_key": self.anthropic_api_key}

        elif provider == "gigachat":
            if not self.gigachat_client_id or not self.gigachat_client_secret:
                raise ValueError(
                    "GigaChat credentials not configured. "
                    "Set KTTC_GIGACHAT_CLIENT_ID and KTTC_GIGACHAT_CLIENT_SECRET"
                )
            return {
                "client_id": self.gigachat_client_id,
                "client_secret": self.gigachat_client_secret,
                "scope": self.gigachat_scope,
            }

        elif provider == "yandex":
            if not self.yandex_api_key or not self.yandex_folder_id:
                raise ValueError(
                    "Yandex GPT credentials not configured. "
                    "Set KTTC_YANDEX_API_KEY and KTTC_YANDEX_FOLDER_ID"
                )
            return {
                "api_key": self.yandex_api_key,
                "folder_id": self.yandex_folder_id,
            }

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def get_llm_provider_key(self, provider: str | None = None) -> str:
        """Get API key for specified LLM provider (legacy method).

        Deprecated: Use get_llm_provider_credentials() instead.

        Args:
            provider: Provider name. Uses default if None.

        Returns:
            API key string

        Raises:
            ValueError: If API key not configured or provider uses different auth
        """
        provider = provider or self.default_llm_provider
        creds = self.get_llm_provider_credentials(provider)

        if "api_key" in creds:
            return creds["api_key"]
        else:
            raise ValueError(
                f"Provider {provider} does not use simple API key authentication. "
                f"Use get_llm_provider_credentials() instead."
            )


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance.

    Returns:
        Settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.default_model)
        gpt-4-turbo
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
