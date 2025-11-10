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

    # Default LLM Configuration
    default_llm_provider: str = Field(
        default="openai",
        description="Default LLM provider (openai or anthropic)",
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

    def get_llm_provider_key(self, provider: str | None = None) -> str:
        """Get API key for specified LLM provider.

        Args:
            provider: Provider name (openai or anthropic). Uses default if None.

        Returns:
            API key string

        Raises:
            ValueError: If API key not configured

        Example:
            >>> settings = Settings()
            >>> key = settings.get_llm_provider_key("openai")
        """
        provider = provider or self.default_llm_provider

        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured. Set KTTC_OPENAI_API_KEY")
            return self.openai_api_key
        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured. Set KTTC_ANTHROPIC_API_KEY")
            return self.anthropic_api_key
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")


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
