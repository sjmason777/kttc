"""LLM integration layer for KTTC.

Provides abstract interface and concrete implementations for
various LLM providers (OpenAI, Anthropic, Yandex, Sber GigaChat).
"""

from .anthropic_provider import AnthropicProvider
from .base import (
    BaseLLMProvider,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .gigachat_provider import GigaChatProvider
from .openai_provider import OpenAIProvider
from .prompts import PromptTemplate, PromptTemplateError
from .yandex_provider import YandexGPTProvider

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "YandexGPTProvider",
    "GigaChatProvider",
    "PromptTemplate",
    "PromptTemplateError",
    "LLMError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
]
