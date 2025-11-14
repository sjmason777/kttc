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
from .complexity_router import ComplexityEstimator, ComplexityRouter, ComplexityScore
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
    "ComplexityEstimator",
    "ComplexityRouter",
    "ComplexityScore",
]
