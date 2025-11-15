# Providers API

LLM provider implementations for different services.

## OpenAIProvider

::: kttc.llm.OpenAIProvider
    options:
      show_source: false
      heading_level: 3

### Example

```python
from kttc.llm import OpenAIProvider
from kttc.agents import AgentOrchestrator

provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.3,
    timeout=60
)

# Use with orchestrator
orchestrator = AgentOrchestrator(provider)
```

## AnthropicProvider

::: kttc.llm.AnthropicProvider
    options:
      show_source: false
      heading_level: 3

### Example

```python
from kttc.llm import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022",
    temperature=0.3
)
```

## GigaChatProvider

::: kttc.llm.GigaChatProvider
    options:
      show_source: false
      heading_level: 3

### Example

```python
from kttc.llm import GigaChatProvider

provider = GigaChatProvider(
    client_id="your-client-id",
    client_secret="your-client-secret",
    model="GigaChat-Pro"
)
```

## YandexGPTProvider

::: kttc.llm.YandexGPTProvider
    options:
      show_source: false
      heading_level: 3

### Example

```python
from kttc.llm import YandexGPTProvider

provider = YandexGPTProvider(
    api_key="your-api-key",
    folder_id="your-folder-id",
    model="yandexgpt"
)
```

## Base Provider

::: kttc.llm.BaseLLMProvider
    options:
      show_source: false
      heading_level: 3

All providers inherit from this base class.

## Complexity Routing

### ComplexityRouter

::: kttc.llm.ComplexityRouter
    options:
      show_source: false
      heading_level: 4

Automatically routes requests to different providers based on text complexity.

**Example:**

```python
from kttc.llm import ComplexityRouter, OpenAIProvider

# Configure router with different providers for different complexity levels
router = ComplexityRouter(
    simple_provider=OpenAIProvider(model="gpt-3.5-turbo"),
    medium_provider=OpenAIProvider(model="gpt-4-turbo"),
    complex_provider=AnthropicProvider(model="claude-3-5-sonnet-20241022")
)

# Router automatically selects provider
provider = await router.select_provider(task)
```

### ComplexityEstimator

::: kttc.llm.ComplexityEstimator
    options:
      show_source: false
      heading_level: 4

Estimates text complexity for smart routing.

## Error Handling

All providers can raise these exceptions:

### LLMError

::: kttc.llm.LLMError
    options:
      show_source: false
      heading_level: 4

Base exception for all LLM errors.

### LLMAuthenticationError

::: kttc.llm.LLMAuthenticationError
    options:
      show_source: false
      heading_level: 4

Raised when authentication fails (invalid API key).

### LLMRateLimitError

::: kttc.llm.LLMRateLimitError
    options:
      show_source: false
      heading_level: 4

Raised when rate limits are exceeded.

### LLMTimeoutError

::: kttc.llm.LLMTimeoutError
    options:
      show_source: false
      heading_level: 4

Raised when request times out.

## Example: Error Handling

```python
from kttc.llm import (
    OpenAIProvider,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError
)

provider = OpenAIProvider(api_key="sk-...")

try:
    response = await provider.complete(messages=[...])
except LLMAuthenticationError:
    print("Invalid API key")
except LLMRateLimitError:
    print("Rate limit exceeded, please retry later")
except LLMTimeoutError:
    print("Request timed out")
```

## Prompt Templates

::: kttc.llm.PromptTemplate
    options:
      show_source: false
      heading_level: 3

Create reusable prompt templates.

**Example:**

```python
from kttc.llm import PromptTemplate

template = PromptTemplate(
    template="Translate the following {lang} text: {text}",
    required_variables=["lang", "text"]
)

prompt = template.format(lang="English", text="Hello")
```
