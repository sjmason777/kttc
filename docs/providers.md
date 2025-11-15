# Supported LLM Providers

KTTC supports multiple LLM providers for flexibility and cost optimization.

## OpenAI

### Setup

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

### Supported Models

| Model | Best For | Cost | Speed |
|-------|----------|------|-------|
| `gpt-4` | Complex translations | $$$ | Medium |
| `gpt-4-turbo` | General use | $$ | Fast |
| `gpt-3.5-turbo` | Simple translations | $ | Very Fast |

### Usage

```bash
kttc check source.txt translation.txt \
    --provider openai \
    --source-lang en --target-lang es
```

### Python API

```python
from kttc.llm import OpenAIProvider

provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.3
)
```

---

## Anthropic (Claude)

### Setup

```bash
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### Supported Models

| Model | Best For | Cost | Speed |
|-------|----------|------|-------|
| `claude-3-5-sonnet-20241022` | Most tasks | $$ | Fast |
| `claude-3-opus-20240229` | Highest quality | $$$ | Medium |
| `claude-3-haiku-20240307` | Speed & cost | $ | Very Fast |

### Usage

```bash
kttc check source.txt translation.txt \
    --provider anthropic \
    --source-lang en --target-lang es
```

### Python API

```python
from kttc.llm import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022",
    temperature=0.3
)
```

---

## GigaChat (Russian Provider)

### Setup

```bash
export KTTC_GIGACHAT_CLIENT_ID="your-client-id"
export KTTC_GIGACHAT_CLIENT_SECRET="your-client-secret"
```

### Supported Models

| Model | Best For |
|-------|----------|
| `GigaChat-Pro` | Professional translations |
| `GigaChat` | General use |

### Usage

```bash
kttc check source.txt translation.txt \
    --provider gigachat \
    --source-lang en --target-lang ru
```

### Python API

```python
from kttc.llm import GigaChatProvider

provider = GigaChatProvider(
    client_id="your-client-id",
    client_secret="your-client-secret",
    model="GigaChat-Pro"
)
```

**Note:** GigaChat is particularly good for Russian language translations.

---

## YandexGPT

### Setup

```bash
export KTTC_YANDEXGPT_API_KEY="your-api-key"
export KTTC_YANDEXGPT_FOLDER_ID="your-folder-id"
```

### Supported Models

| Model | Best For |
|-------|----------|
| `yandexgpt` | General translations |
| `yandexgpt-lite` | Fast & cheap |

### Usage

```bash
kttc check source.txt translation.txt \
    --provider yandexgpt \
    --source-lang en --target-lang ru
```

### Python API

```python
from kttc.llm import YandexGPTProvider

provider = YandexGPTProvider(
    api_key="your-api-key",
    folder_id="your-folder-id",
    model="yandexgpt"
)
```

---

## Provider Comparison

### Cost Comparison

| Provider | Model | Cost per 1M tokens | Quality | Speed |
|----------|-------|-------------------|---------|-------|
| OpenAI | GPT-4 | $$$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| OpenAI | GPT-4 Turbo | $$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| OpenAI | GPT-3.5 Turbo | $ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Anthropic | Claude 3.5 Sonnet | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Anthropic | Claude 3 Opus | $$$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Anthropic | Claude 3 Haiku | $ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GigaChat | GigaChat-Pro | $$ | ⭐⭐⭐⭐ (Russian) | ⭐⭐⭐ |

### Language Support

| Provider | English | Spanish | French | German | Chinese | Russian |
|----------|---------|---------|--------|--------|---------|---------|
| OpenAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Anthropic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| GigaChat | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| YandexGPT | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Smart Routing

KTTC can automatically select the optimal provider and model based on text complexity:

```bash
kttc check source.txt translation.txt \
    --smart-routing \
    --source-lang en --target-lang es
```

**How it works:**

1. **Simple texts** (< 100 chars) → Cheap model (GPT-3.5, Claude Haiku)
2. **Medium texts** (100-500 chars) → Mid-range model (GPT-4 Turbo)
3. **Complex texts** (> 500 chars) → Best model (GPT-4, Claude Sonnet)

**Cost savings:** Up to 60% reduction in API costs.

---

## Benchmarking Providers

Compare providers for your specific use case:

```bash
kttc benchmark \
    --source test.txt \
    --providers openai,anthropic,gigachat \
    --source-lang en --target-lang es
```

**Output:**

```
Provider Benchmark Results:
┌──────────┬──────────┬────────┬──────────┐
│ Provider │ MQM Score│ Time   │ Cost     │
├──────────┼──────────┼────────┼──────────┤
│ OpenAI   │ 96.5     │ 2.3s   │ $0.045   │
│ Anthropic│ 97.2     │ 1.8s   │ $0.038   │
│ GigaChat │ 94.8     │ 3.1s   │ $0.032   │
└──────────┴──────────┴────────┴──────────┘
```

---

## Provider Selection Guide

### Choose OpenAI if:
- You need broad language support
- You want reliable, consistent quality
- You're already using OpenAI for other tasks

### Choose Anthropic if:
- You need highest quality for critical translations
- You want faster response times
- You need detailed reasoning

### Choose GigaChat if:
- You're translating to/from Russian
- You need local Russian compliance
- Cost is a priority for Russian content

### Choose YandexGPT if:
- You're in Russia or CIS countries
- You need Russian language specialization
- You want local data residency

---

## API Rate Limits

| Provider | Requests/min | Tokens/min |
|----------|-------------|------------|
| OpenAI | 3,500 | 90,000 |
| Anthropic | 4,000 | 100,000 |
| GigaChat | 1,000 | 50,000 |
| YandexGPT | 500 | 30,000 |

**Note:** Limits vary by subscription tier. Check your provider's documentation.

---

## Troubleshooting

### Rate Limit Errors

If you hit rate limits:

```bash
# Add delay between requests
kttc batch --file data.csv --delay 1.0

# Use smart routing to reduce API calls
kttc check source.txt translation.txt --smart-routing
```

### Authentication Errors

Check your API keys:

```bash
# Verify key is set
echo $KTTC_OPENAI_API_KEY

# Test with verbose mode
kttc check source.txt translation.txt --provider openai -vv
```

### Model Not Available

Some models may not be available in all regions:

```bash
# Try a different model
kttc check source.txt translation.txt \
    --provider openai \
    --model gpt-4-turbo
```
