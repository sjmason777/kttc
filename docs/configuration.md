# Configuration

KTTC can be configured through environment variables, configuration files, and command-line options.

## Environment Variables

### API Keys

Set API keys for your LLM providers:

```bash
# OpenAI
export KTTC_OPENAI_API_KEY="sk-..."

# Anthropic
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."

# GigaChat (Russian provider)
export KTTC_GIGACHAT_CLIENT_ID="your-client-id"
export KTTC_GIGACHAT_CLIENT_SECRET="your-client-secret"

# YandexGPT
export KTTC_YANDEXGPT_API_KEY="your-api-key"
export KTTC_YANDEXGPT_FOLDER_ID="your-folder-id"
```

### Default Settings

Configure default behavior:

```bash
# Default LLM provider
export KTTC_DEFAULT_PROVIDER="openai"

# Default quality threshold
export KTTC_DEFAULT_THRESHOLD="95.0"

# Default source language
export KTTC_DEFAULT_SOURCE_LANG="en"

# Enable smart routing by default
export KTTC_SMART_ROUTING="true"
```

## Configuration File

Create a `.kttc.yml` file in your project directory or home directory (`~/.kttc.yml`):

```yaml
# .kttc.yml
default:
  provider: openai
  threshold: 95.0
  smart_routing: true
  output_format: json

providers:
  openai:
    model: gpt-4
    temperature: 0.3
    max_tokens: 2000

  anthropic:
    model: claude-3-5-sonnet-20241022
    temperature: 0.3
    max_tokens: 2000

glossaries:
  - name: base
    path: ./glossaries/base.json
    auto_load: true

  - name: technical
    path: ./glossaries/technical.json
    auto_load: false

quality:
  min_mqm_score: 95.0
  fail_on_critical: true
  fail_on_major: false

output:
  format: text
  colors: true
  verbose: false
```

## Command-Line Options

Command-line options override configuration file and environment variables.

Priority order (highest to lowest):
1. Command-line options
2. Configuration file (`.kttc.yml`)
3. Environment variables
4. Default values

Example:

```bash
# Uses provider from command line (highest priority)
kttc check source.txt translation.txt \
    --provider anthropic \
    --source-lang en --target-lang es
```

## Provider Configuration

### OpenAI

```yaml
providers:
  openai:
    api_key: ${KTTC_OPENAI_API_KEY}  # Use env variable
    model: gpt-4
    temperature: 0.3
    max_tokens: 2000
    timeout: 60
```

Available models:
- `gpt-4` - Most capable
- `gpt-4-turbo` - Faster and cheaper
- `gpt-3.5-turbo` - Cheapest

### Anthropic

```yaml
providers:
  anthropic:
    api_key: ${KTTC_ANTHROPIC_API_KEY}
    model: claude-3-5-sonnet-20241022
    temperature: 0.3
    max_tokens: 2000
    timeout: 60
```

Available models:
- `claude-3-5-sonnet-20241022` - Most capable
- `claude-3-opus-20240229` - Highest quality
- `claude-3-haiku-20240307` - Fastest

### GigaChat

```yaml
providers:
  gigachat:
    client_id: ${KTTC_GIGACHAT_CLIENT_ID}
    client_secret: ${KTTC_GIGACHAT_CLIENT_SECRET}
    model: GigaChat-Pro
    temperature: 0.3
    timeout: 60
```

## Quality Settings

### MQM Thresholds

Configure quality thresholds:

```yaml
quality:
  # Minimum MQM score to pass
  min_mqm_score: 95.0

  # Fail on critical issues
  fail_on_critical: true

  # Fail on major issues
  fail_on_major: false

  # Maximum number of minor issues
  max_minor_issues: 5
```

### Agent Configuration

Enable/disable specific agents:

```yaml
agents:
  accuracy:
    enabled: true
    weight: 0.3

  fluency:
    enabled: true
    weight: 0.25

  terminology:
    enabled: true
    weight: 0.2

  hallucination:
    enabled: true
    weight: 0.15

  context:
    enabled: true
    weight: 0.1
```

## Smart Routing

Configure smart routing for cost optimization:

```yaml
smart_routing:
  enabled: true

  # Models for different complexity levels
  simple:
    provider: openai
    model: gpt-3.5-turbo

  medium:
    provider: openai
    model: gpt-4-turbo

  complex:
    provider: anthropic
    model: claude-3-5-sonnet-20241022

  # Complexity thresholds
  thresholds:
    simple_max_chars: 100
    medium_max_chars: 500
```

## Glossaries

### Auto-loading Glossaries

```yaml
glossaries:
  - name: base
    path: ./glossaries/base.json
    auto_load: true  # Load automatically

  - name: technical
    path: ./glossaries/technical.json
    auto_load: false  # Load only when specified
```

### Glossary Format

```json
{
  "name": "Technical Terms",
  "version": "1.0",
  "terms": [
    {
      "source": "API",
      "target": "API",
      "context": "Keep as-is, do not translate",
      "case_sensitive": true
    },
    {
      "source": "cloud",
      "target": "nube",
      "context": "Technology context",
      "case_sensitive": false
    }
  ]
}
```

## Output Configuration

### Output Formats

```yaml
output:
  # Default format: text, json, yaml
  format: text

  # Enable colors in terminal
  colors: true

  # Verbosity level: 0 (quiet), 1 (normal), 2 (verbose)
  verbosity: 1

  # Include detailed issue explanations
  detailed_issues: true

  # Save results to file
  save_to_file: false
  default_output_dir: ./results
```

## Logging

Configure logging:

```yaml
logging:
  # Log level: DEBUG, INFO, WARNING, ERROR
  level: INFO

  # Log to file
  file: ./logs/kttc.log

  # Log format
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  # Rotate logs
  rotate: true
  max_bytes: 10485760  # 10MB
  backup_count: 5
```

## Example Complete Configuration

```yaml
# .kttc.yml - Complete example
default:
  provider: openai
  threshold: 95.0
  smart_routing: true
  output_format: json

providers:
  openai:
    api_key: ${KTTC_OPENAI_API_KEY}
    model: gpt-4
    temperature: 0.3

  anthropic:
    api_key: ${KTTC_ANTHROPIC_API_KEY}
    model: claude-3-5-sonnet-20241022
    temperature: 0.3

smart_routing:
  enabled: true
  simple:
    provider: openai
    model: gpt-3.5-turbo
  medium:
    provider: openai
    model: gpt-4-turbo
  complex:
    provider: anthropic
    model: claude-3-5-sonnet-20241022

glossaries:
  - name: base
    path: ./glossaries/base.json
    auto_load: true

quality:
  min_mqm_score: 95.0
  fail_on_critical: true
  fail_on_major: false
  max_minor_issues: 5

output:
  format: text
  colors: true
  verbosity: 1
  detailed_issues: true

logging:
  level: INFO
  file: ./logs/kttc.log
```
