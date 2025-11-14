# Default Glossaries

This directory contains default glossaries provided with KTTC.

## Available Glossaries

### base.json
Common programming and technical terms across all projects.

**Terms:** API, Docker, JSON, HTTP, REST, endpoint, pipeline, etc.

**Usage:**
```bash
kttc check --glossary base ...
```

### technical.json
Software development and DevOps terminology.

**Terms:** machine learning, neural network, database, kubernetes, etc.

**Usage:**
```bash
kttc check --glossary base,technical ...
```

## Creating Custom Glossaries

### From CSV
```bash
kttc glossary create my-project --from-csv my-terms.csv
```

**CSV Format:**
```csv
source,target,source_lang,target_lang,domain,notes
API,API,en,ru,technical,Keep uppercase
endpoint,эндпоинт,en,ru,technical,Accepted transliteration
```

### From JSON
```bash
kttc glossary create my-project --from-json my-terms.json
```

**JSON Format:**
```json
{
  "metadata": {
    "name": "My Project Terms",
    "version": "1.0.0"
  },
  "entries": [
    {
      "source": "API",
      "target": "API",
      "source_lang": "en",
      "target_lang": "ru",
      "do_not_translate": true
    }
  ]
}
```

## Management Commands

```bash
# List all glossaries
kttc glossary list

# Show glossary contents
kttc glossary show base

# Merge glossaries
kttc glossary merge base technical --output combined

# Validate glossary
kttc glossary validate my-terms.json

# Export to CSV
kttc glossary export technical --format csv
```

## Glossary Locations

Glossaries are searched in this order:

1. **Project directory:** `./glossaries/`
   - Version controlled with project
   - Team-shared glossaries

2. **User directory:** `~/.kttc/glossaries/`
   - Personal glossaries
   - Cross-project terms

## Best Practices

1. **Use base glossary** for common technical terms
2. **Create project-specific glossaries** for custom terminology
3. **Combine multiple glossaries** for complex projects
4. **Mark brand names** with `do_not_translate: true`
5. **Add notes** for translators about term usage
6. **Use domains** to organize terms by category

## Example Workflow

```bash
# Start with base terms
kttc check --glossary base --source "The API endpoint..." --translation "..."

# Add project-specific terms
kttc glossary create my-project --from-csv project-terms.csv

# Use both
kttc check --glossary base,my-project --source "..." --translation "..."

# Share with team (commit to git)
git add glossaries/my-project.json
git commit -m "Add project glossary"
```
