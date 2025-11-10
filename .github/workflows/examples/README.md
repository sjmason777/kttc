# GitHub Actions Workflow Examples

This directory contains example workflows demonstrating various ways to use KTTC in your CI/CD pipeline.

## Examples

### 1. Basic Check (`basic-check.yml`)

Simple quality check triggered on pull requests when translation files change.

**Use case**: Basic CI/CD integration for translation quality

**Features**:
- Runs on PR with translation file changes
- Single language pair check
- Fails if quality threshold not met

### 2. Multi-Language Check (`multi-language.yml`)

Check multiple target languages in parallel using matrix strategy.

**Use case**: Projects with translations in multiple languages

**Features**:
- Matrix strategy for parallel checks
- Separate reports per language
- Efficient use of CI resources

### 3. PR Comment Integration (`with-pr-comment.yml`)

Posts quality check results as a comment on the pull request.

**Use case**: Immediate feedback to translators

**Features**:
- Formatted PR comments with results
- Detailed error breakdown
- Links to full report

### 4. Scheduled Check (`scheduled-check.yml`)

Runs quality checks on a schedule (e.g., weekly).

**Use case**: Regular quality audits and monitoring

**Features**:
- Cron-based scheduling
- Creates GitHub issues for quality degradation
- Long-term report retention

## Getting Started

1. Choose an example that matches your needs
2. Copy it to `.github/workflows/` in your repository
3. Modify paths and parameters to match your project structure
4. Add required secrets (API keys) to your repository settings
5. Commit and push to trigger the workflow

## Configuration

All examples can be customized with these parameters:

```yaml
with:
  source-dir: 'path/to/source'      # Required
  translation-dir: 'path/to/target' # Required
  source-lang: 'en'                 # Required
  target-lang: 'es'                 # Required
  threshold: '95.0'                 # Optional (default: 95.0)
  provider: 'openai'                # Optional (default: openai)
  parallel: '4'                     # Optional (default: 4)
  output: 'report.json'             # Optional (default: kttc-report.json)
```

## Environment Variables

Set these as repository secrets:

- `OPENAI_API_KEY` - For OpenAI provider
- `ANTHROPIC_API_KEY` - For Anthropic provider

## Testing Locally

Before committing workflows, test them locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or
choco install act-cli  # Windows

# Run workflow
act pull_request -W .github/workflows/examples/basic-check.yml
```

## Best Practices

1. **Trigger on Specific Paths**: Only run when translation files change
2. **Use Matrix Strategy**: Check multiple languages in parallel
3. **Set Appropriate Thresholds**: Balance quality and practicality
4. **Upload Artifacts**: Save reports for analysis
5. **Comment on PRs**: Provide immediate feedback
6. **Handle Failures Gracefully**: Use `continue-on-error` when appropriate

## Support

For more information, see:
- [GitHub Actions Documentation](../../../docs/github-actions.md)
- [KTTC Documentation](../../../docs/)
- [GitHub Issues](https://github.com/your-org/kttc/issues)
