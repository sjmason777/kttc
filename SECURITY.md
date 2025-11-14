**English** · [Русский](SECURITY.ru.md) · [中文](SECURITY.zh.md)

# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of KTTC seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@kt.tc**

### What to Include

Please include the following information in your report:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

### What to Expect

* You should receive an acknowledgment within **48 hours**
* We will send a more detailed response within **7 days** indicating the next steps
* We will keep you informed about the progress toward a fix and full announcement
* We may ask for additional information or guidance

### Disclosure Policy

* Security issues will be handled with the highest priority
* We aim to patch critical vulnerabilities within **30 days**
* Public disclosure will be made after the patch is released
* We will credit you in the security advisory (if you wish)

## Security Best Practices

When using KTTC, please follow these security best practices:

### API Keys

* **Never commit API keys** to version control
* Use environment variables for all API keys:
  ```bash
  export OPENAI_API_KEY="your-key-here"
  export ANTHROPIC_API_KEY="your-key-here"
  ```
* Use `.env` files (which are gitignored) for local development
* Rotate keys regularly

### Input Validation

* Always validate and sanitize user input
* Be cautious when processing translations from untrusted sources
* Use the built-in validation in TranslationTask models

### Dependencies

* Keep dependencies up to date: `pip install -U ".[dev]"`
* Review dependency security with: `pip-audit` (recommended)
* Monitor for security advisories on dependencies

### Production Deployment

* Use HTTPS for all API communications
* Implement rate limiting for API endpoints
* Log security-relevant events
* Run with minimal required permissions
* Use virtual environments to isolate dependencies

## Known Security Considerations

### LLM API Calls

* All translation data is sent to LLM providers (OpenAI, Anthropic)
* **Do not use KTTC for sensitive/confidential content** unless you have appropriate data processing agreements
* Review your LLM provider's data handling policies

### Prompt Injection

* Be aware that malicious input could attempt prompt injection attacks
* The system includes basic safeguards, but always validate critical outputs
* Do not use translations for security-critical applications without human review

### Local Caching

* Cached translations are stored locally in SQLite
* Ensure proper file permissions on cache database
* Clear cache when handling sensitive data

## Security Updates

Security updates will be announced:

* In GitHub Security Advisories
* In the CHANGELOG.md file
* Via email to security@kt.tc subscribers (optional)

## Acknowledgments

We would like to thank the following security researchers for responsibly disclosing vulnerabilities:

* (None yet - be the first!)

---

**Last Updated:** November 10, 2025
