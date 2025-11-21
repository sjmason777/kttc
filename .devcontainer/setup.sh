#!/bin/bash
# Setup script for GitHub Codespaces
# This script runs automatically when the Codespace is created

set -e

echo "🚀 Setting up KTTC development environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install KTTC in editable mode with dev dependencies
echo "📦 Installing KTTC with dev dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create example .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating example .env file..."
    cat > .env << 'EOF'
# KTTC API Keys Configuration
# Uncomment and set your API keys below

# OpenAI (GPT-4, GPT-3.5-turbo)
# KTTC_OPENAI_API_KEY=sk-...

# Anthropic (Claude)
# KTTC_ANTHROPIC_API_KEY=sk-ant-...

# GigaChat (Russian LLM)
# KTTC_GIGACHAT_CLIENT_ID=your-client-id
# KTTC_GIGACHAT_CLIENT_SECRET=your-client-secret

# YandexGPT
# KTTC_YANDEX_API_KEY=your-api-key
# KTTC_YANDEX_FOLDER_ID=your-folder-id

# Logging
# KTTC_LOG_LEVEL=INFO
EOF
    echo "✅ Created .env file (add your API keys there)"
fi

# Run a quick test to verify installation
echo "🧪 Running quick test..."
python3 -c "import kttc; print(f'✅ KTTC v{kttc.__version__} installed successfully!')"

echo ""
echo "✨ Setup complete!"
echo ""
echo "📚 Quick Start:"
echo "  1. Add your API keys to .env file"
echo "  2. Try demo mode: kttc check examples/cli/source_en.txt examples/cli/translation_ru_good.txt --demo"
echo "  3. Run tests: pytest tests/unit/"
echo "  4. See documentation: docs/en/README.md"
echo ""
echo "🎯 Useful commands:"
echo "  kttc --help                    # Show all commands"
echo "  kttc check --demo              # Try demo mode"
echo "  pytest                         # Run tests"
echo "  pre-commit run --all-files     # Run code quality checks"
echo ""
echo "Happy coding! 🚀"
