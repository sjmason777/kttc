#!/bin/bash
# Script for running GigaChat integration tests

set -e  # Exit on error

echo "🧪 KTTC GigaChat Integration Tests"
echo "======================================"
echo ""

# Check Python version
if ! command -v python3.11 &> /dev/null && ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.11 or 3.12 is required"
    echo "   Install with: brew install python@3.11"
    exit 1
fi

# Select Python version
if command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
else
    PYTHON=python3.12
fi

echo "✓ Using: $PYTHON ($($PYTHON --version))"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "   Create .env with GigaChat credentials"
    exit 1
fi

# Check credentials
if ! grep -q "KTTC_GIGACHAT_CLIENT_ID" .env; then
    echo "⚠️  GigaChat credentials not configured in .env"
    echo "   Add KTTC_GIGACHAT_CLIENT_ID and KTTC_GIGACHAT_CLIENT_SECRET"
    exit 1
fi

echo "✓ GigaChat credentials found"
echo ""

# Check package installation
if ! $PYTHON -c "import kttc" 2>/dev/null; then
    echo "⚠️  KTTC is not installed"
    echo "   Run: $PYTHON -m pip install -e \".[dev]\""
    exit 1
fi

echo "✓ KTTC is installed"
echo ""

# Run tests
echo "▶️  Running tests..."
echo "======================================"
echo ""

# Select run mode
if [ "$1" = "fast" ]; then
    echo "🚀 Fast mode (authentication and basic tests only)"
    $PYTHON -m pytest tests/integration/test_gigachat_integration.py::TestGigaChatAuthentication -v
    $PYTHON -m pytest tests/integration/test_gigachat_integration.py::TestGigaChatCompletion::test_completion_simple -v
elif [ "$1" = "full" ]; then
    echo "🔬 Full mode (all tests with detailed output)"
    $PYTHON -m pytest tests/integration/test_gigachat_integration.py -v -s
elif [ "$1" = "auth" ]; then
    echo "🔐 Authentication only"
    $PYTHON -m pytest tests/integration/test_gigachat_integration.py::TestGigaChatAuthentication -v -s
elif [ "$1" = "orchestrator" ]; then
    echo "🎭 Orchestrator only (quality evaluation)"
    $PYTHON -m pytest tests/integration/test_gigachat_integration.py::TestGigaChatWithOrchestrator -v -s
else
    echo "📋 Standard mode (all tests)"
    $PYTHON -m pytest tests/integration/test_gigachat_integration.py -v
fi

echo ""
echo "======================================"
echo "✅ Tests completed!"
echo ""
echo "Available modes:"
echo "  ./run_gigachat_tests.sh          - All tests"
echo "  ./run_gigachat_tests.sh fast     - Fast tests"
echo "  ./run_gigachat_tests.sh full     - With output"
echo "  ./run_gigachat_tests.sh auth     - Authentication only"
echo "  ./run_gigachat_tests.sh orchestrator - Quality evaluation only"
