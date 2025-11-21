#!/bin/bash
# Demo commands for KTTC CLI testing
# Run from project root: bash examples/cli/demo_commands.sh

set -e

EXAMPLES_DIR="examples/cli"
DEMO_MODE="--demo"  # Use --demo for testing without API keys

echo "========================================"
echo "KTTC CLI Demo Commands"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. CHECK COMMAND - Single translation quality check
echo -e "${BLUE}1. CHECK COMMAND - Single translation quality check${NC}"
echo "   Compact mode (default):"
echo "   $ kttc check ${EXAMPLES_DIR}/source_en_realistic.txt ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt --source-lang en --target-lang ru ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc check ${EXAMPLES_DIR}/source_en_realistic.txt ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt --source-lang en --target-lang ru ${DEMO_MODE}
echo ""
echo "---"
echo ""

# 2. CHECK COMMAND - Verbose mode
echo -e "${BLUE}2. CHECK COMMAND - Verbose mode (detailed)${NC}"
echo "   $ kttc check ${EXAMPLES_DIR}/source_en_realistic.txt ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt --source-lang en --target-lang ru --verbose ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc check ${EXAMPLES_DIR}/source_en_realistic.txt ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt --source-lang en --target-lang ru --verbose ${DEMO_MODE}
echo ""
echo "---"
echo ""

# 3. CHECK COMMAND - Good translation (should PASS)
echo -e "${BLUE}3. CHECK COMMAND - Good translation (should PASS)${NC}"
echo "   $ kttc check ${EXAMPLES_DIR}/source_en_realistic.txt ${EXAMPLES_DIR}/translation_ru_realistic_good.txt --source-lang en --target-lang ru ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc check ${EXAMPLES_DIR}/source_en_realistic.txt ${EXAMPLES_DIR}/translation_ru_realistic_good.txt --source-lang en --target-lang ru ${DEMO_MODE} || true
echo ""
echo "---"
echo ""

# 4. COMPARE COMMAND - Compare multiple translations
echo -e "${BLUE}4. COMPARE COMMAND - Compare multiple translations${NC}"
echo "   Compact mode:"
echo "   $ kttc compare --source ${EXAMPLES_DIR}/source_en_realistic.txt \\"
echo "                  --translation ${EXAMPLES_DIR}/translation_ru_realistic_good.txt \\"
echo "                  --translation ${EXAMPLES_DIR}/translation_ru_realistic_medium.txt \\"
echo "                  --translation ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt \\"
echo "                  --source-lang en --target-lang ru ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc compare --source ${EXAMPLES_DIR}/source_en_realistic.txt \
                           --translation ${EXAMPLES_DIR}/translation_ru_realistic_good.txt \
                           --translation ${EXAMPLES_DIR}/translation_ru_realistic_medium.txt \
                           --translation ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt \
                           --source-lang en --target-lang ru ${DEMO_MODE}
echo ""
echo "---"
echo ""

# 5. COMPARE COMMAND - Verbose mode
echo -e "${BLUE}5. COMPARE COMMAND - Verbose mode${NC}"
echo "   $ kttc compare --source ${EXAMPLES_DIR}/source_en_realistic.txt \\"
echo "                  --translation ${EXAMPLES_DIR}/translation_ru_realistic_good.txt \\"
echo "                  --translation ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt \\"
echo "                  --source-lang en --target-lang ru --verbose ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc compare --source ${EXAMPLES_DIR}/source_en_realistic.txt \
                           --translation ${EXAMPLES_DIR}/translation_ru_realistic_good.txt \
                           --translation ${EXAMPLES_DIR}/translation_ru_realistic_bad.txt \
                           --source-lang en --target-lang ru --verbose ${DEMO_MODE}
echo ""
echo "---"
echo ""

# 6. BATCH COMMAND - Process CSV file
echo -e "${BLUE}6. BATCH COMMAND - Process CSV file${NC}"
echo "   $ kttc batch --file ${EXAMPLES_DIR}/batch_demo.csv --output /tmp/batch_results.json ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc batch --file ${EXAMPLES_DIR}/batch_demo.csv --output /tmp/batch_results.json ${DEMO_MODE}
echo ""
echo "---"
echo ""

# 7. BENCHMARK COMMAND - Compare providers (demo mode)
echo -e "${BLUE}7. BENCHMARK COMMAND - Compare providers${NC}"
echo "   $ kttc benchmark --source ${EXAMPLES_DIR}/source_en.txt \\"
echo "                    --source-lang en --target-lang ru \\"
echo "                    --providers openai,anthropic ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc benchmark --source ${EXAMPLES_DIR}/source_en.txt \
                             --source-lang en --target-lang ru \
                             --providers openai,anthropic ${DEMO_MODE} || true
echo ""
echo "---"
echo ""

# 8. CHECK with OUTPUT formats
echo -e "${BLUE}8. CHECK COMMAND - With different output formats${NC}"
echo "   JSON output:"
echo "   $ kttc check ${EXAMPLES_DIR}/source_en.txt ${EXAMPLES_DIR}/translation_ru_bad.txt \\"
echo "                --source-lang en --target-lang ru \\"
echo "                --output /tmp/report.json ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc check ${EXAMPLES_DIR}/source_en.txt ${EXAMPLES_DIR}/translation_ru_bad.txt \
                         --source-lang en --target-lang ru \
                         --output /tmp/report.json ${DEMO_MODE}
echo ""
echo "   Output saved to: /tmp/report.json"
echo ""

echo "   Markdown output:"
echo "   $ kttc check ${EXAMPLES_DIR}/source_en.txt ${EXAMPLES_DIR}/translation_ru_bad.txt \\"
echo "                --source-lang en --target-lang ru \\"
echo "                --output /tmp/report.md ${DEMO_MODE}"
echo ""
read -p "Press Enter to run..."
python3.11 -m kttc check ${EXAMPLES_DIR}/source_en.txt ${EXAMPLES_DIR}/translation_ru_bad.txt \
                         --source-lang en --target-lang ru \
                         --output /tmp/report.md ${DEMO_MODE}
echo ""
echo "   Output saved to: /tmp/report.md"
echo ""
echo "---"
echo ""

# Summary
echo -e "${GREEN}========================================"
echo "Demo completed!"
echo "========================================${NC}"
echo ""
echo "Key observations:"
echo "  1. Compact mode fits on one screen (~10-15 lines)"
echo "  2. Verbose mode shows detailed metrics (~25-35 lines)"
echo "  3. All commands use consistent formatting"
echo "  4. Colors and icons make output easy to scan"
echo ""
echo "Generated files:"
echo "  - /tmp/batch_results.json"
echo "  - /tmp/report.json"
echo "  - /tmp/report.md"
echo ""
echo "View them with:"
echo "  $ cat /tmp/batch_results.json | jq"
echo "  $ cat /tmp/report.md"
echo ""
