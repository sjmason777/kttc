#!/usr/bin/env python3
"""Final check of article translations: EN ↔ RU ↔ ZH (6 pairs).

This script checks all translation pairs of a real article and identifies:
1. Real translation errors
2. False positives (bugs in KTTC)
3. Quality assessment for each pair
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core import TranslationTask
from kttc.utils.config import get_settings

# File paths
FILES = {
    "en": Path("examples/guide/translation-induced/en.txt"),
    "ru": Path("examples/guide/translation-induced/ru.txt"),
    "zh": Path("examples/guide/translation-induced/zh.txt"),
}

# All 6 translation pairs to check
PAIRS = [
    ("en", "ru", "English → Russian"),
    ("en", "zh", "English → Chinese"),
    ("ru", "en", "Russian → English"),
    ("ru", "zh", "Russian → Chinese"),
    ("zh", "en", "Chinese → English"),
    ("zh", "ru", "Chinese → Russian"),
]


async def check_translation_pair(
    orchestrator: AgentOrchestrator,
    source_text: str,
    translation: str,
    src_lang: str,
    tgt_lang: str,
    pair_name: str,
) -> dict:
    """Check a single translation pair.

    Args:
        orchestrator: QA orchestrator
        source_text: Source text
        translation: Translation text
        src_lang: Source language code
        tgt_lang: Target language code
        pair_name: Human-readable pair name

    Returns:
        Dict with results
    """
    print(f"\n{'=' * 80}")
    print(f"Checking: {pair_name}")
    print(f"{'=' * 80}")

    # Create task
    task = TranslationTask(
        source_text=source_text,
        translation=translation,
        source_lang=src_lang,
        target_lang=tgt_lang,
    )

    # Evaluate
    try:
        report = await orchestrator.evaluate(task)

        # Print summary
        print(f"\n✅ MQM Score: {report.mqm_score:.2f}/100")
        print(f"   Status: {'PASS' if report.status == 'pass' else 'FAIL'}")
        print(f"   Errors found: {len(report.errors)}")

        if report.errors:
            print("\n   Error breakdown:")
            for severity in ["critical", "major", "minor"]:
                count = sum(1 for e in report.errors if e.severity.value == severity)
                if count > 0:
                    print(f"   - {severity.capitalize()}: {count}")

        return {
            "pair": pair_name,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "mqm_score": report.mqm_score,
            "pass": report.status == "pass",
            "error_count": len(report.errors),
            "errors": [
                {
                    "category": e.category,
                    "subcategory": e.subcategory,
                    "severity": e.severity.value,
                    "description": e.description,
                    "location": e.location,
                    "suggestion": e.suggestion,
                }
                for e in report.errors
            ],
            "agents_used": list(report.agent_scores.keys()) if report.agent_scores else [],
            "processing_time": (
                report.consensus_metadata.get("total_time", 0) if report.consensus_metadata else 0
            ),
        }

    except Exception as e:
        print(f"\n❌ Error evaluating {pair_name}: {e}")
        import traceback

        traceback.print_exc()
        return {
            "pair": pair_name,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "mqm_score": 0,
            "pass": False,
            "error_count": 0,
            "errors": [],
            "error_message": str(e),
        }


async def main():
    """Main function to check all translation pairs."""
    print("\n" + "=" * 80)
    print("FINAL ARTICLE CHECK: All Translation Pairs")
    print("=" * 80)
    print("\nArticle: Translation-induced conversion killers")
    print("Languages: EN ↔ RU ↔ ZH")
    print(f"Pairs to check: {len(PAIRS)}")
    print("\n")

    # Read all files
    print("Reading files...")
    texts = {}
    for lang, path in FILES.items():
        if not path.exists():
            print(f"❌ File not found: {path}")
            return

        texts[lang] = path.read_text(encoding="utf-8").strip()
        lines = len(texts[lang].split("\n"))
        chars = len(texts[lang])
        print(f"  ✅ {lang.upper()}: {lines} lines, {chars} characters")

    # Initialize
    print("\nInitializing KTTC QA system...")
    settings = get_settings()

    # Try to get LLM
    try:
        api_key = settings.get_llm_provider_key("anthropic")
        from kttc.llm import AnthropicProvider

        llm = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
        print("✅ Using Anthropic Claude 3.5 Haiku")
    except Exception:
        try:
            api_key = settings.get_llm_provider_key("openai")
            from kttc.llm import OpenAIProvider

            llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
            print("✅ Using OpenAI GPT-4o-mini")
        except Exception:
            print("❌ No LLM provider available")
            return

    # Create orchestrator
    orchestrator = AgentOrchestrator(llm)

    # Check all pairs
    results = []
    for i, (src_lang, tgt_lang, pair_name) in enumerate(PAIRS, 1):
        print(f"\n[{i}/{len(PAIRS)}] Processing: {pair_name}")

        result = await check_translation_pair(
            orchestrator=orchestrator,
            source_text=texts[src_lang],
            translation=texts[tgt_lang],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            pair_name=pair_name,
        )

        results.append(result)

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"final_article_check_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "article": "Translation-induced conversion killers",
        "languages": ["en", "ru", "zh"],
        "pairs_checked": len(PAIRS),
        "summary": {
            "avg_mqm": sum(r["mqm_score"] for r in results) / len(results),
            "pass_rate": sum(1 for r in results if r["pass"]) / len(results) * 100,
            "total_errors": sum(r["error_count"] for r in results),
            "total_time": sum(r.get("processing_time", 0) for r in results),
        },
        "results": results,
    }

    output_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")
    print("\nOverall Statistics:")
    print(f"  Average MQM Score: {output_data['summary']['avg_mqm']:.2f}/100")
    print(f"  Pass Rate: {output_data['summary']['pass_rate']:.1f}%")
    print(f"  Total Errors Found: {output_data['summary']['total_errors']}")
    print(f"  Total Processing Time: {output_data['summary']['total_time']:.1f}s")

    print("\nResults by Pair:")
    for result in results:
        status = "✅" if result["pass"] else "❌"
        print(
            f"  {status} {result['pair']:30s}: MQM {result['mqm_score']:6.2f}, "
            f"Errors: {result['error_count']:3d}"
        )

    # Identify worst pairs
    print("\nPairs with Most Errors:")
    sorted_results = sorted(results, key=lambda x: x["error_count"], reverse=True)
    for result in sorted_results[:3]:
        print(
            f"  {result['pair']:30s}: {result['error_count']} errors (MQM: {result['mqm_score']:.2f})"
        )

    print("\n" + "=" * 80)
    print("Check complete! Use this data to generate the Russian report.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
