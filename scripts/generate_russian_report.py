#!/usr/bin/env python3
"""Generate detailed Russian report from article check results."""

import json
from datetime import datetime
from pathlib import Path


def load_latest_results():
    """Load the most recent article check results."""
    results_dir = Path("benchmark_results")

    # Find latest file
    json_files = list(results_dir.glob("final_article_check_*.json"))
    if not json_files:
        print("❌ No results found!")
        return None

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📄 Loading results from: {latest_file.name}")

    with open(latest_file, encoding="utf-8") as f:
        return json.load(f)


def format_errors_russian(errors):
    """Format errors in Russian."""
    if not errors:
        return "Ошибок не обнаружено ✅"

    by_severity = {"critical": [], "major": [], "minor": [], "neutral": []}
    for error in errors:
        severity = error["severity"]
        by_severity[severity].append(error)

    result = []

    severity_names = {
        "critical": "Критические ошибки",
        "major": "Значительные ошибки",
        "minor": "Незначительные ошибки",
        "neutral": "Нейтральные замечания",
    }

    for sev_key, sev_name in severity_names.items():
        errors_list = by_severity[sev_key]
        if errors_list:
            result.append(f"\n**{sev_name} ({len(errors_list)}):**\n")
            for i, err in enumerate(errors_list, 1):
                agent = err.get("agent", "unknown")
                category = err["category"]
                desc = err["description"]
                result.append(f"{i}. [{agent}] {category}: {desc}\n")

    return "".join(result)


def generate_report(data):
    """Generate Russian report."""

    report = []

    report.append("# ДЕТАЛЬНЫЙ ОТЧЕТ ПРОВЕРКИ КАЧЕСТВА ПЕРЕВОДОВ\n")
    report.append(f"\n**Дата:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
    report.append(
        '**Статья:** "Translation-induced conversion killers: 7 invisible mistakes costing you international sales"\n'
    )
    report.append("**Языки:** Английский ↔ Русский ↔ Китайский\n")
    report.append("**Система проверки:** KTTC (Korona Translation Quality Control)\n")
    report.append("\n---\n")

    # Executive Summary
    report.append("\n## 📋 EXECUTIVE SUMMARY\n")
    summary = data["summary"]
    report.append(f"\n- **Средний MQM Score:** {summary['avg_mqm']:.2f}/100\n")
    report.append(f"- **Процент успешных переводов (≥95 MQM):** {summary['pass_rate']:.1f}%\n")
    report.append(f"- **Общее количество найденных ошибок:** {summary['total_errors']}\n")
    report.append(f"- **Время обработки:** {summary['total_time']:.1f} секунд\n")
    report.append("\n---\n")

    # Results by pair
    report.append("\n## 🎯 РЕЗУЛЬТАТЫ ПО КАЖДОЙ ПАРЕ\n")

    for i, result in enumerate(data["results"], 1):
        pair = result["pair"]
        mqm = result["mqm_score"]
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        errors = result["errors"]

        report.append(f"\n### {i}. {pair}\n")
        report.append(f"\n**MQM Score:** {mqm:.2f}/100\n")
        report.append(f"**Статус:** {status}\n")
        report.append(f"**Количество ошибок:** {result['error_count']}\n")

        if errors:
            report.append("\n#### Найденные ошибки:\n")
            report.append(format_errors_russian(errors))

        report.append("\n---\n")

    # Summary table
    report.append("\n## 📊 СВОДНАЯ СТАТИСТИКА\n")
    report.append(
        "\n| Пара перевода | MQM Score | Статус | Критические | Значительные | Незначительные | Нейтральные | Всего ошибок |\n"
    )
    report.append(
        "|--------------|-----------|--------|-------------|--------------|----------------|-------------|-------------|\n"
    )

    total_critical = 0
    total_major = 0
    total_minor = 0
    total_neutral = 0

    for result in data["results"]:
        pair = result["pair"]
        mqm = result["mqm_score"]
        status = "✅" if result["pass"] else "❌"

        critical = sum(1 for e in result["errors"] if e["severity"] == "critical")
        major = sum(1 for e in result["errors"] if e["severity"] == "major")
        minor = sum(1 for e in result["errors"] if e["severity"] == "minor")
        neutral = sum(1 for e in result["errors"] if e["severity"] == "neutral")
        total = result["error_count"]

        total_critical += critical
        total_major += major
        total_minor += minor
        total_neutral += neutral

        report.append(
            f"| {pair:12s} | {mqm:9.2f} | {status:6s} | {critical:11d} | {major:12d} | {minor:14d} | {neutral:11d} | {total:12d} |\n"
        )

    avg_mqm = summary["avg_mqm"]
    report.append(
        f"| **Среднее**  | **{avg_mqm:7.2f}** | --     | **{total_critical:9d}** | **{total_major:10d}** | **{total_minor:12d}** | **{total_neutral:9d}** | **{summary['total_errors']:10d}** |\n"
    )

    # Rankings
    report.append("\n---\n")
    report.append("\n## 🎯 РЕЙТИНГ ПАР ПО КАЧЕСТВУ\n")

    sorted_results = sorted(data["results"], key=lambda x: x["mqm_score"], reverse=True)

    report.append("\n### Лучшие пары (MQM ≥95):\n")
    for i, result in enumerate([r for r in sorted_results if r["mqm_score"] >= 95], 1):
        report.append(
            f"{i}. **{result['pair']}** - MQM {result['mqm_score']:.2f} - {result['error_count']} ошибок\n"
        )

    report.append("\n### Требуют внимания (MQM <95):\n")
    needs_attention = [r for r in sorted_results if r["mqm_score"] < 95]
    if needs_attention:
        for i, result in enumerate(needs_attention, 1):
            report.append(
                f"{i}. **{result['pair']}** - MQM {result['mqm_score']:.2f} - {result['error_count']} ошибок\n"
            )
    else:
        report.append("Все пары соответствуют стандарту качества! ✅\n")

    # Most errors
    report.append("\n### Пары с наибольшим количеством ошибок:\n")
    by_errors = sorted(data["results"], key=lambda x: x["error_count"], reverse=True)
    for i, result in enumerate(by_errors[:3], 1):
        report.append(
            f"{i}. **{result['pair']}** - {result['error_count']} ошибок (MQM: {result['mqm_score']:.2f})\n"
        )

    # Conclusions
    report.append("\n---\n")
    report.append("\n## 💡 ВЫВОДЫ\n")

    report.append("\n### Общая оценка качества:\n")
    if summary["avg_mqm"] >= 95:
        report.append(
            f"✅ **ОТЛИЧНО** - Средний MQM {summary['avg_mqm']:.2f} превосходит стандарт качества\n"
        )
    elif summary["avg_mqm"] >= 90:
        report.append(
            f"✓ **ХОРОШО** - Средний MQM {summary['avg_mqm']:.2f} близок к стандарту качества\n"
        )
    else:
        report.append(
            f"⚠️ **ТРЕБУЕТ УЛУЧШЕНИЯ** - Средний MQM {summary['avg_mqm']:.2f} ниже стандарта\n"
        )

    report.append("\n### Статистика:\n")
    report.append(f"- Проверено пар переводов: {len(data['results'])}\n")
    report.append(f"- Успешных (≥95 MQM): {int(summary['pass_rate'])}%\n")
    report.append(f"- Всего найдено ошибок: {summary['total_errors']}\n")
    report.append(f"  - Критических: {total_critical}\n")
    report.append(f"  - Значительных: {total_major}\n")
    report.append(f"  - Незначительных: {total_minor}\n")
    report.append(f"  - Нейтральных: {total_neutral}\n")

    # Technical details
    report.append("\n---\n")
    report.append("\n## 📝 ТЕХНИЧЕСКИЕ ДЕТАЛИ\n")
    report.append("\n### Параметры проверки:\n")
    report.append(f"- **Дата проверки:** {data['timestamp']}\n")
    report.append(f"- **Статья:** {data['article']}\n")
    report.append(f"- **Языки:** {', '.join(data['languages'])}\n")
    report.append(f"- **Пар проверено:** {data['pairs_checked']}\n")

    # Next steps
    report.append("\n---\n")
    report.append("\n## 🔄 РЕКОМЕНДАЦИИ\n")
    report.append("\n### Для улучшения переводов:\n")

    if needs_attention:
        report.append("1. Уделить особое внимание парам с MQM <95:\n")
        for result in needs_attention:
            report.append(f"   - {result['pair']}: проверить и исправить найденные ошибки\n")

    if total_critical > 0:
        report.append(
            f"2. **Критичные ошибки ({total_critical}):** Требуют немедленного исправления\n"
        )

    if total_major > 0:
        report.append(f"3. **Значительные ошибки ({total_major}):** Рекомендуется исправить\n")

    report.append("\n### Для дальнейшего анализа:\n")
    report.append("1. Изучить детали каждой ошибки в JSON-отчете\n")
    report.append("2. Определить реальные ошибки vs ложные срабатывания\n")
    report.append("3. Создать список улучшений для KTTC на основе false positives\n")

    report.append("\n---\n")
    report.append(f"\n**Отчет сгенерирован:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
    report.append("**Статус:** ✅ Проверка завершена\n")

    return "".join(report)


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("ГЕНЕРАТОР РУССКОГО ОТЧЕТА")
    print("=" * 80)

    # Load results
    data = load_latest_results()
    if not data:
        return

    # Generate report
    print("\n📝 Generating Russian report...")
    report_text = generate_report(data)

    # Save report
    output_file = Path("ОТЧЕТ_ПРОВЕРКИ_ПЕРЕВОДОВ_ФИНАЛ.md")
    output_file.write_text(report_text, encoding="utf-8")

    print(f"\n✅ Report saved to: {output_file}")
    print(f"   Lines: {len(report_text.splitlines())}")
    print(f"   Characters: {len(report_text)}")

    # Print summary
    summary = data["summary"]
    print("\n📊 Summary:")
    print(f"   Average MQM: {summary['avg_mqm']:.2f}/100")
    print(f"   Pass rate: {summary['pass_rate']:.1f}%")
    print(f"   Total errors: {summary['total_errors']}")

    print("\n" + "=" * 80)
    print("✅ Done! Check ОТЧЕТ_ПРОВЕРКИ_ПЕРЕВОДОВ_ФИНАЛ.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
