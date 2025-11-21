<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/content/assets/img/kttc.logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/content/assets/img/kttc.logo.png">
      <img alt="KTTC" title="KTTC" src="docs/content/assets/img/kttc.logo.png">
    </picture>
</p>

[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · **فارسی**

# KTTC - هسته دگرگونی ترجمه دانش

> **📖 مستندات کامل فارسی:** [docs/fa/README.md](docs/fa/README.md)

---

**تضمین کیفیت ترجمه خودکار مبتنی بر هوش مصنوعی**

KTTC از سیستم‌های چند عامله تخصصی برای تشخیص، تحلیل و رفع خودکار مشکلات کیفیت ترجمه بر اساس چارچوب استاندارد صنعتی MQM (Multidimensional Quality Metrics) استفاده می‌کند. کیفیت ترجمه آماده تولید را در عرض چند ثانیه دریافت کنید.

---

## ویژگی‌های کلیدی

- **سیستم تضمین کیفیت چند عامله** - 5 عامل تخصصی دقت، روانی، اصطلاحات، توهم و زمینه را تحلیل می‌کنند
- **امتیازدهی MQM** - معیارهای کیفیت استاندارد صنعتی مورد استفاده در معیارهای WMT
- **مسیریابی هوشمند** - به طور خودکار مدل‌های بهینه را بر اساس پیچیدگی متن انتخاب می‌کند (60٪ صرفه‌جویی در هزینه)
- **اصلاح خودکار** - رفع خطای مبتنی بر LLM با بهبود تکراری (حلقه TEaR)
- **عامل‌های خاص زبان** - بررسی‌های سطح بومی برای انگلیسی، چینی، روسی، هندی و فارسی
- **حافظه ترجمه** - جستجوی معنایی با ردیابی کیفیت و استفاده مجدد
- **مدیریت واژه‌نامه** - اعتبارسنجی اصطلاحات سفارشی و ثبات
- **پردازش دسته‌ای** - پردازش هزاران ترجمه به صورت موازی
- **آماده CI/CD** - ادغام GitHub Actions، کدهای خروج، فرمت‌های خروجی متعدد
- **پشتیبانی چند LLM** - OpenAI، Anthropic، GigaChat، YandexGPT

**عملکرد:** 90٪ کاهش هزینه در مقابل بررسی دستی • 100-1000 برابر سریعتر • هدف کیفیت MQM بالای 95

---

## 🚀 KTTC را به صورت آنلاین امتحان کنید

KTTC را بدون نصب تجربه کنید:

[![Open in Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kttc-ai/kttc/blob/main/examples/kttc_demo.ipynb)
[![Streamlit Demo](https://img.shields.io/badge/Streamlit_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://kttc-demo.streamlit.app)
[![Open in Codespaces](https://img.shields.io/badge/Open_in_Codespaces-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/codespaces/new?repo=kttc-ai/kttc)

- **Google Colab** - آموزش تعاملی با مثال‌ها (5 دقیقه، بدون نیاز به تنظیمات)
- **Streamlit Demo** - رابط وب برای آزمایش ترجمه‌های خود (بدون نیاز به کد)
- **GitHub Codespaces** - محیط توسعه کامل در مرورگر (برای مشارکت‌کنندگان)

---

## شروع سریع

### 1. نصب

```bash
pip install kttc
```

پیشرفت‌های زبانی اختیاری:

```bash
pip install kttc[english]        # انگلیسی: LanguageTool (بیش از 5000 قانون دستوری)
pip install kttc[chinese]        # چینی: HanLP (کلمات اندازه‌گیری، ذرات)
pip install kttc[hindi]          # هندی: Indic NLP + Stanza + Spello
pip install kttc[persian]        # فارسی: DadmaTools (مبتنی بر spaCy)
pip install kttc[all-languages]  # تمام کمک‌کننده‌های زبان
```

### 2. تنظیم کلید API

```bash
export KTTC_OPENAI_API_KEY="sk-..."
# یا
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. بررسی کیفیت ترجمه

```bash
kttc check source.txt translation.txt --source-lang en --target-lang fa
```

**خروجی:**

```
✅ امتیاز MQM: 96.5 (قبول - کیفیت عالی)
📊 5 عامل ترجمه را تحلیل کردند
⚠️  2 مشکل جزئی، 0 اصلی، 0 حیاتی یافت شد
✓ آستانه کیفیت برآورده شد (≥95.0)
```

همین! KTTC با تنظیمات پیش‌فرض هوشمند از جعبه کار می‌کند:
- ✅ مسیریابی هوشمند (به طور خودکار مدل‌های ارزان‌تر را برای متون ساده انتخاب می‌کند)
- ✅ واژه‌نامه خودکار (در صورت وجود از واژه‌نامه 'base' استفاده می‌کند)
- ✅ فرمت خودکار (فرمت خروجی را از پسوند فایل تشخیص می‌دهد)

---

## دستورات

```bash
kttc check source.txt translation.txt          # بررسی کیفیت تک
kttc check source.txt t1.txt t2.txt t3.txt     # مقایسه خودکار چندین ترجمه
kttc check translations.csv                     # تشخیص خودکار حالت دسته‌ای (CSV/JSON)
kttc check source_dir/ trans_dir/              # تشخیص خودکار حالت دسته‌ای دایرکتوری

kttc batch --file translations.csv              # پردازش دسته‌ای صریح
kttc compare --source src.txt -t t1 -t t2      # مقایسه ترجمه‌ها کنار هم
kttc translate --text "Hello" --source-lang en --target-lang fa  # ترجمه با QA
kttc benchmark --source text.txt --providers openai,anthropic    # معیار LLM

# مدیریت واژه‌نامه (ذخیره‌سازی پروژه + کاربر جهانی)
kttc glossary list                              # فهرست تمام واژه‌نامه‌ها
kttc glossary create tech --from-csv terms.csv  # ایجاد واژه‌نامه پروژه
kttc glossary create personal --from-csv my.csv --user  # ایجاد واژه‌نامه کاربر
```

**مراجعه کامل دستورات را ببینید:** [docs/fa/reference/cli-commands.md](docs/fa/reference/cli-commands.md)

---

## API پایتون

```python
import asyncio
from kttc.agents import AgentOrchestrator
from kttc.llm import OpenAIProvider
from kttc.core import TranslationTask

async def check_quality():
    llm = OpenAIProvider(api_key="your-key")
    orchestrator = AgentOrchestrator(llm)

    task = TranslationTask(
        source_text="Hello, world!",
        translation="سلام دنیا!",
        source_lang="en",
        target_lang="fa",
    )

    report = await orchestrator.evaluate(task)
    print(f"امتیاز MQM: {report.mqm_score}")
    print(f"وضعیت: {report.status}")

asyncio.run(check_quality())
```

**مراجعه کامل API را ببینید:** [docs/fa/reference/api-reference.md](docs/fa/reference/api-reference.md)

---

## 📚 مستندات

**مستندات کامل به زبان فارسی موجود است:** [docs/fa/README.md](docs/fa/README.md)

### لینک‌های سریع

- **[راهنمای شروع سریع](docs/fa/tutorials/README.md)** - در 5 دقیقه شروع کنید
- **[راهنمای نصب](docs/fa/guides/README.md)** - دستورالعمل‌های تنظیم دقیق
- **[مرجع CLI](docs/fa/reference/README.md)** - تمام دستورات و گزینه‌ها
- **[معماری](docs/fa/explanation/README.md)** - چگونه KTTC کار می‌کند

### ساختار مستندات

پیروی از چارچوب [Diátaxis](https://diataxis.fr/):

- 📚 **[آموزش‌ها](docs/fa/tutorials/README.md)** - با انجام دادن یاد بگیرید (راهنماهای گام به گام)
- 📖 **[راهنماها](docs/fa/guides/README.md)** - مشکلات خاص را حل کنید (راهنماهای چگونه)
- 📋 **[مرجع](docs/fa/reference/README.md)** - جزئیات فنی را جستجو کنید (API، CLI)
- 💡 **[توضیح](docs/fa/explanation/README.md)** - مفاهیم را درک کنید (معماری، طراحی)

---

## مشارکت

ما از مشارکت استقبال می‌کنیم! برای راهنماها به [CONTRIBUTING.md](CONTRIBUTING.md) مراجعه کنید.

---

## مجوز

تحت مجوز Apache License 2.0. برای جزئیات به [LICENSE](LICENSE) مراجعه کنید.

کپی‌رایت 2025 KTTC AI (https://github.com/kttc-ai)

---

## لینک‌ها

- 📦 [بسته PyPI](https://pypi.org/project/kttc/)
- 📖 [مستندات](docs/fa/)
- 🐛 [ردیاب مسائل](https://github.com/kttc-ai/kttc/issues)
- 💬 [بحث‌ها](https://github.com/kttc-ai/kttc/discussions)
- 🇺🇸 [English Version](README.md)
