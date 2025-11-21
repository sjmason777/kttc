# راهنمای نصب

این راهنما تمام روش‌های نصب و وابستگی‌های اختیاری را پوشش می‌دهد.

## الزامات

- **Python:** 3.11 یا بالاتر
- **سیستم‌عامل‌ها:** Linux، macOS، Windows
- **کلید API:** OpenAI، Anthropic، GigaChat، یا Yandex

## نصب پایه

### استفاده از pip (توصیه می‌شود)

```bash
pip install kttc
```

این وابستگی‌های اصلی (~50MB) را نصب می‌کند:
- رابط CLI
- سیستم QA چند عامله
- NLP روسی (کتابخانه‌های MAWO)
- پشتیبانی چند زبانه پایه (spaCy، jieba)

### استفاده از pipx (محیط جداگانه)

```bash
pipx install kttc
```

مزایا:
- جدا از Python سیستم
- پیکربندی خودکار PATH
- ارتقاء آسان

### از منبع (توسعه)

```bash
git clone https://github.com/kttc-ai/kttc.git
cd kttc
pip install -e ".[dev]"
```

## وابستگی‌های اختیاری

### پشتیبانی زبان انگلیسی

بررسی پیشرفته دستور زبان انگلیسی با LanguageTool را اضافه می‌کند:

```bash
pip install kttc[english]
```

**الزامات:**
- Java 17.0 یا بالاتر
- ~200MB فضای دیسک

ویژگی‌ها:
- بیش از 5000 قانون دستوری
- توافق فاعل و فعل
- بررسی حرف تعریف (a/an/the)
- اعتبارسنجی حرف اضافه

### پشتیبانی زبان چینی

NLP پیشرفته چینی با HanLP را اضافه می‌کند:

```bash
pip install kttc[chinese]
```

ویژگی‌ها:
- اعتبارسنجی کلمات اندازه‌گیری (量词)
- بررسی ذرات جنبه (了/过)
- برچسب‌گذاری POS با دقت بالا (~95%)
- دانلود مدل ~300MB در اولین استفاده

### پشتیبانی زبان هندی

NLP پیشرفته هندی با Indic NLP + Stanza + Spello را اضافه می‌کند:

```bash
pip install kttc[hindi]
```

ویژگی‌ها:
- پردازش خط دیوناگری
- برچسب‌گذاری POS و NER هندی
- بررسی املا
- اعتبارسنجی دستور زبان

### پشتیبانی زبان فارسی

NLP پیشرفته فارسی با DadmaTools را اضافه می‌کند:

```bash
pip install kttc[persian]
```

ویژگی‌ها:
- پردازش خط فارسی
- برچسب‌گذاری POS و NER
- بررسی املا
- اعتبارسنجی دستور زبان

### تمام کمک‌کننده‌های زبان

تمام پیشرفت‌های خاص زبان را نصب کنید:

```bash
pip install kttc[all-languages]
```

معادل با:

```bash
pip install kttc[english,chinese,hindi,persian]
```

## تأیید نصب

نسخه KTTC را بررسی کنید:

```bash
kttc --version
```

با حالت نمایشی تست کنید (نیاز به کلید API ندارد):

```bash
echo "Hello" > source.txt
echo "سلام" > trans.txt
kttc check source.txt trans.txt --source-lang en --target-lang fa --demo
```

## راه‌اندازی کلید API

### OpenAI

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

### Anthropic

```bash
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### پیکربندی دائمی

به پروفایل شل خود (`~/.bashrc`، `~/.zshrc`) اضافه کنید:

```bash
# کلیدهای API KTTC
export KTTC_OPENAI_API_KEY="sk-..."
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

سپس بارگذاری مجدد کنید:

```bash
source ~/.bashrc  # یا ~/.zshrc
```

## ارتقاء

### ارتقاء به آخرین نسخه

```bash
pip install --upgrade kttc
```

### ارتقاء با وابستگی‌های اختیاری

```bash
pip install --upgrade kttc[all-languages,metrics]
```

## حذف نصب

```bash
pip uninstall kttc
```

## عیب‌یابی

### مشکلات نسخه Python

**خطا:** `TypeError: unsupported operand type(s) for |`

**راه‌حل:** از Python 3.11+ استفاده کنید:

```bash
python3.11 -m pip install kttc
```

### اجازه رد شد

**خطا:** `ERROR: Could not install packages`

**راه‌حل:** از نصب کاربر استفاده کنید:

```bash
pip install --user kttc
```

یا از محیط مجازی استفاده کنید:

```bash
python3 -m venv venv
source venv/bin/activate
pip install kttc
```

## مراحل بعدی

- [پیکربندی](configuration.md) - تنظیمات را پیکربندی کنید
- [استفاده از CLI](cli-usage.md) - دستورات را یاد بگیرید
- [شروع سریع](../tutorials/quickstart.md) - اولین بررسی خود را اجرا کنید
