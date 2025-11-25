# مرجع دستورات CLI

مرجع کامل برای تمام دستورات خط فرمان KTTC.

## kttc check

بررسی‌کننده هوشمند کیفیت ترجمه با تشخیص خودکار.

### نحو دستور

```bash
kttc check SOURCE [TRANSLATIONS...] [OPTIONS]
```

### حالت‌های تشخیص خودکار

`kttc check` به طور خودکار تشخیص می‌دهد که می‌خواهید چه کاری انجام دهید:

| ورودی | حالت تشخیص داده شده | رفتار |
|-------|--------------|----------|
| `source.txt translation.txt` | بررسی تکی | ارزیابی کیفیت |
| `source.txt trans1.txt trans2.txt` | مقایسه | مقایسه خودکار |
| `translations.csv` | دسته‌ای (فایل) | پردازش CSV/JSON |
| `source_dir/ trans_dir/` | دسته‌ای (دایرکتوری) | پردازش دایرکتوری‌ها |

### گزینه‌ها

#### لازم (برای حالت‌های تکی/مقایسه)

- `--source-lang CODE` - کد زبان مبدأ (مثل `en`)
- `--target-lang CODE` - کد زبان مقصد (مثل `ru`)

#### ویژگی‌های هوشمند (به طور پیش‌فرض فعال)

- `--smart-routing` / `--no-smart-routing` - انتخاب مدل بر اساس پیچیدگی (پیش‌فرض: فعال)
- `--glossary TEXT` - واژه‌نامه‌های مورد استفاده: `auto` (پیش‌فرض)، `none`، یا نام‌های جدا شده با کاما
- `--output PATH` - تشخیص خودکار فرمت از پسوند (`.json`، `.md`، `.html`)

#### کنترل کیفیت

- `--threshold FLOAT` - حداقل امتیاز MQM (پیش‌فرض: 95.0)
- `--auto-correct` - اصلاح خودکار خطاهای شناسایی شده
- `--correction-level light|full` - سطح اصلاح (پیش‌فرض: `light`)

#### انتخاب مدل

- `--provider openai|anthropic|gigachat|yandex` - ارائه‌دهنده LLM
- `--auto-select-model` - استفاده از مدل بهینه برای جفت زبان
- `--show-routing-info` - نمایش تحلیل پیچیدگی

#### خروجی و جزئیات

- `--format text|json|markdown|html` - فرمت خروجی (تشخیص خودکار را بازنویسی می‌کند)
- `--verbose` - نمایش خروجی مفصل
- `--demo` - حالت نمایشی (بدون فراخوانی API، پاسخ‌های شبیه‌سازی شده)

#### عملکرد و هزینه (جدید در نسخه 1.1)

- `--quick`, `-q` - حالت سریع: یک پاس با 3 عامل اصلی (accuracy، fluency، terminology). سریع‌تر و ارزان‌تر برای متون ساده.
- `--show-cost` - نمایش مصرف توکن و تخمین هزینه API پس از بررسی.

#### حالت مستندات فنی (جدید در نسخه 1.2)

KTTC به طور خودکار مستندات فنی (اسناد CLI، مراجع API، فایل‌های README) را تشخیص می‌دهد و تحلیل سبک ادبی را رد می‌کند. این امر از مثبت‌های کاذب مانند «جریان سیال ذهن» یا «الگوهای حشو» برای محتوای پر از کد جلوگیری می‌کند.

**نشانگرهای تشخیص:**
- بلوک‌های کد (` ```bash `، ` ```python `)
- گزینه‌های CLI (`--option`، `-flag`)
- سرفصل‌ها و جداول Markdown
- اختصارات فنی (API، CLI، SDK، HTTP)
- متغیرهای محیطی (KTTC_*، API_KEY)

### امتیازدهی شدت MQM (به‌روزرسانی در نسخه 1.1)

KTTC از ضرایب شدت استاندارد صنعت برای امتیازدهی MQM استفاده می‌کند:

| شدت | ضریب | تأثیر |
|-----|------|-------|
| Neutral | 0x | بدون جریمه (اطلاعاتی) |
| Minor | 1x | قابل توجه اما بر درک تأثیر نمی‌گذارد |
| Major | 5x | بر درک یا کیفیت تأثیر می‌گذارد |
| Critical | 25x | تغییر شدید معنا یا غیرقابل استفاده |

**فرمول:** `امتیاز کیفیت = 100 - (ETPT / تعداد_کلمات × ضریب_نرمال‌سازی)`

که در آن `ETPT = Σ(تعداد_خطا × ضریب_شدت)`

### نمونه‌ها

**بررسی فایل تکی:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es
```

**مقایسه چند ترجمه (تشخیص خودکار):**

```bash
kttc check source.txt trans1.txt trans2.txt trans3.txt \
  --source-lang en \
  --target-lang ru
```

**پردازش دسته‌ای CSV (تشخیص خودکار، زبان‌ها از فایل):**

```bash
kttc check translations.csv
```

**پردازش دسته‌ای دایرکتوری‌ها:**

```bash
kttc check source_dir/ translation_dir/ \
  --source-lang en \
  --target-lang ru
```

**اصلاح خودکار:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --auto-correct \
  --correction-level full
```

**گزارش HTML (تشخیص خودکار از پسوند):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --output report.html
```

**غیرفعال کردن ویژگی‌های هوشمند:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --no-smart-routing \
  --glossary none
```

**حالت نمایشی (بدون فراخوانی API):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es \
  --demo
```

**حالت سریع (سریع‌تر، ارزان‌تر):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --quick
```

**نمایش مصرف توکن و هزینه:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --show-cost
```

خروجی:
```
✓ ترجمه بررسی کیفیت را گذراند (امتیاز MQM: 96.5)
💰 توکن‌ها: 1,245 (ورودی: 890، خروجی: 355) | فراخوانی‌ها: 5 | هزینه: $0.0234
```

---

## kttc batch

پردازش دسته‌ای چندین ترجمه.

### نحو دستور

**حالت فایل:**

```bash
kttc batch --file FILE [OPTIONS]
```

**حالت دایرکتوری:**

```bash
kttc batch --source-dir DIR --translation-dir DIR \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### گزینه‌ها

#### انتخاب حالت (متقابلاً انحصاری)

- `--file PATH` - فایل دسته‌ای (CSV، JSON، یا JSONL)
- `--source-dir PATH` + `--translation-dir PATH` - حالت دایرکتوری

#### لازم (فقط حالت دایرکتوری)

- `--source-lang CODE` - کد زبان مبدأ
- `--target-lang CODE` - کد زبان مقصد

#### گزینه‌های عمومی

- `--threshold FLOAT` - حداقل امتیاز MQM (پیش‌فرض: 95.0)
- `--output PATH` - مسیر گزارش خروجی (پیش‌فرض: `report.json`)
- `--parallel INT` - تعداد کارگران موازی (پیش‌فرض: 4)
- `--glossary TEXT` - واژه‌نامه‌های مورد استفاده
- `--smart-routing` - فعال کردن مسیریابی مبتنی بر پیچیدگی
- `--show-progress` / `--no-progress` - نمایش نوار پیشرفت (پیش‌فرض: نمایش)
- `--verbose` - خروجی مفصل
- `--demo` - حالت نمایشی

#### فقط حالت فایل

- `--batch-size INT` - اندازه دسته برای گروه‌بندی

### فرمت‌های فایل پشتیبانی شده

**CSV:**

```csv
source,translation,source_lang,target_lang,domain
"Hello world","Hola mundo","en","es","general"
```

**JSON:**

```json
[
  {
    "source": "Hello world",
    "translation": "Hola mundo",
    "source_lang": "en",
    "target_lang": "es",
    "domain": "general"
  }
]
```

**JSONL:**

```jsonl
{"source": "Hello world", "translation": "Hola mundo", "source_lang": "en", "target_lang": "es"}
{"source": "Good morning", "translation": "Buenos días", "source_lang": "en", "target_lang": "es"}
```

### نمونه‌ها

**پردازش فایل CSV:**

```bash
kttc batch --file translations.csv
```

**پردازش JSON با نمایش پیشرفت:**

```bash
kttc batch --file translations.json \
  --show-progress \
  --output results.json
```

**حالت دایرکتوری:**

```bash
kttc batch \
  --source-dir ./source \
  --translation-dir ./translations \
  --source-lang en \
  --target-lang es \
  --parallel 8
```

---

## kttc compare

مقایسه چندین ترجمه در کنار یکدیگر.

### نحو دستور

```bash
kttc compare --source FILE \
  --translation FILE --translation FILE [...] \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### گزینه‌ها

- `--source PATH` - فایل متن مبدأ (لازم)
- `--translation PATH` - فایل ترجمه (می‌توان چندین بار مشخص کرد، لازم)
- `--source-lang CODE` - کد زبان مبدأ (لازم)
- `--target-lang CODE` - کد زبان مقصد (لازم)
- `--threshold FLOAT` - آستانه کیفیت (پیش‌فرض: 95.0)
- `--provider TEXT` - ارائه‌دهنده LLM
- `--verbose` - نمایش مقایسه مفصل

### نمونه‌ها

**مقایسه 3 ترجمه:**

```bash
kttc compare \
  --source text.txt \
  --translation trans1.txt \
  --translation trans2.txt \
  --translation trans3.txt \
  --source-lang en \
  --target-lang ru \
  --verbose
```

---

## kttc translate

ترجمه متن با بررسی و بهبود خودکار کیفیت.

### نحو دستور

```bash
kttc translate --text TEXT \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### گزینه‌ها

- `--text TEXT` - متن برای ترجمه (یا `@file.txt` برای ورودی فایل، لازم)
- `--source-lang CODE` - کد زبان مبدأ (لازم)
- `--target-lang CODE` - کد زبان مقصد (لازم)
- `--threshold FLOAT` - آستانه کیفیت برای بهبود (پیش‌فرض: 95.0)
- `--max-iterations INT` - حداکثر تکرار بهبود (پیش‌فرض: 3)
- `--output PATH` - مسیر فایل خروجی
- `--provider TEXT` - ارائه‌دهنده LLM
- `--verbose` - خروجی مفصل

### نمونه‌ها

**ترجمه متن درون‌خطی:**

```bash
kttc translate --text "Hello, world!" \
  --source-lang en \
  --target-lang es
```

**ترجمه از فایل:**

```bash
kttc translate --text @document.txt \
  --source-lang en \
  --target-lang ru \
  --output translated.txt
```

**با آستانه کیفیت:**

```bash
kttc translate --text "Complex technical text" \
  --source-lang en \
  --target-lang zh \
  --threshold 98 \
  --max-iterations 5
```

---

## kttc benchmark

معیارسنجی چندین ارائه‌دهنده LLM.

### نحو دستور

```bash
kttc benchmark --source FILE \
  --source-lang CODE --target-lang CODE \
  --providers LIST [OPTIONS]
```

### گزینه‌ها

- `--source PATH` - فایل متن مبدأ (لازم)
- `--source-lang CODE` - کد زبان مبدأ (لازم)
- `--target-lang CODE` - کد زبان مقصد (لازم)
- `--providers TEXT` - لیست ارائه‌دهندگان جدا شده با کاما (پیش‌فرض: `gigachat,openai,anthropic`)
- `--threshold FLOAT` - آستانه کیفیت (پیش‌فرض: 95.0)
- `--output PATH` - مسیر فایل خروجی (JSON)
- `--verbose` - خروجی مفصل

### نمونه‌ها

**معیارسنجی تمام ارائه‌دهندگان:**

```bash
kttc benchmark \
  --source text.txt \
  --source-lang en \
  --target-lang ru \
  --providers gigachat,openai,anthropic
```

---

## kttc report

تولید گزارش‌های فرمت شده از نتایج QA.

### نحو دستور

```bash
kttc report INPUT_FILE [OPTIONS]
```

### گزینه‌ها

- `--format markdown|html` - فرمت خروجی (پیش‌فرض: markdown)
- `--output PATH` - مسیر فایل خروجی (در صورت عدم مشخص، خودکار تولید می‌شود)

### نمونه‌ها

**تولید گزارش Markdown:**

```bash
kttc report results.json --format markdown -o report.md
```

**تولید گزارش HTML:**

```bash
kttc report results.json --format html -o report.html
```

---

## kttc glossary

مدیریت واژه‌نامه‌های اصطلاحات با پشتیبانی از ذخیره‌سازی محلی پروژه و عمومی کاربر.

### مکان‌های ذخیره‌سازی

KTTC از ذخیره‌سازی دو سطحی واژه‌نامه پشتیبانی می‌کند:

- **واژه‌نامه‌های پروژه** (پیش‌فرض): `./glossaries/` - در پروژه فعلی ذخیره می‌شود، می‌تواند کنترل نسخه شود
- **واژه‌نامه‌های کاربر** (با پرچم `--user`): `~/.kttc/glossaries/` - واژه‌نامه‌های عمومی در دسترس در تمام پروژه‌ها

**اولویت جستجو**: ابتدا واژه‌نامه‌های پروژه، سپس واژه‌نامه‌های کاربر بررسی می‌شوند.

### دستورات فرعی

#### list

لیست تمام واژه‌نامه‌های موجود از هر دو مکان:

```bash
kttc glossary list
```

نمایش: نام، مکان (پروژه/کاربر)، تعداد اصطلاحات، و مسیر فایل.

#### show

نمایش محتویات واژه‌نامه:

```bash
kttc glossary show NAME [OPTIONS]
```

**گزینه‌ها:**
- `--lang-pair SRC-TGT` - فیلتر بر اساس جفت زبان (مثل `en-ru`)
- `--limit N` - محدود کردن تعداد ورودی‌های نمایش داده شده

#### create

ایجاد واژه‌نامه جدید از فایل CSV یا JSON:

```bash
kttc glossary create NAME --from-csv FILE
# یا
kttc glossary create NAME --from-json FILE
```

**گزینه‌ها:**
- `--from-csv PATH` - ایجاد از فایل CSV (لازم اگر از `--from-json` استفاده نشود)
- `--from-json PATH` - ایجاد از فایل JSON (لازم اگر از `--from-csv` استفاده نشود)
- `--user` - ذخیره در دایرکتوری کاربر (`~/.kttc/glossaries/`) به جای دایرکتوری پروژه

**فرمت CSV** (ستون‌های لازم):

```csv
source,target,source_lang,target_lang,context,notes
API,API,en,es,Keep as-is,Technical term
database,base de datos,en,es,,
```

**فرمت JSON:**

```json
{
  "metadata": {
    "name": "technical",
    "description": "Technical terminology",
    "version": "1.0.0"
  },
  "entries": [
    {
      "source": "API",
      "target": "API",
      "source_lang": "en",
      "target_lang": "es",
      "context": "Keep as-is",
      "notes": "Technical term"
    }
  ]
}
```

#### merge

ادغام چندین واژه‌نامه در یک واژه‌نامه:

```bash
kttc glossary merge GLOSSARY1 GLOSSARY2 [...] --output NAME [OPTIONS]
```

**گزینه‌ها:**
- `--output NAME` - نام واژه‌نامه خروجی (لازم)
- `--user` - ذخیره واژه‌نامه ادغام شده در دایرکتوری کاربر

#### export

صادرات واژه‌نامه به CSV یا JSON:

```bash
kttc glossary export NAME [OPTIONS]
```

**گزینه‌ها:**
- `--format csv|json` - فرمت صادرات (پیش‌فرض: csv)
- `--output PATH` - مسیر فایل خروجی (پیش‌فرض: `{name}.{format}`)

#### validate

اعتبارسنجی فرمت فایل واژه‌نامه:

```bash
kttc glossary validate FILE
```

بررسی:
- فیلدهای لازم (source، target، source_lang، target_lang)
- ورودی‌های تکراری
- مقادیر خالی
- کدهای زبان معتبر

### نمونه‌ها

**لیست تمام واژه‌نامه‌ها (پروژه + کاربر):**

```bash
kttc glossary list
```

خروجی:
```
📚 Project Glossaries (./glossaries/):
  • base (120 terms) - ./glossaries/base.json
  • technical (45 terms) - ./glossaries/technical.json

📚 User Glossaries (~/.kttc/glossaries/):
  • personal (30 terms) - ~/.kttc/glossaries/personal.json
```

**ایجاد واژه‌نامه پروژه از CSV:**

```bash
kttc glossary create medical --from-csv medical-terms.csv
```

در `./glossaries/medical.json` ذخیره می‌شود (می‌توان به git ارسال کرد).

**ایجاد واژه‌نامه عمومی کاربر:**

```bash
kttc glossary create personal --from-csv my-terms.csv --user
```

در `~/.kttc/glossaries/personal.json` ذخیره می‌شود (در تمام پروژه‌ها در دسترس).

**نمایش واژه‌نامه با فیلتر:**

```bash
kttc glossary show base --lang-pair en-ru --limit 10
```

**ادغام چندین واژه‌نامه:**

```bash
kttc glossary merge base technical medical --output combined
```

`./glossaries/combined.json` را با تمام اصطلاحات از سه واژه‌نامه ایجاد می‌کند.

**ادغام به دایرکتوری کاربر:**

```bash
kttc glossary merge base technical --output my-combined --user
```

`~/.kttc/glossaries/my-combined.json` را ایجاد می‌کند.

**صادرات به CSV:**

```bash
kttc glossary export technical --format csv --output technical-export.csv
```

**اعتبارسنجی فایل واژه‌نامه:**

```bash
kttc glossary validate my-glossary.csv
```

خروجی:
```
✓ All required columns present
✓ No duplicate entries found
✓ All language codes valid
✓ No empty values
✅ Glossary is valid
```

### استفاده از واژه‌نامه‌ها در بررسی ترجمه

ارجاع به واژه‌نامه‌ها با نام در `kttc check`:

```bash
# تشخیص خودکار واژه‌نامه 'base' (جستجو در پروژه، سپس کاربر)
kttc check source.txt trans.txt --source-lang en --target-lang ru --glossary auto

# استفاده از واژه‌نامه‌های خاص (جدا شده با کاما)
kttc check source.txt trans.txt --source-lang en --target-lang ru --glossary base,technical,medical

# غیرفعال کردن واژه‌نامه‌ها
kttc check source.txt trans.txt --source-lang en --target-lang ru --glossary none
```

**ترتیب جستجو**: KTTC ابتدا در دایرکتوری پروژه، سپس در دایرکتوری کاربر به دنبال واژه‌نامه‌ها می‌گردد.

## kttc terminology

دسترسی به واژه‌نامه‌های مرجع زبان‌شناختی و اعتبارسنج‌ها برای ارزیابی کیفیت ترجمه.

### دستورات فرعی

#### list

لیست تمام واژه‌نامه‌های مرجع زبان‌شناختی موجود.

```bash
kttc terminology list
```

**گزینه‌ها:**
- `--lang CODE`، `-l CODE` - فیلتر بر اساس کد زبان

**نمونه‌ها:**

```bash
kttc terminology list
kttc terminology list --lang ru
kttc terminology list --lang zh
```

#### show

نمایش محتویات یک واژه‌نامه مرجع زبان‌شناختی.

```bash
kttc terminology show LANGUAGE CATEGORY [OPTIONS]
```

**آرگومان‌ها:**
- `LANGUAGE` - کد زبان (مثل en، ru، zh)
- `CATEGORY` - دسته واژه‌نامه (مثل mqm_core، russian_cases)

**گزینه‌ها:**
- `--limit N`، `-n N` - حداکثر تعداد ورودی‌های نمایش داده شده (پیش‌فرض: 50)
- `--format FORMAT`، `-f FORMAT` - فرمت خروجی: table یا json (پیش‌فرض: table)

**نمونه‌ها:**

```bash
kttc terminology show en mqm_core
kttc terminology show ru russian_cases
kttc terminology show zh chinese_classifiers --limit 20
kttc terminology show en mqm_core --format json
```

#### search

جستجو در تمام واژه‌نامه‌های اصطلاحات.

```bash
kttc terminology search QUERY [OPTIONS]
```

**آرگومان‌ها:**
- `QUERY` - عبارت جستجو

**گزینه‌ها:**
- `--lang CODE`، `-l CODE` - فیلتر بر اساس کد زبان
- `--case-sensitive`، `-c` - جستجوی حساس به بزرگی و کوچکی حروف

**نمونه‌ها:**

```bash
kttc terminology search "mistranslation"
kttc terminology search "genitive" --lang ru
kttc terminology search "classifier" --lang zh
```

#### validate-error

اعتبارسنجی یک نوع خطای MQM بر اساس واژه‌نامه اصطلاحات.

```bash
kttc terminology validate-error ERROR_TYPE [OPTIONS]
```

**آرگومان‌ها:**
- `ERROR_TYPE` - نوع خطای MQM برای اعتبارسنجی

**گزینه‌ها:**
- `--lang CODE`، `-l CODE` - کد زبان (پیش‌فرض: en)

**نمونه‌ها:**

```bash
kttc terminology validate-error mistranslation
kttc terminology validate-error grammar --lang ru
kttc terminology validate-error untranslated --lang en
```

#### validators

لیست اعتبارسنج‌های موجود مخصوص زبان‌های خاص.

```bash
kttc terminology validators
```

**نمونه‌ها:**

```bash
kttc terminology validators
```

---
---

## گزینه‌های عمومی

برای تمام دستورات در دسترس:

- `--ui-lang CODE`، `-L CODE` - زبان رابط کاربری CLI (en، ru، zh، hi، fa) یا 'auto' برای تشخیص خودکار زبان سیستم
- `--version`, `-v` - نمایش نسخه و خروج
- `--help` - نمایش پیام راهنما

---

## کدهای خروج

- `0` - موفقیت (تمام ترجمه‌ها آستانه کیفیت را گذراندند)
- `1` - شکست (یک یا چند ترجمه در آستانه کیفیت شکست خوردند)
- `130` - قطع شده توسط کاربر (Ctrl+C)

---

## متغیرهای محیطی

- `KTTC_OPENAI_API_KEY` - کلید API OpenAI
- `KTTC_ANTHROPIC_API_KEY` - کلید API Anthropic
- `KTTC_GIGACHAT_CLIENT_ID` - شناسه کلاینت GigaChat
- `KTTC_GIGACHAT_CLIENT_SECRET` - رمز کلاینت GigaChat
- `KTTC_YANDEX_API_KEY` - کلید API Yandex GPT
- `KTTC_YANDEX_FOLDER_ID` - شناسه پوشه Yandex GPT

---

## همچنین ببینید

- [راهنمای استفاده از CLI](../guides/cli-usage.md) - نمونه‌های عملی
- [پیکربندی](../guides/configuration.md) - پیکربندی پیشرفته
- [مرجع API](api-reference.md) - Python API
