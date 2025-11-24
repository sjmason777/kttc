# مدیریت واژه‌نامه‌ها

**از داده‌های مرجع زبان‌شناختی برای افزایش کیفیت بررسی ترجمه استفاده کنید.**

واژه‌نامه‌ها دانش زبان‌شناختی خاص هر زبان را ارائه می‌دهند که به عامل‌های KTTC کمک می‌کند ارزیابی‌های کیفیت دقیق‌تری انجام دهند. این شامل قواعد دستوری، تعاریف اصطلاحات و الگوهای خاص زبانی می‌شود.

## مرور کلی

KTTC واژه‌نامه‌های جامعی را برای موارد زیر یکپارچه کرده است:

- **تعاریف خطاهای MQM** - طبقه‌بندی استاندارد خطا از W3C Multidimensional Quality Metrics
- **دستور زبان روسی** - سیستم حالت‌ها، وجه‌های فعل، ذرات
- **طبقه‌بندی‌کننده‌های چینی** - کلمات شمارشی (量词) و الگوهای استفاده
- **دستور زبان هندی** - سیستم حالت‌ها (कारक)، حروف اضافه، اشکال غیرمستقیم
- **دستور زبان فارسی** - ساختار اضافه، افعال مرکب

این واژه‌نامه‌ها به طور خودکار توسط عامل‌های کیفیت در طول ارزیابی بارگذاری و استفاده می‌شوند.

## زبان‌های پشتیبانی شده

### تعاریف هسته MQM

**در دسترس به زبان‌های:** انگلیسی، روسی، چینی، هندی، فارسی

واژه‌نامه MQM (معیارهای کیفیت چندبعدی) تعاریف استاندارد شده خطایی را که توسط همه عامل‌ها استفاده می‌شود، ارائه می‌دهد.

**دسته‌بندی خطاها:**
- دقت: ترجمه نادرست، حذف، افزودن
- روانی: دستور، املا، نقطه‌گذاری
- سبک: ثبت، رسمیت، سازگاری
- اصطلاحات: خطاهای اصطلاحات خاص حوزه

**جدید در نسخه 1.2: نام‌های مستعار انگلیسی**

واژه‌نامه‌های MQM اکنون هم انواع خطاهای انگلیسی و هم زبان بومی را می‌پذیرند. این امر از خرابی‌های اعتبارسنجی جلوگیری می‌کند هنگامی که LLM انواع انگلیسی مانند `inconsistency`، `formatting` یا `untranslated` را برمی‌گرداند.

### واژه‌نامه‌های اصطلاحات IT (جدید در نسخه 1.2)

**در دسترس به زبان‌های:** انگلیسی، روسی، چینی، هندی، فارسی

اصطلاحات جامع IT و توسعه نرم‌افزار برای مستندات فنی:

| زبان | فایل | تعداد |
|------|------|-------|
| انگلیسی | `glossaries/en/it_terminology_en.json` | 200 |
| روسی | `glossaries/ru/it_terminology_ru.json` | 150 |
| چینی | `glossaries/zh/it_terminology_zh.json` | 120 |
| هندی | `glossaries/hi/it_terminology_hi.json` | 100 |
| فارسی | `glossaries/fa/it_terminology_fa.json` | 100 |

**دسته‌های پوشش داده شده:**
- CLI و shell (bash، zsh، stdin، stdout، pipe)
- کنترل نسخه (git، commit، branch، merge، PR)
- فرآیند توسعه (Agile، sprint، deploy، CI/CD)
- معماری (microservice، container، Kubernetes)
- زیرساخت ابری (AWS، Azure، GCP، IaaS)
- داده و ذخیره‌سازی (SQL، NoSQL، Redis، ORM)
- API و یکپارچه‌سازی (REST، GraphQL، OAuth، JWT)
- تست (unit test، mock، TDD، coverage)
- ML/AI (LLM، embedding، fine-tuning، RAG)

### واژه‌نامه زبان روسی

**فایل:** `glossaries/ru/grammar_reference.json`

**شامل:**
- **6 حالت دستوری** (فاعلی، مضاف‌الیهی، مفعولی، مفعولی مستقیم، ابزاری، حرف اضافه‌ای)
- **وجه‌های فعل** (قواعد استفاده از وجه کامل/ناقص)
- **ذرات** (же، ли، бы، не، ни)
- **نشانگرهای ثبت** (تمایز ты/вы)

**استفاده شده توسط عامل:** `RussianFluencyAgent`

### واژه‌نامه زبان چینی

**فایل:** `glossaries/zh/classifiers.json`

**شامل:**
- **طبقه‌بندی‌کننده‌های فردی** (个،只،条،张،本و غیره)
- **طبقه‌بندی‌کننده‌های جمعی** (群،堆،批و غیره)
- **طبقه‌بندی‌کننده‌های ظرف** (杯،碗،盒 و غیره)
- **طبقه‌بندی‌کننده‌های اندازه‌گیری** (米،公斤،升 و غیره)
- **طبقه‌بندی‌کننده‌های زمانی** (年،月،天 و غیره)
- **طبقه‌بندی‌کننده‌های فعلی** (次،遍،趟 و غیره)

**استفاده شده توسط عامل:** `ChineseFluencyAgent`

### واژه‌نامه زبان هندی

**فایل:** `glossaries/hi/grammar_reference.json`

**شامل:**
- **8 حالت دستوری** (कारक - سیستم kārak)
- **حروف اضافه** (को، से، में، पर و غیره)
- **اشکال غیرمستقیم** (قواعد ساخت)
- **توافق جنسیتی** (الگوهای مذکر/مؤنث)

**استفاده شده توسط عامل:** `HindiFluencyAgent`

### واژه‌نامه زبان فارسی

**فایل:** `glossaries/fa/grammar_reference.json`

**شامل:**
- **ساختار اضافه** (قواعد e-zāfe)
- **افعال مرکب** (افعال سبک: کردن، زدن، داشتن و غیره)
- **حروف اضافه** (در، به، از و غیره)
- **ثبت رسمی/غیررسمی** (تمایز شما/تو)

**استفاده شده توسط عامل:** `PersianFluencyAgent`

## نحوه کار واژه‌نامه‌ها

### یکپارچه‌سازی خودکار

واژه‌نامه‌ها هنگام راه‌اندازی عامل‌ها به طور خودکار بارگذاری می‌شوند:

```python
from kttc.agents import RussianFluencyAgent, ChineseFluencyAgent
from kttc.llm import OpenAIProvider

provider = OpenAIProvider(api_key="your-key")

# عامل روسی به طور خودکار grammar_reference.json را بارگذاری می‌کند
russian_agent = RussianFluencyAgent(provider)

# عامل چینی به طور خودکار classifiers.json را بارگذاری می‌کند
chinese_agent = ChineseFluencyAgent(provider)
```

نیاز به پیکربندی نیست - واژه‌نامه‌ها بر اساس نوع عامل بارگذاری می‌شوند.

### غنی‌سازی خطاها

وقتی عامل‌ها خطاها را تشخیص می‌دهند، به طور خودکار توضیحات خطا را با تعاریف واژه‌نامه غنی می‌کنند:

```python
# خطای تشخیص داده شده توسط عامل
error = {
    "category": "fluency",
    "subcategory": "grammar",
    "description": "عدم تطابق حالت"
}

# پس از غنی‌سازی واژه‌نامه
enriched_error = {
    "category": "fluency",
    "subcategory": "grammar",
    "description": "عدم تطابق حالت. در زبان روسی، صفت‌ها باید در حالت، تعداد و جنسیت با اسم‌ها مطابقت داشته باشند. حالت مضاف‌الیهی بعد از نفی‌ها، با اعداد و برای نشان دادن مالکیت استفاده می‌شود."
}
```

### وزن‌دهی امتیاز MQM

واژه‌نامه MQM وزن‌های دسته‌بندی را برای امتیازدهی ارائه می‌دهد:

```python
from kttc.core.mqm import MQMScorer

# بارگذاری وزن‌ها از واژه‌نامه
scorer = MQMScorer(use_glossary_weights=True)

# وزن‌های دسته‌بندی از glossary/en/mqm_core.json:
# accuracy: 1.5 (جریمه بالاتر برای خطاهای دقت)
# terminology: 1.2
# fluency: 1.0
# style: 0.8 (جریمه پایین‌تر برای مسائل سبک)
```

## دستورات CLI

### مشاهده محتوای واژه‌نامه

```bash
# لیست همه واژه‌نامه‌های موجود
kttc glossary list

# مشاهده واژه‌نامه خاص
kttc glossary view --language fa --name grammar_reference

# جستجو در واژه‌نامه
kttc glossary search --language fa --query "اضافه"

# مشاهده در قالب JSON
kttc glossary view --language en --name mqm_core --format json
```

### دستورات اصطلاحات

```bash
# اعتبارسنجی نوع خطا
kttc terminology validate --error-type "mistranslation" --language fa

# لیست دسته‌بندی‌های خطای MQM
kttc terminology list-categories

# دریافت تعریف خطا
kttc terminology define --error-type "grammar" --language fa
```

## قالب فایل واژه‌نامه

واژه‌نامه‌ها به عنوان فایل‌های JSON با ساختار زیر ذخیره می‌شوند:

### قالب MQM Core

```json
{
  "metadata": {
    "name": "MQM Core Error Taxonomy",
    "version": "1.0",
    "language": "fa",
    "description": "معیارهای کیفیت چندبعدی W3C"
  },
  "categories": {
    "accuracy": {
      "weight": 1.5,
      "subcategories": {
        "mistranslation": {
          "definition": "متن هدف به طور دقیق متن منبع را نشان نمی‌دهد",
          "examples": ["گربه → سگ", "خریدن → فروختن"],
          "severity_guidelines": "Major یا Critical"
        }
      }
    }
  }
}
```

### قالب خاص زبان

```json
{
  "metadata": {
    "name": "مرجع دستور زبان فارسی",
    "version": "1.0",
    "language": "fa"
  },
  "ezafe": {
    "rules": {
      "usage": "اتصال اسم به صفت یا اسم به اسم",
      "examples": ["کتاب خوب", "خانه پدر"]
    }
  },
  "compound_verbs": {
    "کردن": {
      "usage": "فعل سبک رایج",
      "examples": ["کار کردن", "صحبت کردن"]
    }
  }
}
```

## استفاده پیشرفته

### واژه‌نامه‌های سفارشی

می‌توانید واژه‌نامه‌های سفارشی را به دایرکتوری `glossaries/` اضافه کنید:

```bash
# ایجاد واژه‌نامه سفارشی
mkdir -p glossaries/fa
cat > glossaries/fa/medical_terms.json <<EOF
{
  "metadata": {
    "name": "اصطلاحات پزشکی",
    "version": "1.0",
    "language": "fa",
    "domain": "medical"
  },
  "terms": {
    "سکته قلبی": {
      "definition": "انفارکتوس میوکارد",
      "synonyms": ["حمله قلبی", "MI"],
      "avoid": ["ایست قلبی"]
    }
  }
}
EOF
```

### دسترسی برنامه‌نویسی

```python
from kttc.terminology import GlossaryManager

# راه‌اندازی مدیر
manager = GlossaryManager()

# بارگذاری واژه‌نامه
mqm_data = manager.load_glossary("fa", "mqm_core")

# دریافت تعریف خاص
definition = mqm_data["categories"]["accuracy"]["subcategories"]["mistranslation"]

# اعتبارسنجی نوع خطا
is_valid, info = manager.validate_mqm_error("mistranslation", "fa")
```

### غیرفعال کردن غنی‌سازی واژه‌نامه

```python
from kttc.agents.parser import ErrorParser

# تجزیه بدون غنی‌سازی
errors = ErrorParser.parse_errors(
    response=llm_response,
    enrich_with_glossary=False  # غیرفعال کردن غنی‌سازی
)

# یا استفاده از وزن‌های پیش‌فرض برای امتیازدهی MQM
from kttc.core.mqm import MQMScorer

scorer = MQMScorer(use_glossary_weights=False)  # استفاده از وزن‌های پیش‌فرض
```

## مزایا

### 1. دقت بهبود یافته

واژه‌نامه‌ها تعاریف زبان‌شناختی دقیقی ارائه می‌دهند که به عامل‌ها کمک می‌کنند:
- خطاهای خاص زبان را با دقت بیشتری شناسایی کنند
- الگوها و قواعد دستوری را درک کنند
- استفاده از اصطلاحات را اعتبارسنجی کنند

### 2. توضیحات بهتر خطا

توضیحات غنی‌شده خطا شامل موارد زیر است:
- توضیحات زبان‌شناختی
- قواعد دستوری
- نمونه‌های استفاده
- تصحیحات

### 3. امتیازدهی سازگار

واژه‌نامه MQM تضمین می‌کند:
- طبقه‌بندی استاندارد شده خطا
- وزن‌های سازگار دسته‌بندی
- امتیازهای کیفیت قابل تکرار
- همسویی با استانداردهای صنعتی (W3C MQM)

### 4. پشتیبانی چندزبانه

همان چارچوب واژه‌نامه برای موارد زیر کار می‌کند:
- 5 زبان (en، ru، zh، hi، fa)
- دسته‌بندی‌های متعدد خطا
- ویژگی‌های خاص زبان
- سازگاری‌های فرهنگی

## موقعیت فایل‌ها

```
glossaries/
├── en/
│   └── mqm_core.json          # تعاریف MQM انگلیسی
├── ru/
│   ├── mqm_core.json          # تعاریف MQM روسی
│   └── grammar_reference.json # قواعد دستور روسی
├── zh/
│   ├── mqm_core.json          # تعاریف MQM چینی
│   └── classifiers.json       # کلمات شمارشی چینی
├── hi/
│   ├── mqm_core.json          # تعاریف MQM هندی
│   └── grammar_reference.json # قواعد دستور هندی
└── fa/
    ├── mqm_core.json          # تعاریف MQM فارسی
    └── grammar_reference.json # قواعد دستور فارسی
```

## مستندات مرتبط

- **[دستورات CLI](../reference/cli-commands.md)** - دستورات واژه‌نامه و اصطلاحات
- **[معماری](../explanation/architecture.md)** - چگونگی یکپارچه‌سازی واژه‌نامه‌ها با عامل‌ها
- **[ویژگی‌های زبانی](language-features.md)** - قابلیت‌های خاص زبان

## عیب‌یابی

### واژه‌نامه بارگذاری نمی‌شود

**مشکل:** عامل از داده‌های واژه‌نامه استفاده نمی‌کند

**راه‌حل:**
```python
# بررسی وجود فایل واژه‌نامه
import os
glossary_path = "glossaries/fa/grammar_reference.json"
print(f"وجود دارد: {os.path.exists(glossary_path)}")

# فعال‌سازی گزارش‌گیری اشکال‌زدایی
import logging
logging.basicConfig(level=logging.DEBUG)
```

### واژه‌نامه سفارشی پیدا نشد

**مشکل:** واژه‌نامه سفارشی بارگذاری نمی‌شود

**راه‌حل:**
- اطمینان حاصل کنید فایل در دایرکتوری صحیح است: `glossaries/{language}/{name}.json`
- بررسی کنید سینتکس JSON معتبر است
- تأیید کنید بخش metadata موجود است

### غنی‌سازی خطا کار نمی‌کند

**مشکل:** خطاها با تعاریف غنی نمی‌شوند

**راه‌حل:**
```python
# صریحاً غنی‌سازی را فعال کنید
from kttc.agents.parser import ErrorParser

errors = ErrorParser.parse_errors(
    response=response,
    enrich_with_glossary=True,  # صریحاً فعال کنید
    language="fa"  # زبان را مشخص کنید
)
```

## عملکرد

**بارگذاری واژه‌نامه:**
- زمان بارگذاری: <100ms به ازای هر واژه‌نامه
- استفاده از حافظه: ~1-5MB به ازای هر واژه‌نامه
- پس از اولین بارگذاری کش می‌شود

**غنی‌سازی خطا:**
- زمان جستجو: <1ms به ازای هر خطا
- نیاز به فراخوانی API نیست
- نتایج قطعی

## بهبودهای آینده

ویژگی‌های برنامه‌ریزی شده واژه‌نامه:
- واژه‌نامه‌های قابل ویرایش توسط کاربر از طریق رابط وب
- واژه‌نامه‌های خاص حوزه (حقوقی، پزشکی، فنی)
- کنترل نسخه واژه‌نامه
- واژه‌نامه‌های مشارکتی جامعه
- ادغام واژه‌نامه و حل تعارض
