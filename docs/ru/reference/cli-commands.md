# Справочник команд CLI

Полный справочник всех команд командной строки KTTC.

## kttc check

Умный инспектор качества переводов с автоопределением.

### Синтаксис

```bash
kttc check SOURCE [TRANSLATIONS...] [OPTIONS]
```

### Режимы автоопределения

`kttc check` автоматически определяет, что вы хотите сделать:

| Входные данные | Определённый режим | Поведение |
|-------|--------------|----------|
| `source.txt translation.txt` | Одиночная проверка | Оценка качества |
| `source.txt trans1.txt trans2.txt` | Сравнение | Автоматическое сравнение |
| `translations.csv` | Пакетный (файл) | Обработка CSV/JSON |
| `source_dir/ trans_dir/` | Пакетный (каталог) | Обработка каталогов |

### Параметры

#### Обязательные (для режимов одиночной проверки/сравнения)

- `--source-lang CODE` - Код исходного языка (например, `en`)
- `--target-lang CODE` - Код целевого языка (например, `ru`)

#### Умные функции (включены по умолчанию)

- `--smart-routing` / `--no-smart-routing` - Выбор модели на основе сложности (по умолчанию: включено)
- `--glossary TEXT` - Используемые глоссарии: `auto` (по умолчанию), `none`, или названия через запятую
- `--output PATH` - Автоопределение формата по расширению (`.json`, `.md`, `.html`)

#### Контроль качества

- `--threshold FLOAT` - Минимальный балл MQM (по умолчанию: 95.0)
- `--auto-correct` - Автоматически исправлять обнаруженные ошибки
- `--correction-level light|full` - Уровень коррекции (по умолчанию: `light`)

#### Выбор модели

- `--provider openai|anthropic|gigachat|yandex` - Провайдер LLM
- `--auto-select-model` - Использовать оптимальную модель для языковой пары
- `--show-routing-info` - Отображать анализ сложности

#### Вывод и подробность

- `--format text|json|markdown|html` - Формат вывода (переопределяет автоопределение)
- `--verbose` - Показать подробный вывод
- `--demo` - Демо-режим (без API-вызовов, имитированные ответы)

### Примеры

**Проверка одного файла:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es
```

**Сравнение нескольких переводов (автоопределение):**

```bash
kttc check source.txt trans1.txt trans2.txt trans3.txt \
  --source-lang en \
  --target-lang ru
```

**Пакетная обработка CSV (автоопределение, языки из файла):**

```bash
kttc check translations.csv
```

**Пакетная обработка каталогов:**

```bash
kttc check source_dir/ translation_dir/ \
  --source-lang en \
  --target-lang ru
```

**Автокоррекция:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --auto-correct \
  --correction-level full
```

**HTML-отчёт (автоопределение по расширению):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --output report.html
```

**Отключить умные функции:**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --no-smart-routing \
  --glossary none
```

**Демо-режим (без API-вызовов):**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es \
  --demo
```

---

## kttc batch

Пакетная обработка нескольких переводов.

### Синтаксис

**Файловый режим:**

```bash
kttc batch --file FILE [OPTIONS]
```

**Режим каталогов:**

```bash
kttc batch --source-dir DIR --translation-dir DIR \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### Параметры

#### Выбор режима (взаимоисключающие)

- `--file PATH` - Пакетный файл (CSV, JSON или JSONL)
- `--source-dir PATH` + `--translation-dir PATH` - Режим каталогов

#### Обязательные (только для режима каталогов)

- `--source-lang CODE` - Код исходного языка
- `--target-lang CODE` - Код целевого языка

#### Общие параметры

- `--threshold FLOAT` - Минимальный балл MQM (по умолчанию: 95.0)
- `--output PATH` - Путь к выходному отчёту (по умолчанию: `report.json`)
- `--parallel INT` - Количество параллельных воркеров (по умолчанию: 4)
- `--glossary TEXT` - Используемые глоссарии
- `--smart-routing` - Включить маршрутизацию на основе сложности
- `--show-progress` / `--no-progress` - Показывать прогресс-бар (по умолчанию: показывать)
- `--verbose` - Подробный вывод
- `--demo` - Демо-режим

#### Только для файлового режима

- `--batch-size INT` - Размер пакета для группировки

### Поддерживаемые форматы файлов

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

### Примеры

**Обработка CSV-файла:**

```bash
kttc batch --file translations.csv
```

**Обработка JSON с прогрессом:**

```bash
kttc batch --file translations.json \
  --show-progress \
  --output results.json
```

**Режим каталогов:**

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

Сравнение нескольких переводов бок о бок.

### Синтаксис

```bash
kttc compare --source FILE \
  --translation FILE --translation FILE [...] \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### Параметры

- `--source PATH` - Файл с исходным текстом (обязательно)
- `--translation PATH` - Файл с переводом (можно указать несколько раз, обязательно)
- `--source-lang CODE` - Код исходного языка (обязательно)
- `--target-lang CODE` - Код целевого языка (обязательно)
- `--threshold FLOAT` - Порог качества (по умолчанию: 95.0)
- `--provider TEXT` - Провайдер LLM
- `--verbose` - Показать подробное сравнение

### Примеры

**Сравнить 3 перевода:**

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

Перевод текста с автоматической проверкой качества и доработкой.

### Синтаксис

```bash
kttc translate --text TEXT \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### Параметры

- `--text TEXT` - Текст для перевода (или `@file.txt` для ввода из файла, обязательно)
- `--source-lang CODE` - Код исходного языка (обязательно)
- `--target-lang CODE` - Код целевого языка (обязательно)
- `--threshold FLOAT` - Порог качества для доработки (по умолчанию: 95.0)
- `--max-iterations INT` - Максимальное количество итераций доработки (по умолчанию: 3)
- `--output PATH` - Путь к выходному файлу
- `--provider TEXT` - Провайдер LLM
- `--verbose` - Подробный вывод

### Примеры

**Перевод встроенного текста:**

```bash
kttc translate --text "Hello, world!" \
  --source-lang en \
  --target-lang es
```

**Перевод из файла:**

```bash
kttc translate --text @document.txt \
  --source-lang en \
  --target-lang ru \
  --output translated.txt
```

**С порогом качества:**

```bash
kttc translate --text "Complex technical text" \
  --source-lang en \
  --target-lang zh \
  --threshold 98 \
  --max-iterations 5
```

---

## kttc benchmark

Бенчмарк нескольких провайдеров LLM.

### Синтаксис

```bash
kttc benchmark --source FILE \
  --source-lang CODE --target-lang CODE \
  --providers LIST [OPTIONS]
```

### Параметры

- `--source PATH` - Файл с исходным текстом (обязательно)
- `--source-lang CODE` - Код исходного языка (обязательно)
- `--target-lang CODE` - Код целевого языка (обязательно)
- `--providers TEXT` - Список провайдеров через запятую (по умолчанию: `gigachat,openai,anthropic`)
- `--threshold FLOAT` - Порог качества (по умолчанию: 95.0)
- `--output PATH` - Путь к выходному файлу (JSON)
- `--verbose` - Подробный вывод

### Примеры

**Бенчмарк всех провайдеров:**

```bash
kttc benchmark \
  --source text.txt \
  --source-lang en \
  --target-lang ru \
  --providers gigachat,openai,anthropic
```

---

## kttc report

Генерация форматированных отчётов из результатов QA.

### Синтаксис

```bash
kttc report INPUT_FILE [OPTIONS]
```

### Параметры

- `--format markdown|html` - Формат вывода (по умолчанию: markdown)
- `--output PATH` - Путь к выходному файлу (автоматически генерируется, если не указан)

### Примеры

**Генерация Markdown-отчёта:**

```bash
kttc report results.json --format markdown -o report.md
```

**Генерация HTML-отчёта:**

```bash
kttc report results.json --format html -o report.html
```

---

## kttc glossary

Управление терминологическими глоссариями.

### Подкоманды

#### list

Список доступных глоссариев:

```bash
kttc glossary list
```

#### show

Показать детали глоссария:

```bash
kttc glossary show NAME
```

#### add

Добавить запись в глоссарий:

```bash
kttc glossary add NAME \
  --source TEXT \
  --target TEXT \
  --lang-pair SRC-TGT
```

#### import

Импорт глоссария из файла:

```bash
kttc glossary import NAME \
  --file PATH \
  --format csv|json|tbx
```

### Примеры

**Список глоссариев:**

```bash
kttc glossary list
```

**Показать базовый глоссарий:**

```bash
kttc glossary show base
```

**Добавить термин:**

```bash
kttc glossary add medical \
  --source "myocardial infarction" \
  --target "инфаркт миокарда" \
  --lang-pair en-ru
```

**Импорт из CSV:**

```bash
kttc glossary import technical \
  --file terms.csv \
  --format csv
```

---

## Глобальные параметры

Доступны для всех команд:

- `--version`, `-v` - Показать версию и выйти
- `--help` - Показать справочное сообщение

---

## Коды выхода

- `0` - Успех (все переводы прошли порог качества)
- `1` - Неудача (один или несколько переводов не прошли порог качества)
- `130` - Прервано пользователем (Ctrl+C)

---

## Переменные окружения

- `KTTC_OPENAI_API_KEY` - API-ключ OpenAI
- `KTTC_ANTHROPIC_API_KEY` - API-ключ Anthropic
- `KTTC_GIGACHAT_CLIENT_ID` - ID клиента GigaChat
- `KTTC_GIGACHAT_CLIENT_SECRET` - Секрет клиента GigaChat
- `KTTC_YANDEX_API_KEY` - API-ключ Yandex GPT
- `KTTC_YANDEX_FOLDER_ID` - ID папки Yandex GPT

---

## См. также

- [Руководство по использованию CLI](../guides/cli-usage.md) - Практические примеры
- [Конфигурация](../guides/configuration.md) - Расширенная конфигурация
- [Справочник API](api-reference.md) - Python API