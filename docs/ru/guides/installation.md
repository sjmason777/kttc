# Руководство по установке

Это руководство охватывает все методы установки и дополнительные зависимости.

## Требования

- **Python:** 3.11 или выше
- **Операционные системы:** Linux, macOS, Windows
- **API-ключ:** OpenAI, Anthropic, GigaChat, или Yandex

## Базовая установка

### Использование pip (Рекомендуется)

```bash
pip install kttc
```

Это установит основные зависимости (примерно 50 МБ):
- Интерфейс командной строки
- Многоагентная система обеспечения качества
- Русский NLP (библиотеки MAWO)
- Базовая многоязычная поддержка (spaCy, jieba)

### Использование pipx (Изолированная среда)

```bash
pipx install kttc
```

Преимущества:
- Изолировано от системного Python
- Автоматическая настройка PATH
- Простое обновление

### Из исходного кода (Разработка)

```bash
git clone https://github.com/kttc-ai/kttc.git
cd kttc
pip install -e ".[dev]"
```

## Дополнительные зависимости

### Метрики (Семантическое сходство)

Добавляет метрики сходства на уровне предложений:

```bash
pip install kttc[metrics]
```

Включает:
- sentence-transformers
- Оценка семантического сходства

### Поддержка английского языка

Добавляет расширенную проверку английской грамматики с LanguageTool:

```bash
pip install kttc[english]
```

**Требования:**
- Java 17.0 или выше
- Примерно 200 МБ дискового пространства

Функции:
- Более 5000 грамматических правил
- Согласование подлежащего и сказуемого
- Проверка артиклей (a/an/the)
- Проверка предлогов

**Установка Java (при необходимости):**

```bash
# macOS
brew install openjdk@17

# Ubuntu/Debian
sudo apt install openjdk-17-jre

# Windows
# Скачайте с https://adoptium.net/
```

**Скачивание модели spaCy:**

```bash
python3 -m spacy download en_core_web_md
```

### Поддержка китайского языка

Добавляет расширенный китайский NLP с HanLP:

```bash
pip install kttc[chinese]
```

Функции:
- Проверка счетных слов (量词)
- Проверка видовых частиц (了/过)
- Высокоточное POS-тегирование (~95%)
- Примерно 300 МБ загрузка модели при первом использовании

### Все языковые помощники

Установка всех языковых расширений:

```bash
pip install kttc[all-languages]
```

Эквивалентно:

```bash
pip install kttc[english,chinese]
```

### Полная установка (Разработка + Все функции)

```bash
pip install kttc[full,dev]
```

## Проверка установки

Проверьте версию KTTC:

```bash
kttc --version
```

Тест с демонстрационным режимом (API-ключ не нужен):

```bash
echo "Hello" > source.txt
echo "Hola" > trans.txt
kttc check source.txt trans.txt --source-lang en --target-lang es --demo
```

## Настройка API-ключа

### OpenAI

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

### Anthropic

```bash
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### GigaChat

```bash
export KTTC_GIGACHAT_CLIENT_ID="your-client-id"
export KTTC_GIGACHAT_CLIENT_SECRET="your-client-secret"
```

### Yandex GPT

```bash
export KTTC_YANDEX_API_KEY="your-api-key"
export KTTC_YANDEX_FOLDER_ID="your-folder-id"
```

### Постоянная конфигурация

Добавьте в профиль оболочки (`~/.bashrc`, `~/.zshrc`):

```bash
# KTTC API Keys
export KTTC_OPENAI_API_KEY="sk-..."
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

Затем перезагрузите:

```bash
source ~/.bashrc  # или ~/.zshrc
```

## Обновление

### Обновление до последней версии

```bash
pip install --upgrade kttc
```

### Обновление с дополнительными зависимостями

```bash
pip install --upgrade kttc[all-languages,metrics]
```

## Удаление

```bash
pip uninstall kttc
```

Удалите загруженные модели:

```bash
# Модели spaCy
python3 -m spacy uninstall en_core_web_md

# Модели HanLP (если установлены)
rm -rf ~/.hanlp
```

## Устранение неполадок

### Проблемы с версией Python

**Ошибка:** `TypeError: unsupported operand type(s) for |`

**Решение:** Используйте Python 3.11+:

```bash
python3.11 -m pip install kttc
```

### Java не найдена (LanguageTool)

**Ошибка:** `java.lang.RuntimeException: Could not find java`

**Решение:** Установите Java 17+:

```bash
# Проверьте версию Java
java -version

# Должно показать: openjdk version "17.0.x" или выше
```

### Отказано в доступе

**Ошибка:** `ERROR: Could not install packages`

**Решение:** Используйте пользовательскую установку:

```bash
pip install --user kttc
```

Или используйте виртуальную среду:

```bash
python3 -m venv venv
source venv/bin/activate
pip install kttc
```

## Следующие шаги

- [Конфигурация](configuration.md) - Настройка параметров
- [Использование CLI](cli-usage.md) - Изучите команды
- [Быстрый старт](../tutorials/quickstart.md) - Запустите первую проверку