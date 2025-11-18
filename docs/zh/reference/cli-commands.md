# CLI 命令参考

所有 KTTC 命令行命令的完整参考。

## kttc check

具有自动检测功能的智能翻译质量检查器。

### 语法

```bash
kttc check SOURCE [TRANSLATIONS...] [OPTIONS]
```

### 自动检测模式

`kttc check` 自动检测您想要执行的操作：

| 输入 | 检测模式 | 行为 |
|-------|--------------|----------|
| `source.txt translation.txt` | 单文件检查 | 质量评估 |
| `source.txt trans1.txt trans2.txt` | 比较 | 自动比较 |
| `translations.csv` | 批处理（文件） | 处理 CSV/JSON |
| `source_dir/ trans_dir/` | 批处理（目录） | 处理目录 |

### 选项

#### 必需（适用于单文件/比较模式）

- `--source-lang CODE` - 源语言代码（例如 `en`）
- `--target-lang CODE` - 目标语言代码（例如 `ru`）

####智能功能（默认启用）

- `--smart-routing` / `--no-smart-routing` - 基于复杂度的模型选择（默认：启用）
- `--glossary TEXT` - 要使用的术语表：`auto`（默认）、`none` 或逗号分隔的名称
- `--output PATH` - 从扩展名自动检测格式（`.json`、`.md`、`.html`）

#### 质量控制

- `--threshold FLOAT` - 最低 MQM 分数（默认：95.0）
- `--auto-correct` - 自动修复检测到的错误
- `--correction-level light|full` - 修正级别（默认：`light`）

#### 模型选择

- `--provider openai|anthropic|gigachat|yandex` - LLM 提供商
- `--auto-select-model` - 为语言对使用最优模型
- `--show-routing-info` - 显示复杂度分析

#### 输出和详细程度

- `--format text|json|markdown|html` - 输出格式（覆盖自动检测）
- `--verbose` - 显示详细输出
- `--demo` - 演示模式（不进行 API 调用，模拟响应）

### 示例

**单文件检查：**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es
```

**比较多个翻译（自动检测）：**

```bash
kttc check source.txt trans1.txt trans2.txt trans3.txt \
  --source-lang en \
  --target-lang ru
```

**批处理 CSV（自动检测，从文件获取语言）：**

```bash
kttc check translations.csv
```

**批处理目录：**

```bash
kttc check source_dir/ translation_dir/ \
  --source-lang en \
  --target-lang ru
```

**自动修正：**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --auto-correct \
  --correction-level full
```

**HTML 报告（从扩展名自动检测）：**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --output report.html
```

**禁用智能功能：**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang ru \
  --no-smart-routing \
  --glossary none
```

**演示模式（不进行 API 调用）：**

```bash
kttc check source.txt translation.txt \
  --source-lang en \
  --target-lang es \
  --demo
```

---

## kttc batch

批量处理多个翻译。

### 语法

**文件模式：**

```bash
kttc batch --file FILE [OPTIONS]
```

**目录模式：**

```bash
kttc batch --source-dir DIR --translation-dir DIR \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### 选项

#### 模式选择（互斥）

- `--file PATH` - 批处理文件（CSV、JSON 或 JSONL）
- `--source-dir PATH` + `--translation-dir PATH` - 目录模式

#### 必需（仅目录模式）

- `--source-lang CODE` - 源语言代码
- `--target-lang CODE` - 目标语言代码

#### 通用选项

- `--threshold FLOAT` - 最低 MQM 分数（默认：95.0）
- `--output PATH` - 输出报告路径（默认：`report.json`）
- `--parallel INT` - 并行工作进程数（默认：4）
- `--glossary TEXT` - 要使用的术语表
- `--smart-routing` - 启用基于复杂度的路由
- `--show-progress` / `--no-progress` - 显示进度条（默认：显示）
- `--verbose` - 详细输出
- `--demo` - 演示模式

#### 仅文件模式

- `--batch-size INT` - 分组的批次大小

### 支持的文件格式

**CSV：**

```csv
source,translation,source_lang,target_lang,domain
"Hello world","Hola mundo","en","es","general"
```

**JSON：**

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

**JSONL：**

```jsonl
{"source": "Hello world", "translation": "Hola mundo", "source_lang": "en", "target_lang": "es"}
{"source": "Good morning", "translation": "Buenos días", "source_lang": "en", "target_lang": "es"}
```

### 示例

**处理 CSV 文件：**

```bash
kttc batch --file translations.csv
```

**处理 JSON 并显示进度：**

```bash
kttc batch --file translations.json \
  --show-progress \
  --output results.json
```

**目录模式：**

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

并排比较多个翻译。

### 语法

```bash
kttc compare --source FILE \
  --translation FILE --translation FILE [...] \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### 选项

- `--source PATH` - 源文本文件（必需）
- `--translation PATH` - 翻译文件（可多次指定，必需）
- `--source-lang CODE` - 源语言代码（必需）
- `--target-lang CODE` - 目标语言代码（必需）
- `--threshold FLOAT` - 质量阈值（默认：95.0）
- `--provider TEXT` - LLM 提供商
- `--verbose` - 显示详细比较

### 示例

**比较 3 个翻译：**

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

翻译文本，自动质量检查和改进。

### 语法

```bash
kttc translate --text TEXT \
  --source-lang CODE --target-lang CODE [OPTIONS]
```

### 选项

- `--text TEXT` - 要翻译的文本（或 `@file.txt` 用于文件输入，必需）
- `--source-lang CODE` - 源语言代码（必需）
- `--target-lang CODE` - 目标语言代码（必需）
- `--threshold FLOAT` - 改进的质量阈值（默认：95.0）
- `--max-iterations INT` - 最大改进迭代次数（默认：3）
- `--output PATH` - 输出文件路径
- `--provider TEXT` - LLM 提供商
- `--verbose` - 详细输出

### 示例

**翻译内联文本：**

```bash
kttc translate --text "Hello, world!" \
  --source-lang en \
  --target-lang es
```

**从文件翻译：**

```bash
kttc translate --text @document.txt \
  --source-lang en \
  --target-lang ru \
  --output translated.txt
```

**设置质量阈值：**

```bash
kttc translate --text "Complex technical text" \
  --source-lang en \
  --target-lang zh \
  --threshold 98 \
  --max-iterations 5
```

---

## kttc benchmark

基准测试多个 LLM 提供商。

### 语法

```bash
kttc benchmark --source FILE \
  --source-lang CODE --target-lang CODE \
  --providers LIST [OPTIONS]
```

### 选项

- `--source PATH` - 源文本文件（必需）
- `--source-lang CODE` - 源语言代码（必需）
- `--target-lang CODE` - 目标语言代码（必需）
- `--providers TEXT` - 逗号分隔的提供商列表（默认：`gigachat,openai,anthropic`）
- `--threshold FLOAT` - 质量阈值（默认：95.0）
- `--output PATH` - 输出文件路径（JSON）
- `--verbose` - 详细输出

### 示例

**测试所有提供商：**

```bash
kttc benchmark \
  --source text.txt \
  --source-lang en \
  --target-lang ru \
  --providers gigachat,openai,anthropic
```

---

## kttc report

从质量评估结果生成格式化报告。

### 语法

```bash
kttc report INPUT_FILE [OPTIONS]
```

### 选项

- `--format markdown|html` - 输出格式（默认：markdown）
- `--output PATH` - 输出文件路径（如未指定则自动生成）

### 示例

**生成 Markdown 报告：**

```bash
kttc report results.json --format markdown -o report.md
```

**生成 HTML 报告：**

```bash
kttc report results.json --format html -o report.html
```

---

## kttc glossary

管理术语表，支持项目本地和用户全局存储。

### 存储位置

KTTC 支持两层术语表存储：

- **项目术语表**（默认）：`./glossaries/` - 存储在当前项目中，可以进行版本控制
- **用户术语表**（使用 `--user` 标志）：`~/.kttc/glossaries/` - 全局术语表，在所有项目中可用

**搜索优先级**：首先检查项目术语表，然后检查用户术语表。

### 子命令

#### list

列出两个位置的所有可用术语表：

```bash
kttc glossary list
```

显示：名称、位置（项目/用户）、术语数量和文件路径。

#### show

显示术语表内容：

```bash
kttc glossary show NAME [OPTIONS]
```

**选项：**
- `--lang-pair SRC-TGT` - 按语言对过滤（例如 `en-zh`）
- `--limit N` - 限制显示的条目数量

#### create

从 CSV 或 JSON 文件创建新术语表：

```bash
kttc glossary create NAME --from-csv FILE
# 或
kttc glossary create NAME --from-json FILE
```

**选项：**
- `--from-csv PATH` - 从 CSV 文件创建（如果不使用 `--from-json` 则必需）
- `--from-json PATH` - 从 JSON 文件创建（如果不使用 `--from-csv` 则必需）
- `--user` - 保存到用户目录（`~/.kttc/glossaries/`）而不是项目目录

**CSV 格式**（必需列）：

```csv
source,target,source_lang,target_lang,context,notes
API,API,en,zh,保持不变,技术术语
database,数据库,en,zh,,
```

**JSON 格式：**

```json
{
  "metadata": {
    "name": "technical",
    "description": "技术术语",
    "version": "1.0.0"
  },
  "entries": [
    {
      "source": "API",
      "target": "API",
      "source_lang": "en",
      "target_lang": "zh",
      "context": "保持不变",
      "notes": "技术术语"
    }
  ]
}
```

#### merge

将多个术语表合并为一个：

```bash
kttc glossary merge GLOSSARY1 GLOSSARY2 [...] --output NAME [OPTIONS]
```

**选项：**
- `--output NAME` - 输出术语表名称（必需）
- `--user` - 将合并的术语表保存到用户目录

#### export

将术语表导出为 CSV 或 JSON：

```bash
kttc glossary export NAME [OPTIONS]
```

**选项：**
- `--format csv|json` - 导出格式（默认：csv）
- `--output PATH` - 输出文件路径（默认：`{name}.{format}`）

#### validate

验证术语表文件格式：

```bash
kttc glossary validate FILE
```

检查：
- 必需字段（source、target、source_lang、target_lang）
- 重复条目
- 空值
- 有效的语言代码

### 示例

**列出所有术语表（项目 + 用户）：**

```bash
kttc glossary list
```

输出：
```
📚 项目术语表 (./glossaries/):
  • base (120 个术语) - ./glossaries/base.json
  • technical (45 个术语) - ./glossaries/technical.json

📚 用户术语表 (~/.kttc/glossaries/):
  • personal (30 个术语) - ~/.kttc/glossaries/personal.json
```

**从 CSV 创建项目术语表：**

```bash
kttc glossary create medical --from-csv medical-terms.csv
```

保存到 `./glossaries/medical.json`（可以提交到 git）。

**创建全局用户术语表：**

```bash
kttc glossary create personal --from-csv my-terms.csv --user
```

保存到 `~/.kttc/glossaries/personal.json`（在所有项目中可用）。

**显示带过滤的术语表：**

```bash
kttc glossary show base --lang-pair en-zh --limit 10
```

**合并多个术语表：**

```bash
kttc glossary merge base technical medical --output combined
```

创建 `./glossaries/combined.json`，包含三个术语表的所有术语。

**合并到用户目录：**

```bash
kttc glossary merge base technical --output my-combined --user
```

创建 `~/.kttc/glossaries/my-combined.json`。

**导出为 CSV：**

```bash
kttc glossary export technical --format csv --output technical-export.csv
```

**验证术语表文件：**

```bash
kttc glossary validate my-glossary.csv
```

输出：
```
✓ 所有必需列都存在
✓ 未发现重复条目
✓ 所有语言代码有效
✓ 没有空值
✅ 术语表有效
```

### 在翻译检查中使用术语表

在 `kttc check` 中按名称引用术语表：

```bash
# 自动检测 'base' 术语表（搜索项目，然后用户）
kttc check source.txt trans.txt --source-lang en --target-lang zh --glossary auto

# 使用特定术语表（逗号分隔）
kttc check source.txt trans.txt --source-lang en --target-lang zh --glossary base,technical,medical

# 禁用术语表
kttc check source.txt trans.txt --source-lang en --target-lang zh --glossary none
```

**搜索顺序**：KTTC 首先在项目目录中搜索术语表，然后在用户目录中搜索。

---

## 全局选项

适用于所有命令：

- `--version`，`-v` - 显示版本并退出
- `--help` - 显示帮助信息

---

## 退出代码

- `0` - 成功（所有翻译均通过质量阈值）
- `1` - 失败（一个或多个翻译未通过质量阈值）
- `130` - 用户中断（Ctrl+C）

---

## 环境变量

- `KTTC_OPENAI_API_KEY` - OpenAI API 密钥
- `KTTC_ANTHROPIC_API_KEY` - Anthropic API 密钥
- `KTTC_GIGACHAT_CLIENT_ID` - GigaChat 客户端 ID
- `KTTC_GIGACHAT_CLIENT_SECRET` - GigaChat 客户端密钥
- `KTTC_YANDEX_API_KEY` - Yandex GPT API 密钥
- `KTTC_YANDEX_FOLDER_ID` - Yandex GPT 文件夹 ID

---

## 另请参阅

- [CLI 使用指南](../guides/cli-usage.md) - 实用示例
- [配置](../guides/configuration.md) - 高级配置
- [API 参考](api-reference.md) - Python API