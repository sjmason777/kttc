# 术语表管理

**利用语言学参考数据提升翻译质量检查。**

术语表提供语言特定的语言学知识，帮助 KTTC 的代理进行更准确的质量评估。包括语法规则、术语定义和语言特定模式。

## 概述

KTTC 集成了以下全面的术语表：

- **MQM 错误定义** - W3C 多维质量指标标准错误分类
- **俄语语法** - 格系统、动词体、小品词
- **中文分类词** - 量词（量词）和使用模式
- **印地语语法** - 格系统（कारक）、后置词、斜格形式
- **波斯语语法** - Ezafe 结构（اضافه）、复合动词

这些术语表在评估期间由质量代理自动加载和使用。

## 支持的语言

### MQM 核心定义

**可用语言：** 英语、俄语、中文、印地语、波斯语

MQM（多维质量指标）术语表提供所有代理使用的标准化错误定义。

**错误类别：**
- 准确性：误译、遗漏、增译
- 流畅性：语法、拼写、标点
- 风格：语域、正式度、一致性
- 术语：领域特定术语错误

**v1.2 新功能：英语别名**

MQM 术语表现在同时接受英语和本地语言错误类型。这可以防止 LLM 返回英语类型（如 `inconsistency`、`formatting` 或 `untranslated`）时出现验证失败。

### IT 术语表（v1.2 新功能）

**可用语言：** 英语、俄语、中文、印地语、波斯语

技术文档的全面 IT 和软件开发术语：

| 语言 | 文件 | 术语数 |
|------|------|--------|
| 英语 | `glossaries/en/it_terminology_en.json` | 200 |
| 俄语 | `glossaries/ru/it_terminology_ru.json` | 150 |
| 中文 | `glossaries/zh/it_terminology_zh.json` | 120 |
| 印地语 | `glossaries/hi/it_terminology_hi.json` | 100 |
| 波斯语 | `glossaries/fa/it_terminology_fa.json` | 100 |

**涵盖的类别：**
- CLI 和 shell（bash、zsh、stdin、stdout、pipe）
- 版本控制（git、commit、branch、merge、PR）
- 开发流程（Agile、sprint、deploy、CI/CD）
- 架构（microservice、container、Kubernetes）
- 云基础设施（AWS、Azure、GCP、IaaS）
- 数据和存储（SQL、NoSQL、Redis、ORM）
- API 和集成（REST、GraphQL、OAuth、JWT）
- 测试（unit test、mock、TDD、coverage）
- ML/AI（LLM、embedding、fine-tuning、RAG）

### 俄语术语表

**文件：** `glossaries/ru/grammar_reference.json`

**包含：**
- **6 个语法格** （主格、属格、与格、宾格、工具格、前置格）
- **动词体** （完成体/未完成体使用规则）
- **小品词** （же、ли、бы、не、ни）
- **语域标记** （ты/вы 区分）

**使用代理：** `RussianFluencyAgent`

### 中文术语表

**文件：** `glossaries/zh/classifiers.json`

**包含：**
- **个体量词** （个、只、条、张、本等）
- **集合量词** （群、堆、批等）
- **容器量词** （杯、碗、盒等）
- **度量量词** （米、公斤、升等）
- **时间量词** （年、月、天等）
- **动量词** （次、遍、趟等）

**使用代理：** `ChineseFluencyAgent`

### 印地语术语表

**文件：** `glossaries/hi/grammar_reference.json`

**包含：**
- **8 个语法格** （कारक - kārak 系统）
- **后置词** （को、से、में、पर 等）
- **斜格形式** （构成规则）
- **性别一致** （阳性/阴性模式）

**使用代理：** `HindiFluencyAgent`

### 波斯语术语表

**文件：** `glossaries/fa/grammar_reference.json`

**包含：**
- **Ezafe 结构** （اضافه - e-zāfe 规则）
- **复合动词** （轻动词：کردن、زدن、داشتن 等）
- **介词** （در、به、از 等）
- **正式/非正式语域** （شما/تو 区分）

**使用代理：** `PersianFluencyAgent`

## 术语表的工作原理

### 自动集成

代理初始化时自动加载术语表：

```python
from kttc.agents import RussianFluencyAgent, ChineseFluencyAgent
from kttc.llm import OpenAIProvider

provider = OpenAIProvider(api_key="your-key")

# 俄语代理自动加载 grammar_reference.json
russian_agent = RussianFluencyAgent(provider)

# 中文代理自动加载 classifiers.json
chinese_agent = ChineseFluencyAgent(provider)
```

无需配置 - 术语表根据代理类型自动加载。

### 错误增强

当代理检测到错误时，会自动用术语表定义增强错误描述：

```python
# 代理检测到的错误
error = {
    "category": "fluency",
    "subcategory": "grammar",
    "description": "量词使用错误"
}

# 术语表增强后
enriched_error = {
    "category": "fluency",
    "subcategory": "grammar",
    "description": "量词使用错误。在中文中，量词必须与名词正确搭配。例如：一本书（正确），一个书（在大多数情况下不正确）。"
}
```

### MQM 评分权重

MQM 术语表提供评分的类别权重：

```python
from kttc.core.mqm import MQMScorer

# 从术语表加载权重
scorer = MQMScorer(use_glossary_weights=True)

# 来自 glossary/en/mqm_core.json 的类别权重：
# accuracy: 1.5（准确性错误惩罚更高）
# terminology: 1.2
# fluency: 1.0
# style: 0.8（风格问题惩罚较低）
```

## CLI 命令

### 查看术语表内容

```bash
# 列出所有可用术语表
kttc glossary list

# 查看特定术语表
kttc glossary view --language ru --name grammar_reference

# 在术语表中搜索
kttc glossary search --language zh --query "量词"

# 以 JSON 格式查看
kttc glossary view --language en --name mqm_core --format json
```

### 术语命令

```bash
# 验证错误类型
kttc terminology validate --error-type "mistranslation" --language en

# 列出 MQM 错误类别
kttc terminology list-categories

# 获取错误定义
kttc terminology define --error-type "grammar" --language zh
```

## 术语表文件格式

术语表以 JSON 文件格式存储，结构如下：

### MQM Core 格式

```json
{
  "metadata": {
    "name": "MQM Core Error Taxonomy",
    "version": "1.0",
    "language": "zh",
    "description": "W3C 多维质量指标"
  },
  "categories": {
    "accuracy": {
      "weight": 1.5,
      "subcategories": {
        "mistranslation": {
          "definition": "译文未准确表达源文本的意思",
          "examples": ["猫 → 狗", "买 → 卖"],
          "severity_guidelines": "Major 或 Critical"
        }
      }
    }
  }
}
```

### 语言特定格式

```json
{
  "metadata": {
    "name": "中文分类词参考",
    "version": "1.0",
    "language": "zh"
  },
  "classifiers": {
    "individual": {
      "个": {
        "usage": "通用个体量词",
        "examples": ["一个人", "三个苹果"]
      },
      "本": {
        "usage": "书籍、刊物等",
        "examples": ["一本书", "两本杂志"]
      }
    }
  }
}
```

## 高级用法

### 自定义术语表

您可以在 `glossaries/` 目录中添加自定义术语表：

```bash
# 创建自定义术语表
mkdir -p glossaries/zh
cat > glossaries/zh/medical_terms.json <<EOF
{
  "metadata": {
    "name": "医学术语",
    "version": "1.0",
    "language": "zh",
    "domain": "medical"
  },
  "terms": {
    "心肌梗死": {
      "definition": "心脏病发作",
      "synonyms": ["心梗", "MI"],
      "avoid": ["心脏骤停"]
    }
  }
}
EOF
```

### 程序化访问

```python
from kttc.terminology import GlossaryManager

# 初始化管理器
manager = GlossaryManager()

# 加载术语表
mqm_data = manager.load_glossary("zh", "mqm_core")

# 获取特定定义
definition = mqm_data["categories"]["accuracy"]["subcategories"]["mistranslation"]

# 验证错误类型
is_valid, info = manager.validate_mqm_error("mistranslation", "zh")
```

### 禁用术语表增强

```python
from kttc.agents.parser import ErrorParser

# 解析时不增强
errors = ErrorParser.parse_errors(
    response=llm_response,
    enrich_with_glossary=False  # 禁用增强
)

# 或使用默认权重进行 MQM 评分
from kttc.core.mqm import MQMScorer

scorer = MQMScorer(use_glossary_weights=False)  # 使用默认权重
```

## 优势

### 1. 提高准确性

术语表提供精确的语言学定义，帮助代理：
- 更准确地识别语言特定错误
- 理解语法模式和规则
- 验证术语使用

### 2. 更好的错误描述

增强的错误描述包括：
- 语言学解释
- 语法规则
- 使用示例
- 更正建议

### 3. 一致的评分

MQM 术语表确保：
- 标准化的错误分类
- 一致的类别权重
- 可重现的质量评分
- 符合行业标准（W3C MQM）

### 4. 多语言支持

同一术语表框架适用于：
- 5 种语言（en、ru、zh、hi、fa）
- 多个错误类别
- 语言特定功能
- 文化适应

## 文件位置

```
glossaries/
├── en/
│   └── mqm_core.json          # 英语 MQM 定义
├── ru/
│   ├── mqm_core.json          # 俄语 MQM 定义
│   └── grammar_reference.json # 俄语语法规则
├── zh/
│   ├── mqm_core.json          # 中文 MQM 定义
│   └── classifiers.json       # 中文量词
├── hi/
│   ├── mqm_core.json          # 印地语 MQM 定义
│   └── grammar_reference.json # 印地语语法规则
└── fa/
    ├── mqm_core.json          # 波斯语 MQM 定义
    └── grammar_reference.json # 波斯语语法规则
```

## 相关文档

- **[CLI 命令](../reference/cli-commands.md)** - 术语表和术语命令
- **[架构](../explanation/architecture.md)** - 术语表如何与代理集成
- **[语言特性](language-features.md)** - 语言特定功能

## 故障排除

### 术语表未加载

**问题：** 代理未使用术语表数据

**解决方案：**
```python
# 检查术语表文件是否存在
import os
glossary_path = "glossaries/zh/classifiers.json"
print(f"存在: {os.path.exists(glossary_path)}")

# 启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 自定义术语表未找到

**问题：** 自定义术语表未加载

**解决方案：**
- 确保文件在正确目录：`glossaries/{language}/{name}.json`
- 检查 JSON 语法是否有效
- 验证 metadata 部分是否存在

### 错误增强不工作

**问题：** 错误未用定义增强

**解决方案：**
```python
# 显式启用增强
from kttc.agents.parser import ErrorParser

errors = ErrorParser.parse_errors(
    response=response,
    enrich_with_glossary=True,  # 显式启用
    language="zh"  # 指定语言
)
```

## 性能

**术语表加载：**
- 加载时间：每个术语表 <100ms
- 内存使用：每个术语表 ~1-5MB
- 首次加载后缓存

**错误增强：**
- 查找时间：每个错误 <1ms
- 无需 API 调用
- 确定性结果

## 未来增强

计划的术语表功能：
- 通过 Web UI 编辑术语表
- 领域特定术语表（法律、医学、技术）
- 术语表版本控制
- 社区贡献的术语表
- 术语表合并和冲突解决
