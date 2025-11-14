# 安装指南

本指南涵盖所有安装方法和可选依赖项。

## 系统要求

- **Python:** 3.11 或更高版本
- **操作系统:** Linux, macOS, Windows
- **API密钥:** OpenAI, Anthropic, GigaChat, 或 Yandex

## 基本安装

### 使用 pip（推荐）

```bash
pip install kttc
```

这将安装核心依赖项（约50MB）：
- CLI界面
- 多代理问答系统
- 俄语NLP（MAWO库）
- 基础多语言支持（spaCy, jieba）

### 使用 pipx（独立环境）

```bash
pipx install kttc
```

优势：
- 与系统Python隔离
- 自动PATH配置
- 便于升级

### 从源码安装（开发）

```bash
git clone https://github.com/kttc-ai/kttc.git
cd kttc
pip install -e ".[dev]"
```

## 可选依赖项

### 指标（语义相似性）

添加句子级相似性指标：

```bash
pip install kttc[metrics]
```

包含：
- sentence-transformers
- 语义相似性评分

### 英语语言支持

使用LanguageTool添加高级英语语法检查：

```bash
pip install kttc[english]
```

**系统要求：**
- Java 17.0 或更高版本
- 约200MB磁盘空间

功能：
- 5,000+语法规则
- 主谓一致性检查
- 冠词检查（a/an/the）
- 介词验证

**安装Java（如需要）：**

```bash
# macOS
brew install openjdk@17

# Ubuntu/Debian
sudo apt install openjdk-17-jre

# Windows
# 从 https://adoptium.net/ 下载
```

**下载spaCy模型：**

```bash
python3 -m spacy download en_core_web_md
```

### 中文语言支持

使用HanLP添加高级中文NLP：

```bash
pip install kttc[chinese]
```

功能：
- 量词验证
- 助词检查（了/过）
- 高精度词性标注（约95%）
- 首次使用时约300MB模型下载

### 所有语言助手

安装所有语言特定增强功能：

```bash
pip install kttc[all-languages]
```

等同于：

```bash
pip install kttc[english,chinese]
```

### 完整安装（开发 + 所有功能）

```bash
pip install kttc[full,dev]
```

## 验证安装

检查KTTC版本：

```bash
kttc --version
```

使用演示模式测试（无需API密钥）：

```bash
echo "Hello" > source.txt
echo "Hola" > trans.txt
kttc check source.txt trans.txt --source-lang en --target-lang es --demo
```

## API密钥设置

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

### 持久配置

添加到你的shell配置文件（`~/.bashrc`, `~/.zshrc`）：

```bash
# KTTC API Keys
export KTTC_OPENAI_API_KEY="sk-..."
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

然后重新加载：

```bash
source ~/.bashrc  # or ~/.zshrc
```

## 升级

### 升级到最新版本

```bash
pip install --upgrade kttc
```

### 升级包含可选依赖项

```bash
pip install --upgrade kttc[all-languages,metrics]
```

## 卸载

```bash
pip uninstall kttc
```

删除已下载的模型：

```bash
# spaCy模型
python3 -m spacy uninstall en_core_web_md

# HanLP模型（如已安装）
rm -rf ~/.hanlp
```

## 故障排除

### Python版本问题

**错误：** `TypeError: unsupported operand type(s) for |`

**解决方案：** 使用Python 3.11+：

```bash
python3.11 -m pip install kttc
```

### 未找到Java（LanguageTool）

**错误：** `java.lang.RuntimeException: Could not find java`

**解决方案：** 安装Java 17+：

```bash
# 检查Java版本
java -version

# 应显示：openjdk version "17.0.x" 或更高版本
```

### 权限被拒绝

**错误：** `ERROR: Could not install packages`

**解决方案：** 使用用户安装：

```bash
pip install --user kttc
```

或使用虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install kttc
```

## 下一步

- [配置](configuration.md) - 配置设置
- [CLI使用](cli-usage.md) - 学习命令
- [快速入门](../tutorials/quickstart.md) - 运行第一次检查