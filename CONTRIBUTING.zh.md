[English](CONTRIBUTING.md) · [Русский](CONTRIBUTING.ru.md) · **中文**

# 为 KTTC 贡献代码

感谢您对为 KTTC（知识翻译嬗变核心）贡献代码的兴趣！我们欢迎来自社区的贡献。

## 目录

- [贡献者许可协议](#贡献者许可协议)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码质量标准](#代码质量标准)
- [拉取请求流程](#拉取请求流程)
- [编码标准](#编码标准)

## 贡献者许可协议

**重要**：通过向本项目提交贡献，您同意以下条款：

### 版权许可授权

您特此向 KTTC AI (https://github.com/kttc-ai) 以及接收由 KTTC AI 分发的软件的接收方授予永久的、全球的、非排他性的、免费的、免版税的、不可撤销的版权许可，以复制、准备衍生作品、公开展示、公开执行、再许可和分发您的贡献及此类衍生作品。

### 专利许可授权

您特此向 KTTC AI 以及接收由 KTTC AI 分发的软件的接收方授予永久的、全球的、非排他性的、免费的、免版税的、不可撤销的（除本节中规定的情况外）专利许可，以制造、委托制造、使用、销售要约、销售、进口和其他方式转让作品，其中此类许可仅适用于您可许可的、因您的贡献单独或与提交此类贡献的作品结合而必然被侵犯的专利权利要求。

### 声明

您声明：
1. 您在法律上有权授予上述许可
2. 您的每项贡献都是您的原创作品
3. 您的贡献提交包括您所知晓的任何第三方许可或其他限制的完整详情

### 代表雇主的提交

如果您作为雇佣工作的一部分进行贡献，您声明您已获得雇主的许可进行贡献，或您的雇主已放弃对您向 KTTC AI 贡献的此类权利。

### 协议

**通过向本项目提交拉取请求或贡献，您确认已阅读本贡献者许可协议并同意其条款。**

---

**企业贡献者注意事项**：如果您代表公司进行贡献，我们可能需要企业贡献者许可协议（CCLA）。请通过项目仓库联系我们。

## 如何贡献

### 报告错误

如果您发现错误，请在 GitHub 上创建问题，包含：
- 清晰、描述性的标题
- 重现问题的步骤
- 期望行为与实际行为的对比
- 您的环境（操作系统、Python 版本、KTTC 版本）
- 任何相关的日志或错误信息

### 建议改进

我们欢迎功能请求！请创建包含以下内容的问题：
- 功能的清晰描述
- 用例和好处
- 任何相关的示例或模型

### 贡献代码

1. Fork 仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 按照我们的编码标准进行修改
4. 根据需要编写或更新测试
5. 确保所有质量检查通过
6. 提交更改（请参见下面的提交信息指南）
7. 推送到您的 fork
8. 开启拉取请求

## 开发环境设置

### 先决条件

- Python 3.11 或 3.12
- Git

### 设置说明

```bash
# 克隆您的 fork
git clone https://github.com/YOUR_USERNAME/kttc.git
cd kttc

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 安装开发依赖
python3.11 -m pip install -e ".[dev]"

# 安装预提交钩子（自动）
# 钩子已在 .git/hooks/pre-commit 中配置
```

### 运行测试

```bash
# 运行所有测试
python3.11 -m pytest

# 运行特定测试套件
python3.11 -m pytest tests/unit/
python3.11 -m pytest tests/integration/

# 运行并查看覆盖率
python3.11 -m pytest --cov=kttc --cov-report=html
```

## 代码质量标准

所有代码贡献必须满足由自动检查强制执行的严格质量标准。

### 预提交钩子

安装的预提交钩子在每次提交前自动运行以确保代码质量。

#### 检查内容：

1. **Black** - 代码格式化
   - 确保一致的代码风格
   - 在检查模式下运行（不会修改文件）

2. **Ruff** - Python 代码检查器
   - 捕获常见错误和代码异味
   - 强制执行最佳实践

3. **MyPy** - 静态类型检查器（仅 src/ 文件）
   - 在 `--strict` 模式下运行
   - 确保类型安全

#### 钩子已安装

预提交钩子位于 `.git/hooks/pre-commit` 并自动运行。

#### 示例输出

**当代码有问题时：**
```
🔍 Running pre-commit checks...

📝 Files to check:
  - src/kttc/example.py

▶️  Running Black...
✗ Black: Code needs formatting
  Run: python3.11 -m black src/kttc/example.py

❌ Pre-commit checks FAILED
```

**当代码正确时：**
```
🔍 Running pre-commit checks...

📝 Files to check:
  - src/kttc/example.py

▶️  Running Black...
✓ Black: Code is formatted

▶️  Running Ruff...
✓ Ruff: No linting issues

▶️  Running MyPy (strict mode)...
✓ MyPy: No type errors

✅ All pre-commit checks PASSED
```

### 修复问题

如果预提交钩子失败，请在提交前修复问题：

```bash
# 修复格式
python3.11 -m black src/ tests/ examples/

# 修复代码检查问题
python3.11 -m ruff check --fix src/ tests/ examples/

# 检查类型
python3.11 -m mypy --strict src/
```

### 跳过检查（不推荐）

在极少数情况下，您可以跳过钩子：

```bash
git commit --no-verify
```

**警告：** 这只应在特殊情况下使用。

## 手动质量检查

您可以在提交前手动运行检查：

```bash
# 运行所有检查
python3.11 -m black --check src/ tests/ examples/
python3.11 -m ruff check src/ tests/ examples/
python3.11 -m mypy --strict src/

# 自动修复问题
python3.11 -m black src/ tests/ examples/
python3.11 -m ruff check --fix src/ tests/ examples/
```

## 代码风格要求

### 仅使用英语

- 所有代码、注释、文档和提交信息必须使用英语
- 源代码中不允许使用西里尔文（俄文）文本
- 详情请参阅 `claude.md`

### 类型提示

- 所有函数必须有类型提示（严格模式）
- 使用现代 Python 3.11+ 语法：`str | None` 而不是 `Optional[str]`

### 格式化

- 遵循 Black 的代码风格（行长度：100）
- 由 Ruff 排序导入
- 使用 Google 风格的文档字符串

### 测试

- 为新功能编写测试
- 保持 >80% 的代码覆盖率
- 提交前所有测试必须通过

## Python 版本

本项目需要 **Python 3.11+**

始终使用：
```bash
python3.11 -m <command>
```

而不是：
```bash
python3 -m <command>  # 错误 - 这是 Python 3.9
```

详情请参阅 `claude.md`。

## 拉取请求流程

1. **更新文档**：如需要，更新 README.md 或其他文档
2. **添加测试**：确保新功能有适当的测试覆盖率（>80%）
3. **遵循编码标准**：使用 Black 进行格式化，Ruff 进行代码检查，MyPy 进行类型检查
4. **编写清晰的提交信息**：使用常规提交格式（见下文）
5. **保持 PR 专注**：每个 PR 只包含一个功能或修复
6. **通过所有质量检查**：预提交钩子必须通过

### 提交信息指南

我们遵循[常规提交](https://www.conventionalcommits.org/)：

```
<类型>(<范围>): <主题>

<正文>

<页脚>
```

**类型：**
- `feat`：新功能
- `fix`：错误修复
- `docs`：文档更改
- `style`：代码风格更改（格式化，无逻辑更改）
- `refactor`：代码重构
- `test`：添加或更新测试
- `chore`：维护任务

**示例：**
```
feat(agents): add context-aware agent for cultural nuances

Implement new agent that analyzes cultural context in translations
and suggests culturally appropriate alternatives.

Closes #123
```

## 编码标准

### 文档

- 编写清晰、自文档化的代码
- 为所有公共模块、函数、类和方法添加文档字符串（Google 风格）
- 使用有意义的变量和函数名
- 保持函数专注和简小
- 为复杂逻辑添加内联注释

### 最佳实践

- 为新功能编写单元测试
- 测试边缘情况和错误条件
- 使用描述性测试名称
- 避免深层嵌套（最多 3-4 层）
- 遵循单一职责原则

## 有问题？

- 查看 `README.md` 了解项目概述
- 查看 `CLAUDE.md` 了解开发指南
- 在 GitHub 上开启问题

---

感谢您为 KTTC 贡献代码！您的贡献有助于为每个人改进翻译质量保证。
