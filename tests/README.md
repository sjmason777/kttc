# KTTC 测试套件

现代化、快速且严格的测试套件，遵循 2025 年最佳实践。

## 测试结构（测试金字塔）

```
tests/
├── unit/                    # 58% (37 个测试) - 快速（总共约 1.5 秒）
│   ├── test_cli.py          # CLI 参数解析，输出格式（13 个测试）
│   ├── test_agents.py       # 带有模拟 LLM 的代理逻辑（12 个测试）
│   └── test_orchestrator.py # 编排和 MQM 评估（12 个测试）
├── integration/             # 31% (20 个测试) - 中等速度（约 1.9 秒）
│   └── test_cli_flow.py     # 完整的 CLI 工作流与真实组件（20 个测试）
├── e2e/                     # 11% (7 个测试) - 慢速（使用 --run-e2e 运行）
│   └── test_real_api.py     # 真实的 Anthropic API 调用（7 个测试）
├── conftest.py              # 通用 fixtures 和模拟
└── pytest.ini               # 测试标记和配置
```

## 快速开始

```bash
# 运行所有快速测试（unit + integration，约 2 秒）
pytest tests/ -m "not e2e" -v

# 仅运行单元测试（最快，约 1.5 秒）
pytest tests/unit/ -v

# 仅运行集成测试（约 1.9 秒）
pytest tests/integration/ -v

# 运行 E2E 测试（慢速，需要 KTTC_ANTHROPIC_API_KEY）
export KTTC_ANTHROPIC_API_KEY="your-key-here"
pytest tests/e2e/ --run-e2e -v

# 运行特定测试文件
pytest tests/unit/test_cli.py -v

# 使用标记运行测试
pytest -m unit         # 仅单元测试
pytest -m integration  # 仅集成测试
pytest -m e2e --run-e2e  # 仅 E2E 测试

# 运行覆盖率测试（仅快速测试）
pytest tests/ -m "not e2e" --cov=kttc --cov-report=html

# 运行所有测试包括 E2E（需要 API 密钥）
pytest tests/ --run-e2e -v
```

## 测试标记

- `@pytest.mark.unit` - 快速、隔离的模拟测试
- `@pytest.mark.integration` - 多个组件，中等速度
- `@pytest.mark.e2e` - 端到端测试，真实 API 调用
- `@pytest.mark.slow` - 任何超过 5 秒的测试

## 当前状态

**总测试数：** ✅ 64 个测试（57 个快速，7 个 E2E）

| 测试类型     | 数量       | 百分比  | 速度      | 状态   |
|--------------|------------|---------|-----------|--------|
| Unit         | 37         | 58%     | ~1.5秒    | ✅ PASS |
| Integration  | 20         | 31%     | ~1.9秒    | ✅ PASS |
| E2E          | 7          | 11%     | 慢速*     | ✅ PASS |

**\*E2E 测试需要 `--run-e2e` 标志和真实 API 密钥**

### 单元测试详细分解
| 组件         | 测试数 | 状态   |
|--------------|--------|--------|
| CLI          | 13     | ✅ PASS |
| Agents       | 12     | ✅ PASS |
| Orchestrator | 12     | ✅ PASS |

### 集成测试详细分解
| 组件                   | 测试数 | 状态   |
|------------------------|--------|--------|
| CLI Integration Flow   | 4      | ✅ PASS |
| Agent Pipeline         | 2      | ✅ PASS |
| Batch Processing       | 1      | ✅ PASS |
| Error Handling         | 3      | ✅ PASS |
| Language Pairs         | 4      | ✅ PASS |
| Output Formats         | 2      | ✅ PASS |
| Text Processing        | 3      | ✅ PASS |
| Performance            | 1      | ✅ PASS |

### E2E 测试详细分解
| 组件                   | 测试数 | 状态   |
|------------------------|--------|--------|
| Real Anthropic API     | 3      | ✅ PASS |
| Real API Language Pairs| 2      | ✅ PASS |
| Complex Scenarios      | 2      | ✅ PASS |

## 测试理念

### 严格测试
测试**严格**且**诚实** - 它们能发现真实的错误：
- 发现了不正确的 LLM 响应格式（JSON vs ERROR_START/END）
- 验证了 Pydantic 模型约束
- 测试真实行为，而非假设

### AAA 模式
所有测试都遵循 Arrange-Act-Assert 模式：
```python
# Arrange - 准备
agent = AccuracyAgent(mock_llm)

# Act - 执行
errors = await agent.evaluate(task)

# Assert - 断言
assert len(errors) == 1
assert errors[0].category == "accuracy"
```

### 模拟策略
- **单元测试：** 模拟所有外部依赖（LLM、文件 I/O）
- **集成测试：** 仅模拟外部 API
- **E2E 测试：** 无模拟，真实 API 调用

## Fixtures

### 可用的 Fixtures（来自 conftest.py）
- `mock_llm` - 返回无错误
- `mock_llm_with_errors` - 返回准确性错误
- `sample_translation_task` - Hello/Hola 示例
- `sample_translation_error` - ErrorAnnotation 示例
- `sample_qa_report` - QA 报告示例
- `temp_text_files` - 临时源文件/翻译文件
- `cli_runner` - Typer CLI 测试运行器

## 添加新测试

### 1. 单元测试示例
```python
@pytest.mark.unit
class TestMyFeature:
    @pytest.mark.asyncio
    async def test_feature_works(self, mock_llm: Any) -> None:
        # Arrange
        feature = MyFeature(mock_llm)

        # Act
        result = await feature.process()

        # Assert
        assert result is not None
```

### 2. 集成测试示例
```python
@pytest.mark.integration
class TestFullFlow:
    @pytest.mark.asyncio
    async def test_cli_to_report(self, temp_text_files: tuple[Path, Path]) -> None:
        # 测试从 CLI 到报告生成的完整流程
        ...
```

### 3. E2E 测试示例
```python
@pytest.mark.e2e
class TestRealAPI:
    @pytest.mark.asyncio
    async def test_anthropic_translation_check(self) -> None:
        # 需要真实 API 密钥 - 如果不是 --run-e2e 则跳过
        ...
```

## CI/CD 集成

测试在 GitHub Actions 中自动运行：
- **Pre-commit：** 仅单元测试（<2秒）
- **PR 验证：** Unit + Integration（<5分钟）
- **夜间构建：** 所有测试包括 E2E

## 核心原则（2025 年最佳实践）

1. ✅ **快速反馈** - 单元测试 <2 秒
2. ✅ **确定性** - 无不稳定测试，无外部依赖
3. ✅ **单一断言焦点** - 每个测试验证一件事
4. ✅ **有意义的命名** - 测试名称描述行为
5. ✅ **AAA 模式** - Arrange、Act、Assert
6. ✅ **外部 API 模拟** - 测试单元隔离
7. ✅ **严格验证** - 测试发现真实错误

## 故障排除

### 测试缓慢
```bash
# 检查哪些测试慢
pytest --durations=10

# 仅运行快速测试
pytest -m "not slow"
```

### 导入错误
```bash
# 在编辑模式下安装
python3.11 -m pip install -e ".[dev]"
```

### 异步警告
```bash
# 已在 pytest.ini 中配置
# asyncio_mode = auto
```

---

**最后更新：** 2025-11-14
**测试数量：** 64 个测试（37 个 unit，20 个 integration，7 个 E2E）
**总耗时：** 约 2 秒（unit + integration），E2E 测试较慢
**测试金字塔：** 58% unit / 31% integration / 11% E2E（目标：50/40/10）

## 总结

成功使用 2025 年最佳实践从头重建了 KTTC 测试套件：
- **之前：** 941 个测试耗时 30+ 分钟，包含重型 ML 模型
- **之后：** 64 个聚焦测试约 2 秒（快 99.9%！）
- **质量：** 测试发现了真实错误（JSON vs ERROR_START/END 格式问题）
- **覆盖率：** 完整的测试金字塔，包含 unit、integration 和 E2E 测试
- **可维护性：** 清晰结构、AAA 模式、全面的 fixtures