---
name: op-test
description: Generate UT test cases for torch_npu custom operators, following the test patterns in test/test_custom_ops/. Use when the user asks to write, create, or generate unit tests for a torch_npu operator (e.g. "write tests for npu_xxx", "generate UT for this operator"). Produces a complete test_{operator_name}.py file placed in test/test_custom_ops/.
---

# torch_npu 算子 UT 测试生成

为 torch_npu 自定义算子生成单元测试，遵循 `test/test_custom_ops/` 目录下的标准模式。

## 工作流

1. 读取 `references/patterns.md` 了解两类测试模式及选择依据
2. 根据算子类型选择合适模式，参考同类已有测试文件
3. 输出到 `test/test_custom_ops/test_{operator_name}.py`
4. 运行测试验证：`python test/test_custom_ops/test_{operator_name}.py`

## 核心规范

- 测试类继承 `TestCase`，从 `torch_npu.testing.testcase` 导入
- 每个测试方法加 `@SupportedDevices(['Ascend910B'])` 装饰器
- 用 `torch.manual_seed(0)` 固定随机种子保证可复现
- 精度断言：fp16 用 `0.001`，fp32 用 `0.0001`
- 文件末尾加 `if __name__ == "__main__": run_tests()`
