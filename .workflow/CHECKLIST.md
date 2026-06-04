# CHECKLIST — 检查清单

> 各阶段的质量检查项。帮助正确完成任务，确保不遗漏关键验证。
> 检查项来源：自我发现的问题、用户提供的要求、历史经验教训。
>
> **审查要求**: 使用 subagent 在独立上下文中客观检查。发现问题必须诚实报告。

## 审查原则

- 基于事实，不基于假设
- 可验证，可复现
- 明确标注：✅ 通过 / ❌ 不通过 / ⬜ 不适用

## 阶段一：PyTorch 接口封装

### YAML 配置
- [ ] func schema 语法正确，参数类型、名称、默认值与设计文档一致
- [ ] op_api 版本范围正确
- [ ] in-place 参数标记 `Tensor(a!)` 等使用正确
- [ ] 可选参数使用 `Tensor?` 标记
- [ ] 返回值类型和数量正确

### C++ 适配
- [ ] 文件命名遵循 `<OpName>KernelNpuOpApi.cpp` 规范
- [ ] 命名空间为 `op_api`
- [ ] 引入正确的头文件（`OpApiInterface.h`、`op_api_common.h`）
- [ ] 平台检查：`SocVersion >= Ascend950`
- [ ] 输入校验：shape、dtype、维度、head_nums 长度和值
- [ ] 可选输入非空校验（q_rotation、k_rotation、v_scale）
- [ ] 输出 tensor 正确分配（shape、dtype）
- [ ] EXEC_NPU_CMD 参数顺序与 ACLNN 接口一致
- [ ] fp8 tensor 是否需要 TensorWrapper 处理
- [ ] 错误信息包含 `OPS_ERROR(ErrCode::PARAM)` 等错误码

### Schema 测试
- [ ] func schema 字符串与 YAML 中完全一致
- [ ] version 数组与 YAML 中 op_api 版本一致

### Python 单元测试
- [ ] 测试类继承 `TestCase`
- [ ] 使用 `@SupportedDevices(['Ascend950'])` 装饰器
- [ ] 覆盖正常路径（基本 shape、dtype 验证）
- [ ] 覆盖报错路径（缺失可选输入、head_nums 不合法等）
- [ ] 使用 `run_tests()` 入口

### 代码风格
- [ ] 文件命名与已有算子风格一致
- [ ] Copyright 头正确
- [ ] 常量命名遵循仓库规范（如 `DIM_ZERO` 等）
- [ ] 无遗留 TODO/FIXME 未处理

## 通用检查项

### 代码规范
- [ ] 文件命名与注释匹配
- [ ] Header guard 与文件名匹配
- [ ] 无遗留 TODO/FIXME 未处理

### 功能验证
- [ ] 编译通过（LSP 检查无错误）
- [ ] 测试通过（语法和逻辑正确）
- [ ] 不影响已有功能
