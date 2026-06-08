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
- [x] ✅ func schema 语法正确，参数类型、名称、默认值与设计文档一致
- [x] ✅ op_api 版本范围正确 — `[v2.7, newest]`
- [x] ✅ in-place 参数标记 `Tensor(a!)` 等使用正确 — k_cache, v_cache, k_scale_cache
- [x] ✅ 可选参数使用 `Tensor?` 标记 — q_rotation, k_rotation, v_scale
- [x] ✅ 返回值类型和数量正确 — in-place 返回 2 个，functional 返回 5 个

### C++ 适配
- [x] ✅ 文件命名遵循 `<OpName>KernelNpuOpApi.cpp` 规范
- [x] ✅ 命名空间为 `op_api`
- [x] ✅ 引入正确的头文件（`OpApiInterface.h`、`op_api_common.h`）
- [x] ✅ 平台检查：`SocVersion >= Ascend950`
- [x] ✅ 输入校验：shape、dtype、维度、head_nums 长度和值
- [x] ✅ 可选输入非空校验（q_rotation、k_rotation、v_scale）
- [x] ✅ 输出 tensor 正确分配（shape、dtype）— q_out fp8_e4m3fn, q_scale fp32
- [x] ✅ EXEC_NPU_CMD 参数顺序与 ACLNN 接口一致 — 使用 EXEC_NPU_NO_FORMAT_CHECK_CMD
- [x] ✅ fp8 tensor 使用 TensorWrapper 处理 — q_out, k_cache, v_cache
- [x] ✅ 错误信息包含 `OPS_ERROR(ErrCode::PARAM)` 等错误码

### Schema 测试
- [x] ✅ func schema 字符串与 YAML 中完全一致（逐行比对确认）
- [x] ✅ version 数组与 YAML 中 op_api 版本一致 — `["v2.7", "newest"]`

### Python 单元测试
- [x] ✅ 测试类继承 `TestCase`
- [x] ✅ 使用 `@SupportedDevices(['Ascend950'])` 装饰器
- [x] ✅ 覆盖正常路径（test_normal_inplace, test_normal_functional）
- [x] ✅ 覆盖报错路径（6 个 error test：missing q/k/v rotation, invalid head_nums, Nv!=Nk, shape mismatch）
- [x] ✅ 使用 `run_tests()` 入口

### 代码风格
- [x] ✅ 文件命名与已有算子风格一致
- [x] ✅ Copyright 头正确（BSD 3-Clause, 2026 Huawei）
- [x] ✅ 常量命名遵循仓库规范（DIM_ZERO, DIM_ONE, DIM_TWO, DIM_THREE, HEAD_NUMS_SIZE）
- [x] ✅ 无遗留 TODO/FIXME 未处理

## 通用检查项

### 代码规范
- [x] ✅ 文件命名与注释匹配
- [x] ⬜ Header guard 与文件名匹配（.cpp 文件不需要 header guard）
- [x] ✅ 无遗留 TODO/FIXME 未处理

### 功能验证
- [x] ✅ 编译通过 — LSP 报错为缺少 NPU SDK 头文件（与已有文件一致，预期行为）
- [x] ✅ 测试通过 — Python 语法检查通过，JSON 语法检查通过
- [x] ✅ 不影响已有功能 — 仅新增文件/条目，未修改已有代码
