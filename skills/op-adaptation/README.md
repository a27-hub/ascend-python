# op-adaptation 技能使用说明

## 概述

`op-adaptation` 技能用于将 CANN (Ascend) 算子集成到 torch_npu 后端，遵循标准化的工作流程。

## 前置条件

在使用本技能前，请确保您已准备好以下输入文件：

1. **算子文档** (`.md` 文件) - 包含 aclnn 函数签名、参数类型、输入/输出标记、数据类型/形状约束
2. **算子定义文件** (`_def.cpp` 文件) - 包含 REQUIRED/OPTIONAL 标记、Attr 默认值、SymInt 类型

## 使用示例

命令行
```
/op-adaptation 将aclnnDynamicQuant算子接入torch_npu后端，对应的输入文件位于./aclnn_input/下。
```

## 工作流程概览

完整的算子集成流程包含 9 个步骤：

```
1. 解析输入文件
      ↓
2. 确认算子信息
      ↓
3. 生成 YAML
      ↓
4. 生成 C++ 代码
      ↓
5. 注册 Meta
      ↓
6. 编译构建
      ↓
7. 运行测试
      ↓
8. 生成 API 文档
      ↓
9. 生成总结报告
```

---

## 详细步骤说明

### 步骤 1: 解析输入文件

**输入**: `.md` 文档 + `_def.cpp` 定义文件

**提取内容**:

| 来源文件 | 提取信息 |
|---------|---------|
| `.md` 文件 | aclnn 函数签名、参数类型、输入/输出标记、数据类型/形状约束 |
| `_def.cpp` 文件 | REQUIRED/OPTIONAL 标记、Attr 默认值、SymInt 类型 |

**冲突解决规则**:

| 冲突类型 | 优先使用 |
|---------|---------|
| 可选性 | `_def.cpp` 的 ParamType |
| 默认值 | `_def.cpp` 的 Attr |
| 命名 | `_def.cpp` 的 snake_case |
| 约束 | `.md` 文档 |

**信息分类**:

- **可从文件派生**: 参数名、类型、可选性、默认值、格式支持、工作空间管理、形状/数据类型约束
- **需询问用户**: CANN 版本分发 (V2/V3/V5)、SoC 特定分发、.List 重载、额外函数变体、internal_format_opapi 字段、特殊数据格式支持

**完整性自检**:

解析完成后，执行以下检查：
1. 遍历 `_def.cpp` 中所有 `this->Input("xxx")` 调用，确保每个参数都记录在 CHECKLIST.md
2. 检查是否有 "not enabled" 注释的参数，标记为 OPTIONAL
3. 解决默认值冲突（优先使用 `_def.cpp` 的 Attr 值）

**输出**: 生成 `CHECKLIST.md`（使用 `references/checklist-template.md` 模板）

---

### 步骤 2: 确认算子信息

向用户询问以下信息：

- [ ] CANN 版本变体（V2/V3/V5）
- [ ] SoC 特定分发（Ascend950 等）
- [ ] .List 重载需求
- [ ] 额外的 torch_npu 函数变体
- [ ] internal_format_opapi 字段
- [ ] 特殊数据格式支持（mxfp4 等）

**完成后**: 在 CHECKLIST.md 中标记此步骤为完成

---

### 步骤 3: 生成 YAML 配置

**目标文件**: `op_plugin/config/op_plugin_functions.yaml`

**参考文档**: `references/yaml-configuration-guide.md`

**关键要点**:
- 函数签名必须与 C++ 实现匹配
- `int[]?` 类型映射为 `c10::OptionalIntArrayRef`
- `str` 类型映射为 `c10::string_view`

**完成后**: 更新 CHECKLIST.md

---

### 步骤 4: 生成 C++ 实现

**目标文件**: `op_plugin/ops/opapi/{Name}KernelNpuOpApi.cpp`

**参考文档**: `references/cpp-implementation-guide.md`

**实现要点**:
- 使用 TensorWrapper 处理张量
- 使用 EXEC_NPU_CMD 调用 aclnn 接口
- 正确处理可选参数和默认值

**完成后**: 更新 CHECKLIST.md

---

### 步骤 5: 注册 Meta

**目标文件**: `op_plugin/python/meta/_meta_registrations.py`

**参考文档**: `references/meta-registration-guide.md`

**完成后**: 更新 CHECKLIST.md

---

### 步骤 6: 编译构建

**命令**:

```bash
# 如果 build 目录已存在，先删除
rm -rf build

# 执行构建
bash ci/build.sh --python=3.10 --pytorch=v2.10.0 2>&1 | tee build.log

# 安装生成的 wheel 包
pip install --force-reinstall build/pytorch/dist/torch_npu-*.whl
```

**构建失败处理**:
- 阅读 `build.log` 定位根本原因
- 常见问题：
  - YAML 函数签名与 C++ 实现不匹配
  - 类型映射错误（如 `int[]?` → `c10::OptionalIntArrayRef`）

**完成后**: 更新 CHECKLIST.md

---

### 步骤 7: 运行测试

使用 `op-test` 技能生成和执行测试。

**收集**: 测试报告（用于最终总结）

**完成后**: 更新 CHECKLIST.md

---

### 步骤 8: 生成 API 文档

使用 `op-apidoc` 技能创建文档：

**目标文件**: `docs/zh/custom_APIs/torch_npu/torch_npu-npu_{operator_name}.md`

**完成后**: 更新 CHECKLIST.md

---

### 步骤 9: 生成集成总结

使用 `references/summary-template.md` 模板创建总结报告：

**目标文件**: `notes/summary_{YYYY-MM-DD}.md`

**要求**:
- 使用中文
- 包含步骤 7 的测试报告
- 如当天已有总结文件，追加到现有文件

**完成后**: 标记 CHECKLIST.md 为全部完成

---

## 重要提醒

### ⚠️ 范围控制（关键）

**只实现以下内容**:
1. 输入文件（.md 和 _def.cpp）中明确定义的内容
2. 用户在步骤 2 中明确请求的内容

**不要添加**:
- 输入文件中不存在的额外参数
- 用户未请求的额外函数变体
- 参考实现中但输入文件中不存在的功能

### ✅ 最佳实践

1. **始终生成并维护 CHECKLIST.md** - 在整个工作流程中跟踪进度
2. **参考类似算子** - 按类别或参数模式搜索，仅学习实现模式（TensorWrapper、EXEC_NPU_CMD 用法），不要复制不在输入文件中的功能
3. **检查生成的签名** - 添加 YAML 后在 `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h` 中验证
4. **每个步骤后更新 CHECKLIST.md**
5. **始终生成集成总结**

---

## 常见问题

### 构建错误：undefined reference to op_api::npu_xxx

**原因**: YAML 函数签名与 C++ 实现不匹配

**解决**: 检查 `build/pytorch/third_party/op-plugin/op_plugin/OpApiInterface.h` 中的生成签名

常见类型映射错误：
| YAML 类型 | 正确 C++ 类型 | 错误 C++ 类型 |
|----------|-------------|-------------|
| `int[]?` | `c10::OptionalIntArrayRef` | `c10::optional<at::IntArrayRef>` |
| `str` | `c10::string_view` | `std::string` |

### 测试失败：shape mismatch

**可能原因**:
- INT32 输出（INT4 打包）: `last_dim = input_last_dim / 8`
- 量化 scale 形状因 quant_mode 而异

---

## 参考文档位置

| 文档 | 路径 |
|-----|------|
| 检查清单模板 | `references/checklist-template.md` |
| YAML 配置指南 | `references/yaml-configuration-guide.md` |
| C++ 实现指南 | `references/cpp-implementation-guide.md` |
| Meta 注册指南 | `references/meta-registration-guide.md` |
| 总结模板 | `references/summary-template.md` |

---

## 相关技能

- `op-test` - 生成和执行算子测试
- `op-apidoc` - 生成 API 文档
