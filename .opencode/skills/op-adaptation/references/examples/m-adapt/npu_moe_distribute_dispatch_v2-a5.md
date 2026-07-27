# npu_moe_distribute_dispatch_v2 A5 平台适配（M-adapt 案例 · 版本分派范本）

> 本案例是 op-plugin **gap #5 覆盖度校验 + 运行时门控**的**正解范本**。
> 适用于 aclnn **多版本变体**(V2/V3/V4…)、需在 A5 上做版本探测与回退的算子。
> 本案例**不是一个新 m-adapt 操作的端到端记录**——`npu_moe_distribute_dispatch_v2` 实际**已通过 master PR 适配完毕**。本案例的价值是把 codebase 里**做对了的成熟工程模式**（`MoeDistributeDispatchV2KernelOpApi.cpp`）固化为 skill 模板。

**type**: `domain-reference`（不是 workflow-review — 没有"这次做了什么"的时间线，而是把已有的多版本 aclnn 模式归档为可复用知识）

**基线环境**：op-plugin 26.0.0 / CANN 9.0.0-beta.2 / 硬件 `Ascend950PR`

## Background

`npu_moe_distribute_dispatch_v2` 是 MoE 模型的分布式 dispatch 算子（专家路由/通信）。在 op-plugin 26.0.0 分支**已是 aclnn 适配**：

- yaml: `op_api: all_version`（无 acl_op）
- 手写 cpp: `op_plugin/ops/opapi/MoeDistributeDispatchV2KernelOpApi.cpp`
- **实际 aclnn 调用是版本分派的**：根据当前 CANN + A5 是否有 `aclnnMoeDistributeDispatchV4/V3/V2`，动态选最合适的版本（V4 优先 → V3 兜底 → V2 最后）

CANN 侧的复杂情况：
- `aclnnMoeDistributeDispatch` / `V2` / `V3` / `V4` 四个 aclnn 符号**都存在**（链接期 OK）；
- 但 ascend950 opp config **只登记到 `moedistributedispatchv2`**——V3/V4 在 A5 上**没有 kernel**（要 CANN 9.1.0 才有）；
- 写成老练的 cpp：跑时探测、能用 V4 走 V4、不能用 V4 走 V3、还不能走 V2、传了 V3/V4 新参数时**硬报错**（让用户升级 CANN），不静默丢弃。

这个模式**是 CANN 版本/A5 覆盖度问题的标准解**——gap #5 的所有"硬编 `EXEC_NPU_CMD`"反例，对应到本模式就是正确解。

## Solution

**核心范式：版本分派 + 特性 gate + 优雅报错**，三件套：

```cpp
// A5 branch（外层 GetSocVersion() 路由已切到 A5）
if (check_aclnn_kernel_available("aclnnMoeDistributeDispatchV4")) {
    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV4, /* ...V4 full args... */);
} else if (check_aclnn_kernel_available("aclnnMoeDistributeDispatchV3")) {
    // V4 独有参数 (performance_info) 在 V3 路径上必须硬报错
    TORCH_CHECK(!performance_info.has_value(),
        "The performance_info is not supported in this CANN version, "
        "aclnnMoeDistributeDispatchV4 is not available, please update CANN version.",
        OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV3, /* ...V3 args... */);
} else {
    // V3 独有参数 (elastic_info / zero_expert_num / copy_expert_num / const_expert_num)
    // 在 V2 路径上必须硬报错
    TORCH_CHECK(!check_v3_param(elastic_info, zero_expert_num, copy_expert_num, const_expert_num),
        "The aclnnMoeDistributeDispatchV3 is not supported",
        OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV2, /* ...V2 args... */);
}
```

## File Changes（说明性，非本次端到端动作）

实际本案例**没有 file change**——所有 cpp 已在 master 合并。下方列出"如果要新做一个类似算子该碰哪些文件"，给后来者参考：

| File | Action | Details |
|------|--------|---------|
| `op_plugin/ops/opapi/<OpName>KernelNpuOpApi.cpp` | 新增/修改 | 套用上述三段式骨架 |
| `op_plugin/config/op_plugin_functions.yaml` | 修改 | 加 `op_api: all_version` |
| `op_plugin/python/meta/_meta_registrations.py` | 新增 | `@impl` 一行 |
| `test/.../test_<op_name>.py` | 修改 | `@SupportedDevices(['Ascend910A','Ascend910B'])` vs `['Ascend950']` |

### M-coverage 必须先做的三步（写 C++ 之前！）

1. **识别代码实际调用的精确变体**：本例是 `aclnnMoeDistributeDispatchV4/V3/V2`——**精确到 V4**，不是基名。
2. **grep 装的 CANN 的 ascend950 opp**（注意是**精确变体名**不是基名）：
   ```bash
   D=$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/config/ascend950
   grep -rhoiE '"[a-z0-9]*moedistributedispatch[a-z0-9]*"' "$D"/*.json \
     | tr 'A-Z' 'a-z' | sort -u
   # → 应看到 "moedistributedispatchv2" (本例 V4/V3 缺)
   ```
3. **双向警告**（这是 skill #5 修复后的精修）：
   - **假阳性**（moe 形态）：opp 只有 `v2` → 别误判 `V4` 在 A5 有；可能它根本没 kernel。
   - **假阴性**（grouped_matmul 形态）：`V4/V5` 共用 base `groupedmatmul` kernel → 查 `v4` 查不到**不代表缺**，要再查基名。
   - 不确定时**以实测为准**（在 A5 上实际 `check_aclnn_kernel_available("aclnnXxxV4")` 探）。

## Pitfalls (本模式反例 —— 别这么写)

- **硬编 `EXEC_NPU_CMD(aclnnXxxV4, ...)` 不探测**：V4 在 9.0.0 的 A5 上没 kernel → 调用时崩，难以定位。
- **只 `if (V4) V4 else V2` 二段，不报错**：用户传了 V3/V4 新参（如 `performance_info`）会被**静默丢弃**，下游错乱。
- **粗查 opp kernel 名**：`grep "moedistributedispatch"`（基名）会因 v2 命中而**假阳性**误判 V4 在场。
- **用 `c10_npu::GetSocVersion()` 替代 `check_aclnn_kernel_available`**：前者只查 SoC，不查 CANN 侧 kernel 是否真有；版本分派场景必须用后者。
- **忘记在 V2 兜底分支加 `TORCH_CHECK` 守门**：用户传 V3/V4 独占参数时不会报错，直接走 V2（参数被吞）→ 输出形状对不上。

## A5 Platform Specific Info (M-adapt only)

### A5 adaptation basics

| Field | Value |
|-------|-------|
| Target SoC | Ascend950 (Atlas A5 training series) |
| aclnn operator | `aclnnMoeDistributeDispatchV2` / `V3` / `V4`（**多版本**）|
| Routing | `GetSocVersion()` (外层) + `check_aclnn_kernel_available()` (内层版本分派) |
| Source of truth | `MoeDistributeDispatchV2KernelOpApi.cpp` line 183-211 |

### A5-specific handling

| Aspect | Description | Applies? |
|--------|-------------|----------|
| **version dispatch** | `aclnnV4` → `aclnnV3` → `aclnnV2` 三级回退 | **是（核心）** |
| **per-feature `TORCH_CHECK` gate** | V4-only 参数（`performance_info`）在 V3/V2 路径硬报错 | **是** |
| **multi-output internal API** | 多输出张量（按 V4/V3/V2 协议不同）| 是 |
| dtype cast | aclnn 内部处理 | 否 |
| output transpose | 无 | 否 |
| backward | 不涉及 | N/A |
| aclnn new params | V3/V4 各自新增参数 | **是** |
| A5-specific constraints | V3/V4 需 CANN 9.1.0（9.0.0 缺 A5 kernel）| **是** |

### A5 test design notes

- **平台隔离**：所有用例必须 `@SupportedDevices(['Ascend950'])`——A2/A3 不适用此模式。
- **三套版本测试**：
  - V4 可用环境：跑全功能用例（含 V3/V4 新参）；
  - 仅 V3 可用环境：含 V3 新参可跑，含 V4-only 参应报 PARAM 错；
  - 仅 V2 可用环境：含任何 V3+ 参应报 PARAM 错。
- **CANN 版本覆盖**：测试矩阵需覆盖 9.0.0（仅 V2）/ 9.1.0（全可用）/ 9.2.0+（向后兼容）。
- **error 路径**：传 V3/V4 独占参数到不兼容 CANN 应抛 `RuntimeError` 而非崩。

## 适用范围与互补案例

**本案例覆盖**：
- aclnn **多版本**(V2..VN) 在同一 op 中分派
- 旧版本缺 Vn+1 独占参数时需守门
- 升 CANN 时无需重编译 op-plugin（CANN 升级后 `check_aclnn_kernel_available` 自动探测到 V4）
- 与 M-coverage 双警告配套使用

**不覆盖的请看**：
| 场景 | 推荐看 |
|------|--------|
| acl_op-only、acl_op → op_api 全链路（含整编/A5 实测）| [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| 简单 aclnn、**无版本分派** | [npu_iou-a5](npu_iou-a5.md), [npu_ciou-a5](npu_ciou-a5.md), [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| 算子**已通过 gen_opapi 适配**（不应触发 M-adapt）| [npu_swiglu-a5](npu_swiglu-a5.md) |
| 按入参**特性** gate（`sink.has_value` 决定 V5/V3/V2）+ 双 layout 分派 | 参考 `FlashAttentionV2KernelNpuOpApi.cpp`（更复杂版本）|
| 编译期 cgroup/MAX_JOBS | [npu_mish_backward-a5](npu_mish_backward-a5.md) + `build-guide.md` 顶部 |
| M-coverage 完整流程 | `m-adapt-guide.md` §4 |
| `check_aclnn_kernel_available` API 细节 | `m-adapt-cpp-patterns-guide.md` §4 |
