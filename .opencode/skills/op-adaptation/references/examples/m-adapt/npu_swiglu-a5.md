# npu_swiglu A5 平台适配（M-adapt 案例 · "已适配"判别 · `gen_opapi` 形态）

> 本案例演示 M-adapt 中**最常见的 false-positive 来源**：算子已经通过 **`gen_opapi:` 结构化适配器**完成了 A5 适配，**不应当再触发 M-adapt**。
> 适用于"yaml 看起来没有 acl_op 但也没有手写 OpApi.cpp"这类**容易被误判**的算子。

**type**: `domain-reference`

**基线环境**：op-plugin 26.0.0 / CANN 9.0.0-beta.2 / 硬件 `Ascend950PR`

## Background

`npu_swiglu` 是 MoE/LLM 常用的激活函数算子（split-then-SwiGLU）。op-plugin 26.0.0 分支**已是 A5 适配**——但适配代码**完全自动生成**，没手写 cpp。

yaml 长这样（关键部分）：

```yaml
- func: npu_swiglu(Tensor self, int dim=-1) -> Tensor
    op_api: all_version
    exposed: all_version
    gen_opapi:                        # ← 结构化适配器 (自动生成)
      out:
        size: 'swiglu_backward_infershape(self, dim)'
        dtype: self
      exec: aclnnSwiGlu              # ← 直接绑 aclnn
```

特征：
- ✅ 有 `op_api: all_version`（aclnn 路径已注册）
- ❌ **没有** `acl_op:`（无老 ACL 路径）
- ❌ `op_plugin/ops/opapi/` 下**没有** `SwigluKernelNpuOpApi.cpp`（也没 `SwigluKernelOpApi.cpp`）
- ✅ 实际是 `gen_opapi:` 块 + `exec: aclnnSwiGlu` 在编译时**自动生成**翻译代码

A5 实测：`torch_npu.npu_swiglu(x, dim=-1)` **跑通**（输入末维 16 → 输出 8，行为正确）。

## Solution

**核心：识别 `gen_opapi:` 块，立刻报告"已适配、勿动"**。

### 检测命令（Step 0 状态预分类）

```bash
# 看 yaml 块
grep -A6 "func: npu_swiglu(" op_plugin/config/op_plugin_functions.yaml

# 关键检测
grep -A6 "func: npu_swiglu(" op_plugin/config/op_plugin_functions.yaml | grep -q "gen_opapi:" \
  && echo "STATE B: gen_opapi structured adapter — out of scope"
```

### 改做什么：**什么都不做**

`gen_opapi:` 块本身就是 op-plugin 编译时自动产出 `*KernelNpuOpApi.cpp`（含 aclnn 调用的）的机制。**M-adapt 模式与 `gen_opapi` 是互斥的**——按 `SKILL.md` 原文：

> **CRITICAL**: Modify mode does NOT handle gen_opapi structured adapters.

若强行进入 M-adapt 写一份手写 `SwigluKernelNpuOpApi.cpp`，**会和 gen_opapi 生成的代码产生冲突/重复链接**。所以正确行为是 STOP。

## File Changes

**无任何 file change**——M-adapt 在此场景下应当**不写任何东西**。这是 case 的核心价值：**示范"何时不写"**。

| File | Action | Details |
|------|--------|---------|
| (无) | — | 算子已由 `gen_opapi:` 自动适配，**不要**手写 OpApi.cpp、**不要**改 yaml、**不要**碰 meta（meta 自动生成）|

## Pitfalls

- **最致命**：看到 yaml 块没有 acl_op、也没有 `*OpApi.cpp` 手写文件 → 误判"未适配" → 进 M-adapt → 写一份手写 cpp → **链接期重复符号/冲突**。
- **次致命**：套用 4 条件 C3 "ops/acl/ 无老实现 → 用 Create 模式" 的边界规则 → 把 `npu_swiglu` 错送到 Create 模式 → 又生成一份 cpp → 同样冲突。
- **误判的根源**：skill 此前对 `gen_opapi:` **零识别**（gap #4）。本次 skill 更新后加了 M-discover Step 0 状态预分类（A 已适配手写 / **B 已适配 gen_opapi** / C 真候选 / D 该 Create），但**如果用户或 AI agent 直接从 4 条件入口进入，会被 C3 边界规则误导**。**务必先看 Step 0**。
- **检测本身也容易漏**：必须 `grep -A6 "func: <op>(" yaml` 拿到完整块，再 `grep -q "gen_opapi:"`。`grep "<op_name>" yaml` 单独搜只命中 `func:` 行，看不到 `gen_opapi:` 子块。

### `gen_opapi` 的版本区段变体（**yaml 用 `[vN.M, newest]`**）

`npu_swiglu` 的 yaml 是 `op_api: all_version`——只要装了 CANN 任何 ≥ 0 版本都注册该 opapi 路径。但真实代码里**还有第三种形态**——`gen_opapi:` 与**版本区段**同时出现：

```yaml
- func: npu_clipped_swiglu(Tensor x, *, Tensor? group_index=None, int dim=-1, ...)
    op_api: [v2.6, newest]               # ← 最低 CANN 2.6 才注册
    exposed: [v2.6, newest]
    gen_opapi:
      out: { size: 'swiglu_backward_infershape(x, dim)', dtype: x }
      exec: aclnnClippedSwiglu
```

特征：
- `op_api: [v2.6, newest]` 含义：算子 opapi 路径仅在 CANN 2.6 及以上注册；旧版 CANN 调它会**找不到入口**。
- 判定仍属 **State B（已适配-gen_opapi）**——不要进 M-adapt。
- **额外校验**：M-coverage 步骤需追加 **CANN 版本下限校验**——查装 CANN 的 version.info 与 yaml 最低版本号比较。若装的 CANN < v2.6，应**降级到 acl_op 路径**（如果 yaml 同时有 `acl_op:` 行）并报错"please update CANN version"。

> 易踩坑：只检查 `op_api:` 是否为 `all_version` 的话，**会误判 [v2.6, newest] 为"opapi 未注册"**。任何 `op_api:` 值（`all_version` / `[v2.1, newest]` / `latest` 等）都算 State B 存在。

## A5 Platform Specific Info (M-adapt only)

### A5 adaptation basics

| Field | Value |
|-------|-------|
| Target SoC | Ascend950 (Atlas A5 training series) |
| aclnn operator | `aclnnSwiGlu` (codegen 自动绑定) |
| Routing | `op_api: all_version` + 编译时 codegen，**无运行时 SoC 分支** |
| Adaptation form | `gen_opapi:` 结构化适配器（不是手写 cpp）|
| M-adapt verdict | **DO NOT TRIGGER** — 算子已 A5 适配 |

### A5-specific handling

| Aspect | Description | Applies? |
|--------|-------------|----------|
| 形态 | `gen_opapi:` 结构化适配器，**非**手写 cpp | **是（与其他 case 关键差异）**|
| dtype cast | aclnn 内部处理 | 否 |
| output transpose | 无 | 否 |
| backward | 单独的 `npu_swiglu_backward` 走同模式 | N/A（无 backward 在本例）|
| multi-output | 单输出 | 否 |
| aclnn new params | 无 | 否 |
| A5-specific constraints | 无 | 否 |
| M-adapt 触发 | **不要** | **是 (本 case 核心)** |

### A5 test design notes

- **不需要 `@SupportedDevices(['Ascend950'])` 隔离**：gen_opapi 在所有 SoC 走同一段自动生成代码，行为一致。
- **测试统一**：和原 acl_op 时代相同测试套即可，无须分平台。
- **A5 验证**：在 A5 上 `torch_npu.npu_swiglu(x, dim=-1)` 直接跑通，无需任何分支。

## 适用范围与互补案例

**本案例覆盖**：
- 算子**已通过 `gen_opapi:` 适配**——M-adapt 应当**拒绝**触发
- Step 0 状态 B（已适配-gen_opapi）的判别与处置
- M-adapt 与 codegen 路径互斥的事实

**不覆盖的请看**：
| 场景 | 推荐看 |
|------|--------|
| aclnn **多版本**(V2..VN) + `check_aclnn_kernel_available` 回退 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) |
| 算子**已通过手写 cpp 适配**（op_api + 实际有 *KernelNpuOpApi.cpp）| [npu_iou-a5](npu_iou-a5.md), [npu_ciou-a5](npu_ciou-a5.md)（Step 0 状态 A）|
| acl_op-only、需补 op_api（含整编/A5 实测）| [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| 完整 Step 0 状态预分类表 | `m-adapt-guide.md` §2 |
| `gen_opapi:` 字段语义（`size` / `dtype` / `exec`）| 已有 modify-mode 案例 [npu_moe_token_permute_grad_v2-refactor](npu_moe_token_permute_grad_v2-refactor.md) |

## 与其他 m-adapt 案例的判别对照表

| 案例 | yaml 关键特征 | `opapi/` 文件 | **M-adapt verdict** |
|------|--------------|---------------|---------------------|
| `npu_iou-a5` | `op_api:` + 手写 cpp 已存在 | 有 `IouKernelNpuOpApi.cpp` | **State A: 已适配, 勿动** |
| `npu_ciou-a5` | `op_api:` + 手写 cpp 已存在 | 有 `CiouKernelNpuOpApi.cpp` | **State A: 已适配, 勿动** |
| `npu_swiglu-a5` (本例) | `op_api:` + **`gen_opapi:`** | **无手写文件** | **State B: 已适配(gen_opapi), 勿动** |
| `npu_mish_backward-a5` | 仅 `acl_op:`, 无 `op_api:` | 无 | **State C: 真候选 → 走 m-adapt** |
| `npu_moe_distribute_dispatch_v2-a5` | `op_api: all_version` + 手写 cpp | 有 | **State A: 已适配, 勿动**（但作为"做对的样子"参考）|
