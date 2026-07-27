# npu_iou A5 平台适配（M-adapt 案例）

> 本案例演示 M-adapt 模式中"aclnn 输出 shape 与用户接口不一致"的典型场景：需要 `transpose(0, 1).contiguous()` 后处理。
> 来源：master commit `1f356ca8`（PR !4910）。

## Background

`npu_iou` 算子用于计算预测检测框与真实检测框之间的交并比（IoU）或前景交叉比（IoF）。原有实现基于 ACL 接口，**内部将输入强制转换为 fp16** 进行计算。Atlas A5 训练系列产品（Ascend950）需要通过 ACLNN 接口 `aclnnIou` 重新对接，**原生支持 fp32/fp16，精度更高**。

**关键点**：
- 算子签名零变化（`npu_iou(bboxes, gtboxes, mode=0)` 不变）
- 老平台保留 ACL 路径（强制转 fp16，精度有损）
- A5 平台走新的 ACLNN 路径（原生 fp32/fp16，精度无损）
- **aclnnIou 输出 shape `(n, m)`，用户接口期望 `(m, n)`** — 需要 transpose
- 算子无 backward（不需要硬报错处理）

## Solution

采用 M-adapt 模式的标准三段式骨架，**重点关注段 3.2 输出转置**：

1. **段 1（固定 SOC 路由）**：`GetSocVersion() < Ascend950` → 老路径
2. **段 2（适配段）**：aclnnIou 原生支持 fp32/fp16，**无需 dtype 转换**；直接预分配 → ACLNN 调用
3. **段 3（特殊处理）**：输出转置 `transpose(0, 1).contiguous()`（核心差异点）

## File Changes

| File | Action | Details |
|------|--------|---------|
| `op_plugin/ops/opapi/IouKernelNpuOpApi.cpp` | 新增 | ACLNN 对接实现，含 SOC 版本路由、输出转置 |
| `op_plugin/python/meta/_meta_registrations.py` | 新增 | 注册 `npu_iou` 的 Meta 实现（注意 shape 是用户接口的 `(m, n)`） |
| `op_plugin/config/op_plugin_functions.yaml` | 修改 | 为 `npu_iou` 添加 `op_api: all_version` |
| `test/test_base_ops/test_iou.py` | 修改 | 新增 Ascend950 专属用例，添加 `@SupportedDevices` 装饰器 |
| `docs/zh/custom_APIs/torch_npu/（beta）torch_npu-npu_iou.md` | 修改 | 新增 Atlas A5 产品支持 |

### YAML 改动

```yaml
- func: npu_iou(Tensor bboxes, Tensor gtboxes, int mode=0) -> Tensor
  acl_op: all_version
  op_api: all_version       # 新增
  exposed: all_version

# ❌ npu_iou 无 backward，不需要反向 entry
```

### C++ 关键代码（完整 OpApi 文件仅 26 行）

```cpp
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_iou(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    int64_t mode)
{
    // 段 1：SOC 路由
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_iou(bboxes, gtboxes, mode);
    }

    // 段 2：ACLNN 路径（aclnnIou 原生支持 fp32/fp16，无需 dtype 转换）
    // aclnnIou 输出 shape 为 (bboxes.size(0), gtboxes.size(0)) = (n, m)
    c10::SmallVector<int64_t, SIZE> output_size = {bboxes.size(0), gtboxes.size(0)};
    at::Tensor overlap = npu_preparation::apply_tensor_with_format(
        output_size, bboxes.options(), ACL_FORMAT_ND);

    const char *mode_str = mode == 1 ? "iof" : "iou";
    float eps = 0.01;       // aclnn 新增参数（aclnnIou 比 ACL Iou 多 eps 和 aligned）
    bool aligned = false;   // aclnn 新增参数
    EXEC_NPU_CMD(aclnnIou, bboxes, gtboxes, mode_str, eps, aligned, overlap);

    // 段 3.2：输出转置（aclnnIou 输出 (n, m)，用户接口期望 (m, n)）
    return overlap.transpose(0, 1).contiguous();
}

}  // namespace op_api
```

### Meta 注册（注意 shape 是用户接口的最终 shape）

```python
@impl(m, "npu_iou")
def npu_iou_meta(bboxes, gtboxes, mode=0):
    # ⚠️ 必须按用户接口最终 shape 写: (m, n)，不是 aclnn 的 (n, m)
    output_size = [gtboxes.size(0), bboxes.size(0)]
    return torch.empty(output_size, dtype=bboxes.dtype, device=bboxes.device)
```

### 测试设计要点

- `@SupportedDevices(['Ascend910A', 'Ascend910B'])`：910A/910B 旧用例（验证 ACL 路径）
- `@SupportedDevices(['Ascend950'])`：950 专属新用例（验证 ACLNN 路径 + transpose 正确性）
- 形状覆盖：对称 (3×3)、非对称 (6×8)、大规模 (200×100)
- 精度断言：fp16 用例用硬编码期望值 + `assertRtolEqual`；fp32 用例仅做形状断言

## Pitfalls (AI-generated summaries)

- **Meta 函数体中 output_size 写错**（**最易踩坑**）：写成 `[bboxes.size(0), gtboxes.size(0)]`（aclnn 的输出 shape）而非 `[gtboxes.size(0), bboxes.size(0)]`（用户接口的最终 shape）。结果：`torch.compile` 模式下 `npu_iou` 输出维度错乱。
- **忘记 `.contiguous()`**：只调用 `.transpose(0, 1)` 返回非连续内存视图，下游 `view`/`reshape` 操作会失败。
- **fp32 精度参考值用 ACL 路径验证**：ACL 路径强制转 fp16 才有参考值，fp32 在 910A/910B 走的也是 fp16 强转，**不能**作为 A5 fp32 路径的参考。
- **aclnnIou 新增参数未指定**：`eps` 和 `aligned` 是 aclnnIou 相比 ACL Iou 新增的参数，缺一个都会编译失败或运行时报错。
- **删除 `is_cross` 模式兼容测试**：npu_iou 本来就没有 `is_cross` 参数，无需特别处理（与 ciou 不同）。

## A5 Platform Specific Info (M-adapt 模式专用)

### A5 适配基本信息

| 维度 | 值 |
|------|---|
| 目标 SoC | Ascend950 (Atlas A5 训练系列) |
| 对接 aclnn 算子 | `aclnnIou` |
| 路由方式 | `GetSocVersion()` |
| 关联 PR | https://gitcode.com/Ascend/op-plugin/pull/4910 |
| 关联设计文档 | https://gitcode.com/Ascend/op-plugin/issues/86 |

### A5 平台特殊处理

| 维度 | 描述 | 是否涉及 |
|------|------|----------|
| dtype 转换（fp16→fp32→fp16） | aclnnIou 原生支持 fp32/fp16 | **否**（相比 ciou 是关键差异） |
| 输出转置 | aclnnIou 输出 (n, m)，用户接口期望 (m, n) | **是**（核心差异点） |
| 反向不支持 | npu_iou 本来无 backward | 否（无 backward） |
| 多输出内部接口 | 无 | 否（单输出） |
| aclnn 新增参数 | `eps=0.01`、`aligned=false` | **是** |
| A5 平台特殊约束 | 无特殊输入规模约束 | 否 |

### A5 测试用例设计要点

- 平台隔离：`@SupportedDevices(['Ascend950'])` vs `@SupportedDevices(['Ascend910A', 'Ascend910B'])`
- 输入规模：无特殊约束（覆盖 2×1 / 3×3 / 6×8 / 200×100 等）
- 精度断言：fp16 用例用硬编码期望值 + `assertRtolEqual`；fp32 用例仅做形状断言
- 输出转置验证：所有 950 用例的形状断言**间接验证** transpose(0, 1) 正确性

## 适用范围与互补案例

**本案例覆盖的"窄"模式**：
- 单一 aclnn 算子（`aclnnIou`），**无版本变体**（V2/V3/V4…）
- aclnn **确定已在 A5 可用**（无需运行时探测）
- aclnn 输出 shape 与用户接口差异→**单一** transpose 处理
- aclnn 新增参数有限（`eps`、`aligned`），无复杂 enum/特性 gate

**本案例不覆盖、需配合其他 case 看的情况**：
| 场景 | 推荐看 |
|------|--------|
| acl_op-only、需补 op_api（含端到端整编/A5 验证）| [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| aclnn **多版本变体**(V2..VN)、需 `check_aclnn_kernel_available` 回退 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) |
| 算子**已通过 gen_opapi 适配**（不应触发 M-adapt）| [npu_swiglu-a5](npu_swiglu-a5.md) |
| CANN 版本覆盖度校验/缺 kernel 时的优雅报错 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) + `m-adapt-guide.md` §4 |
| 编译期 cgroup/MAX_JOBS 坑 | [npu_mish_backward-a5](npu_mish_backward-a5.md) + `build-guide.md` 顶部 |
