# npu_ciou A5 平台适配（M-adapt 案例）

> 本案例演示 M-adapt 模式的完整执行流程：已有 `npu_ciou` 算子（已 A2/A3 适配），新增 Atlas A5 平台（Ascend950）ACLNN 路径支持。
> 来源：master commit `d91c261b`（PR !4909）。

## Background

`npu_ciou` 算子是目标检测领域常用的 CIoU（Complete Intersection over Union）损失函数的 NPU 实现。原有实现基于 ACL（Ascend Computing Language）接口。Atlas A5 训练系列产品（Ascend950）需要通过 ACLNN（Ascend Computing Language Neural Network）接口 `aclnnCIoU` 重新对接，以获得更好的性能和精度支持。

**关键点**：
- 算子签名零变化（PyTorch 接口 `npu_ciou(...)` 调用方式不变）
- 老平台（A2/A3/310P/910）保留原 ACL 路径
- A5 平台走新的 ACLNN 路径
- 运行时通过 `c10_npu::GetSocVersion()` 路由
- A5 平台**不支持反向** `npu_ciou_backward`

## Solution

采用 M-adapt 模式的标准三段式骨架：

1. **段 1（固定 SOC 路由）**：`GetSocVersion() < Ascend950` → 老路径直接转发
2. **段 2（适配段）**：dtype 转换（fp16→fp32）→ 输出尺寸推导 → 预分配 → ACLNN 调用
3. **段 3（特殊处理）**：dtype 回转（fp32→fp16）→ 返回结果

反向 `npu_ciou_backward` 走硬报错模板：`TORCH_CHECK(false, "...NOT_SUPPORT", OPS_ERROR(ErrCode::NOT_SUPPORT))`

## File Changes

| File | Action | Details |
|------|--------|---------|
| `op_plugin/ops/opapi/CiouKernelNpuOpApi.cpp` | 新增 | ACLNN 对接实现，含 SOC 版本路由、fp16 类型转换、反向硬报错 |
| `op_plugin/python/meta/_meta_registrations.py` | 新增 | 注册 `_npu_ciou` 和 `npu_ciou` 的 Meta 实现 |
| `op_plugin/config/op_plugin_functions.yaml` | 修改 | 为 `_npu_ciou`、`npu_ciou`、`npu_ciou_backward` 添加 `op_api: all_version` |
| `test/test_base_ops/test_ciou.py` | 修改 | 新增 Ascend950 专属用例，添加 `@SupportedDevices` 装饰器，删除 `is_cross=True` 用例 |
| `docs/zh/custom_APIs/torch_npu/（beta）torch_npu-npu_ciou.md` | 修改 | 更新产品支持、默认值、约束说明 |

### YAML 改动

```yaml
# _npu_ciou（内部接口）
- func: _npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> (Tensor, Tensor)
  acl_op: all_version
  op_api: all_version       # 新增

# npu_ciou（用户接口）
- func: npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> Tensor
  acl_op: all_version
  op_api: all_version       # 新增
  exposed: all_version

# npu_ciou_backward（反向，场景 2 硬报错）
- func: npu_ciou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, Tensor? atan_sub, bool trans=False, bool is_cross=True, int mode=0) -> (Tensor, Tensor)
  acl_op: all_version
  op_api: all_version       # 新增（让 C++ 硬报错模板生效）
```

### C++ 关键代码（节选）

```cpp
// 段 1：SOC 路由
if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
    return acl_op::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}

// 段 2.1：dtype 转换（fp16 → fp32）
bool self_is_half = self.scalar_type() == at::kHalf;
bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
at::Tensor self_cp = self_is_half
    ? at_npu::native::custom_ops::_npu_dtype_cast(self, at::kFloat) : self;
at::Tensor gtboxes_cp = gtboxes_is_half
    ? at_npu::native::custom_ops::_npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

// 段 2.2-2.4：输出尺寸推导 + 预分配 + ACLNN 调用
auto output_size = ciou_npu_output_size(self_cp, gtboxes_cp, is_cross);
at::Tensor overlap = npu_preparation::apply_tensor_with_format(
    output_size, self_cp.options(), ACL_FORMAT_ND);
at::Tensor atan_sub = npu_preparation::apply_tensor_with_format(
    output_size, self_cp.options(), ACL_FORMAT_ND);

const char *mode_str = mode == 1 ? "iof" : "iou";
EXEC_NPU_CMD(aclnnCIoU, self_cp, gtboxes_cp, trans, is_cross, mode_str, overlap, atan_sub);

// 段 3.1：dtype 回转
if (self_is_half || gtboxes_is_half) {
    overlap = at_npu::native::custom_ops::_npu_dtype_cast(overlap, at::kHalf);
    atan_sub = at_npu::native::custom_ops::_npu_dtype_cast(atan_sub, at::kHalf);
}
return std::tie(overlap, atan_sub);
```

### Meta 注册

```python
@impl(m, "_npu_ciou")
def _npu_ciou_meta(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False):
    if is_cross:
        output_size = [gtboxes.size(1), self.size(1)]
    else:
        output_size = [1, self.size(1)]
    return (torch.empty(output_size, dtype=self.dtype, device=self.device),
            torch.empty(output_size, dtype=self.dtype, device=self.device))


@impl(m, "npu_ciou")
def npu_ciou_meta(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False):
    if is_cross:
        output_size = [gtboxes.size(1), self.size(1)]
    else:
        output_size = [1, self.size(1)]
    return torch.empty(output_size, dtype=self.dtype, device=self.device)

# ❌ npu_ciou_backward 不注册 Meta
```

### 测试设计要点

- `@SupportedDevices(['Ascend910A', 'Ascend910B'])`：910A/910B 旧用例
- `@SupportedDevices(['Ascend950'])`：950 专属新用例（输入规模 1024/2048/3072）
- 输入约束：950 平台 boxes 第二维度必须为 1024 倍数

## Pitfalls (AI-generated summaries)

- **dtype 转换函数选错**：用旧版 `npu_dtype_cast(t, int64_t)` 而非 `at_npu::native::custom_ops::_npu_dtype_cast(t, at::kFloat)`（接 ScalarType）。后者与 ACL 实现保持一致。
- **反向忘记加 `op_api: all_version`**：只写 `acl_op: all_version` 会导致 A5 平台调用 backward 时不走 OpApi 路径，硬报错模板不生效。
- **Meta 函数体中 output_size 写错**：必须按用户接口最终 shape（不是 aclnn 输出 shape）写。`npu_ciou` 用户接口返回 `(1, n)` 或 `(gtboxes.size(1), self.size(1))`，与 aclnn 一致；但其他算子可能有差异。
- **A5 输入规模违反约束**：n=100（非 1024 倍数）会在 A5 平台触发底层报错，测试用例应只覆盖 1024/2048/3072。
- **忘记删除 `is_cross=True` 用例**：aclnnCIoU 不支持 is_cross=True，必须删除或重定向。

## A5 Platform Specific Info (M-adapt 模式专用)

### A5 适配基本信息

| 维度 | 值 |
|------|---|
| 目标 SoC | Ascend950 (Atlas A5 训练系列) |
| 对接 aclnn 算子 | `aclnnCIoU` |
| 路由方式 | `GetSocVersion()` |
| 关联 PR | https://gitcode.com/Ascend/op-plugin/pull/4909 |
| 关联设计文档 | https://gitcode.com/Ascend/op-plugin/issues/83 |

### A5 平台特殊处理

| 维度 | 描述 | 是否涉及 |
|------|------|----------|
| dtype 转换（fp16→fp32→fp16） | aclnnCIoU 不支持 fp16 输入 | **是** |
| 输出转置 | aclnn 输出 shape 与用户接口一致 | 否 |
| 反向不支持 | A5 平台 aclnn 不支持 backward | **是，硬报错** |
| 多输出内部接口 | `_npu_ciou` 返回 `(overlap, atan_sub)` 元组 | **是** |
| aclnn 新增参数 | `mode_str`（"iou" / "iof" 字符串） | 是（枚举→字符串） |
| A5 平台特殊约束 | boxes 第二维度必须 1024 倍数 | **是** |

### A5 测试用例设计要点

- 平台隔离：`@SupportedDevices(['Ascend950'])` vs `@SupportedDevices(['Ascend910A', 'Ascend910B'])`
- 输入规模约束：950 用例必须用 1024/2048/3072
- 精度断言：fp16 场景只做形状断言（精度有限）；fp32 + iou + atan_sub_flag=True 用 `cpu_op_exec` 做精度对比
- 错误用例：`test_npu_ciou_invalid_shape` 用 n=100 验证 A5 平台应抛 RuntimeError
- 删除用例：所有 `is_cross=True` 用例需删除（aclnnCIoU 不支持）

## 适用范围与互补案例

**本案例覆盖的"窄"模式**：
- 单一 aclnn 算子（`aclnnCIoU`），**无版本变体**
- aclnn **不支持的 dtype** → fp16→fp32 cast + 回 fp16
- aclnn 反向不支持 → 硬报错模板
- 多输出内部接口（`_npu_ciou` 返回 tuple）
- 简单 enum→string 映射（`mode` int → `"iou"`/`"iof"`）

**本案例不覆盖、需配合其他 case 看的情况**：
| 场景 | 推荐看 |
|------|--------|
| acl_op-only、需补 op_api（含端到端整编/A5 验证）| [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| aclnn **多版本变体**(V2..VN)、需 `check_aclnn_kernel_available` 回退 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) |
| 算子**已通过 gen_opapi 适配**（不应触发 M-adapt）| [npu_swiglu-a5](npu_swiglu-a5.md) |
| 按入参特性 gate（`sink.has_value` 决定走 V5/V3/V2）| 参考 `FlashAttentionV2KernelNpuOpApi.cpp` 真实代码 + `m-adapt-cpp-patterns-guide.md` §4 |
| 编译期 cgroup/MAX_JOBS 坑 | [npu_mish_backward-a5](npu_mish_backward-a5.md) + `build-guide.md` 顶部 |
