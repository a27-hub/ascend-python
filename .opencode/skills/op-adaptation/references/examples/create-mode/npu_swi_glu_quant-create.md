# npu_swi_glu_quant-create

> ⚠️ **版本说明（接入前必读）**：本案例记录 **V1 接入**——aclnn 算子 `aclnnSwiGluQuant`、`str quant_mode` 接口、算子名 `npu_swi_glu_quant`（带下划线）、文件 `SwiGluQuantOpApi.cpp`。**当前真实库已演进至 V2**：算子名 `npu_swiglu_quant`（无下划线，全库唯一）、`int quant_mode`、`aclnnSwiGluQuantV2`、新增 `group_list_type`/`dst_type`（见 `op_plugin/config/op_plugin_functions.yaml` 的 `npu_swiglu_quant` 条目）。接入前请以真实库为准对齐算子名与签名。
>
> 案例 C++ 代码与 Pitfall 反映 V1 实现细节。其中 string_view→char* 传递、SwiGLU A/B 顺序、scale 语义、group_index 累计边界、smooth_scales 设备污染等 Pitfall 与版本无关，可作通用教学参考。

## Background

接入 CANN 算子 aclnnSwiGluQuant 到 torch_npu 后端。该算子实现 SwiGlu + 量化融合，支持动态量化和静态量化，仅 MoE 场景。全新接入（Create 模式），V1 only。

## Solution

按 Create 模式完整流程执行 13 个步骤。关键决策：

- **_def.cpp 有 V2 属性（group_list_type, dst_type, INT4输出），用户确认仅实现 V1**，因此 V2 属性排除，y 输出定为 INT8
- **无 autograd**：跳过 C-derivatives
- **YAML `exposed: all_version`**：公开算子，非内部接口

## File Changes

| File | Action | Details |
|------|--------|---------|
| `op_plugin/config/op_plugin_functions.yaml` | add | `npu_swi_glu_quant(Tensor x, *, Tensor? smooth_scales=None, Tensor? offsets=None, Tensor? group_index=None, bool activate_left=False, str quant_mode="dynamic") -> (Tensor, Tensor)` |
| `op_plugin/ops/opapi/SwiGluQuantOpApi.cpp` | add | 4 个 TORCH_CHECK，多输出 shape 计算，EXEC_NPU_CMD 调用 |
| `op_plugin/python/meta/_meta_registrations.py` | add | `npu_swi_glu_quant_meta`: y 最后一维 / 2，scale 去最后一维 |
| `test/test_custom_ops/test_npu_swi_glu_quant.py` | add | 13 用例：动态/静态/分组/dtype/shape/异常 |
| `test/core_tests/test_fake_tensor.py` | add | `TestNPUSwiGluQuant`: 2D/3D 动态+静态 shape/dtype 验证 |
| `test/core_tests/torch_npu_OpApi_schema_all.json` | add | op_api (untyped/typed) + func 三条目 |
| `test/allowlist_for_publicAPI.json` | add | `npu_swi_glu_quant` → `all_version` |
| `docs/zh/custom_APIs/torch_npu/torch_npu-npu_swi_glu_quant.md` | add | 完整 API 文档（产品支持/公式/参数/约束/示例） |
| `docs/zh/custom_APIs/torch_npu/torch_npu_list.md` | add | 导航条目 |
| `codegen/templates/_op_plugin_docs.py` | add | `_add_torch_npu_docstr` docstring 条目 |

**关键 C++ 代码**:

```cpp
std::tuple<at::Tensor, at::Tensor> npu_swi_glu_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& offsets,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    c10::string_view quant_mode)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.size(x.dim() - 1) % SWISH_NUM == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.size(x.dim() - 1) <= MAX_LAST_DIM, "x last dim should not exceed 8192", ...);
    TORCH_CHECK(quant_mode == "dynamic" || quant_mode == "static", ...);

    const at::Tensor& smooth_scales_opt = c10::value_or_else(smooth_scales, [] { return at::Tensor(); });
    const at::Tensor& offsets_opt = c10::value_or_else(offsets, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    std::string quant_mode_str(quant_mode);
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    at::SmallVector<int64_t, op_infer::SIZE> y_size, scale_size;
    for (int i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    y_size.push_back(x.size(x.dim() - 1) / SWISH_NUM);

    at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(aclnnSwiGluQuant, x, smooth_scales_opt, offsets_opt, group_index_opt,
                 activate_left, quant_mode_ptr, y, scale);

    return std::tie(y, scale);
}
```

## Pitfalls

1. **c10::string_view 直接取 .data() 风险**: `DequantSwigluQuantOpApi.cpp` 用的 `std::string` → `.c_str()`，本项目先用了 `c10::string_view::data()`。虽然后续验证两者都 work，但 `std::string` 模式更安全（保证 null-terminated），最终采用了 `std::string` 方式。

2. **参考实现 SwiGLU 的 A/B 顺序反了**: `activate_left=False` 时应激活**后半**（swish on second half），参考实现错误地激活了前半。通过对比 `dequant_swiglu_quant` 参考发现并修复。

3. **scale 语义**: 动态量化时 NPU 返回的是量化 scale（127/max_abs），不是反量化 scale（max_abs/127）。参考实现最初用了反量化公式导致 scale 校验失败。

4. **group_index 是累计边界**: `group_index=[2, 4]` 表示两个组：rows 0:2 和 rows 2:4，不是每组 2 个和 4 个元素。参考实现最初当作单组长度使用导致分组测试失败。

5. **smooth_scales 不支持空指针导致设备污染**: CANN 8.5.0 版本的 aclnnSwiGluQuant 不支持 smooth_scales 传入 null。首次调用不传 smooth_scales 会触发 kernel 异常 (507057)，设备流进入错误状态，后续所有操作（包括 `.npu()` 拷贝）全部失败。设备约 30 秒后自动恢复。所有测试用例必须显式传入 smooth_scales。
