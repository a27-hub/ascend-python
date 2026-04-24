# （beta）at_npu::native::atb::npu_fused_add_topk_div

## 产品支持情况

当前为 beta 接口。已知支持 Atlas 800I A2 推理产品、Atlas A2 训练系列产品和 Atlas A3 训练系列产品。其他产品形态请以当前 NNAL/ATB 版本支持情况为准。

## 功能说明

融合路由权重分配、专家选择等过程，主要用于 DeepSeek MoE 场景。内部融合了 Sigmoid、Add、GroupTopk、Gather、ReduceSum、RealDiv 和 Muls 等小算子。

当前文档覆盖的 C++ wrapper 变体包括：

- `at_npu::native::atb::npu_fused_add_topk_div`
- `at_npu::native::atb::npu_fused_add_topk_div_out`

## 定义文件

`op_plugin/include/atb_ops.h`

## 函数原型

```cpp
std::tuple<at::Tensor, at::Tensor> at_npu::native::atb::npu_fused_add_topk_div(
    const at::Tensor& x, const at::Tensor& add_num,
    const c10::optional<at::Tensor>& mapping_num,
    const c10::optional<at::Tensor>& mapping_table,
    c10::optional<c10::string_view> activation_type_opt, int64_t group_num,
    int64_t group_topk, int64_t n, int64_t k, bool is_norm, double scale,
    bool enable_expert_mapping);

std::tuple<at::Tensor&, at::Tensor&>
at_npu::native::atb::npu_fused_add_topk_div_out(
    const at::Tensor& x, const at::Tensor& add_num,
    const c10::optional<at::Tensor>& mapping_num,
    const c10::optional<at::Tensor>& mapping_table,
    c10::optional<c10::string_view> activation_type_opt, int64_t group_num,
    int64_t group_topk, int64_t n, int64_t k, bool is_norm, double scale,
    bool enable_expert_mapping, at::Tensor& y, at::Tensor& indices);
```

## 参数说明

`x`：输入张量。  
`add_num`：与 `x` 相加的偏置张量。  
`mapping_num`、`mapping_table`：逻辑专家模式下的可选映射输入。  
`activation_type_opt`：可选激活类型，当前通常使用 `activation_sigmoid`。  
`group_num`：分组数量。  
`group_topk`：组级 topk 数量。  
`n`：组内选取 `n` 个值求和。  
`k`：专家级 topk 数量。  
`is_norm`：是否归一化。  
`scale`：归一化后的缩放系数。  
`enable_expert_mapping`：是否使能物理专家到逻辑专家的映射。  

变体差异：

- `npu_fused_add_topk_div` 返回 `(y, indices)`。
- `npu_fused_add_topk_div_out` 额外接收 `y` 和 `indices` 作为输出张量。

## 返回值说明

- 基础变体返回 `(y, indices)`。
- `*_out` 变体返回写回后的 `(y, indices)` 张量引用。

## 约束说明

- `b` 必须为 `group_num` 的整数倍。
- `group_topk <= group_num`。
- `k <= b`。
- `b >= group_num * n`。
- `b <= group_num * 32`。
- 若 `b >= 32`，则 `group_num = 8`。
- `mapping_num` 和 `mapping_table` 的取值范围请与当前 NNAL/ATB 版本保持一致。
