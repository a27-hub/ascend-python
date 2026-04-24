# （beta）at_npu::native::atb::npu_fused_add_topk_div

## 定义文件

op_plugin/include/atb_ops.h

## 函数原型

```cpp
std::tuple<at::Tensor, at::Tensor> at_npu::native::atb::npu_fused_add_topk_div(
    const at::Tensor& x, const at::Tensor& add_num,
    const c10::optional<at::Tensor>& mapping_num,
    const c10::optional<at::Tensor>& mapping_table,
    c10::optional<c10::string_view> activation_type_opt, int64_t group_num,
    int64_t group_topk, int64_t n, int64_t k, bool is_norm, double scale,
    bool enable_expert_mapping)

std::tuple<at::Tensor&, at::Tensor&>
at_npu::native::atb::npu_fused_add_topk_div_out(
    const at::Tensor& x, const at::Tensor& add_num,
    const c10::optional<at::Tensor>& mapping_num,
    const c10::optional<at::Tensor>& mapping_table,
    c10::optional<c10::string_view> activation_type_opt, int64_t group_num,
    int64_t group_topk, int64_t n, int64_t k, bool is_norm, double scale,
    bool enable_expert_mapping, at::Tensor& y, at::Tensor& indices)
```

## 功能说明

融合路由权重分配和专家选择等过程，主要用于 DeepSeek MoE 场景。该接口族覆盖基础输出和 `out` 变体。

## 参数说明

x：Tensor 类型，输入张量。  
add_num：Tensor 类型，与 `x` 相加的偏置张量。  
mapping_num：可选 Tensor 类型，逻辑专家模式下的映射数量输入。  
mapping_table：可选 Tensor 类型，逻辑专家模式下的映射表输入。  
activation_type_opt：可选字符串枚举参数。  
group_num：int64_t 类型，分组数量。  
group_topk：int64_t 类型，组级 topk 数量。  
n：int64_t 类型，组内选取数量。  
k：int64_t 类型，专家级 topk 数量。  
is_norm：bool 类型，是否归一化。  
scale：double 类型，归一化后的缩放系数。  
enable_expert_mapping：bool 类型，是否使能物理专家到逻辑专家的映射。  
y：Tensor 类型，`out` 变体的输出张量。  
indices：Tensor 类型，`out` 变体的索引输出张量。

## 支持的型号?

- <term>Atlas 800I A2 推理产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
