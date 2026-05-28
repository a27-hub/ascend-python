# （beta）at_npu::native::npu_format_cast

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```cpp
at::Tensor at_npu::native::npu_format_cast(const at::Tensor& self, int64_t acl_format, c10::optional<int64_t> customize_dtype = c10::nullopt)
```

## 功能说明

NPU tensor格式转换，返回值类型Tensor，表示转换后的tensor。

## 参数说明

self：Tensor类型，待转换格式的tensor。

acl_format：int64_t型，待转换的格式。

customize_dtype：可选参数，int64_t型，指定格式转换时使用的数据类型（aclDataType枚举值），用于控制FRACTAL_NZ格式的C0对齐值。当且仅当`customize_dtype=3`（对应`ACL_INT32`）时，C0设置为8；其他情况下C0保持默认值。仅<term>Atlas A2 训练系列产品</term>上CANN >= 9.1.0时支持。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
