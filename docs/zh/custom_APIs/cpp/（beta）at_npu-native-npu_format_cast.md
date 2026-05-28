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

customize_dtype：可选参数，指定格式转换时的数据类型，传入PyTorch数据类型对应的整数值。未指定时，`float32`、`int32`默认C0为16，`int8`默认C0为32；当且仅当传入`int(torch.int32)`（即`3`）时，FRACTAL_NZ格式的C0设置为8。

## 约束说明

- **customize_dtype**仅支持<term>Atlas A2 训练系列产品</term>910B/910C且CANN >= 9.1.0的场景，其他产品或CANN版本传入该参数时会抛出异常。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
