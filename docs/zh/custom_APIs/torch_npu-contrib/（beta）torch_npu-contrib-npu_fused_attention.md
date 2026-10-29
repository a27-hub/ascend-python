# （beta）torch_npu.contrib.npu_fused_attention

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

BERT自注意力的融合实现。

## 函数原型

```python
torch_npu.contrib.npu_fused_attention(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, scale=1, keep_prob=0)
```

## 参数说明

- **hidden_states** (`Tensor`)：必选参数，最后一层的hidden_states。数据类型支持`torch.float16`。数据格式支持FRACTAL_NZ。shape为2维，其中第0维需为32的整数倍，第1维支持768或1024。
- **attention_mask** (`Tensor`)：必选参数，掩码张量，用于屏蔽不需要参与注意力计算的token位置（如padding位置）。数据类型支持`torch.float16`。数据格式支持FRACTAL_NZ。shape为4维，形状为$(B, 1, S, S)$，其中B为batch size，S为序列长度。
- **query_kernel** (`Tensor`)：必选参数，query的权重矩阵。数据类型支持`torch.float16`。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **key_kernel** (`Tensor`)：必选参数，key的权重矩阵。数据类型支持`torch.float16`。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **value_kernel** (`Tensor`)：必选参数，value的权重矩阵。数据类型支持`torch.float16`。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **query_bias** (`Tensor`)：必选参数，query的偏差值。数据类型支持`torch.float16`。数据格式支持ND。
- **key_bias** (`Tensor`)：必选参数，key的偏差值。数据类型支持`torch.float16`。数据格式支持ND。
- **value_bias** (`Tensor`)：必选参数，value的偏差值。数据类型支持`torch.float16`。数据格式支持ND。
- **scale** (`double`)：可选参数，计算score的缩放系数，默认值为1。
- **keep_prob** (`float`)：可选参数，计算中保留数据的概率，值等于1 - drop rate，默认值为0。

## 返回值说明

`Tensor`

self attention的结果。

## 约束说明

- 数据格式要求：`hidden_states`、`attention_mask`、`query_kernel`、`key_kernel`、`value_kernel` 必须为FRACTAL_NZ（格式编号29），`query_bias`、`key_bias`、`value_bias` 必须为ND（格式编号2）。
- 在Atlas A2/A3训练系列产品上，需先设置`torch_npu.npu.config.allow_internal_format = True`，`npu_format_cast`才能转为FRACTAL_NZ；否则FRACTAL_NZ会被静默降级为ND，触发格式校验失败。
- 该API为BERT自注意力的融合实现，底层算子针对BERT标准配置分片。序列长度S（`attention_mask`的第2、3维）需满足算子约束，官方示例与测试均使用S=512，较小的S（如S=32）会触发分片失败。
- `scale`与`keep_prob`需以位置参数传入，使用关键字参数（如`scale=0.125, keep_prob=0.5`）会触发`TypeError: apply() takes no keyword arguments`。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.config.allow_internal_format = True
>>> hidden_states = torch_npu.npu_format_cast(torch.rand(12288, 1024).npu().half(), 29)
>>> attention_mask = torch_npu.npu_format_cast(torch.rand(24, 1, 512, 512).npu().half(), 29)
>>> query_kernel = torch_npu.npu_format_cast(torch.rand(1024, 1024).npu().half(), 29)
>>> key_kernel = torch_npu.npu_format_cast(torch.rand(1024, 1024).npu().half(), 29)
>>> value_kernel = torch_npu.npu_format_cast(torch.rand(1024, 1024).npu().half(), 29)
>>> query_bias = torch_npu.npu_format_cast(torch.rand(1024).npu().half(), 2)
>>> key_bias = torch_npu.npu_format_cast(torch.rand(1024).npu().half(), 2)
>>> value_bias = torch_npu.npu_format_cast(torch.rand(1024).npu().half(), 2)
>>> context_layer = torch_npu.contrib.npu_fused_attention(
...     hidden_states, attention_mask, query_kernel, key_kernel, value_kernel,
...     query_bias, key_bias, value_bias, 0.125, 0.5)
>>> print(context_layer.shape)
```
