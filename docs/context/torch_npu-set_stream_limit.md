# （beta）torch.npu.set_stream_limit
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

- 设置指定Stream的Device资源限制。
- 本接口应在调用`torch.npu.set_device_limit`接口之后且在算子执行之前调用，如果对同一stream进行多次设置，将以最后一次设置为准。
- 该接口设置完后，可以跨线程传递stream使用。

## 函数原型

```
torch.npu.set_stream_limit(stream, cube_num=-1, vector_num=-1) -> None
```

## 参数说明

- **stream** (`torch_npu.npu.Stream`)：必选参数，设置控核的流。
- **cube_num** (`int`)：可选参数，设置的cube的核数，默认为-1不设置分核。
- **vector_num** (`int`)：可选参数，设置的vector的核数，默认为-1不设置分核。

## 返回值说明
`None`

代表无返回值。

## 约束说明

- 该接口不支持多线程并发设置同一条流上的控核数，无法保证算子执行时的控核生效值。
- 该接口需要配合torch_npu.npu.config.allow_internal_format = False来使用（是否使用私有格式）。
- 主要使用场景为micro batch多流并行，存在部分老旧算子不支持控核，但不影响算子功能使用，可能会影响多流并行效果。

## 调用示例

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_stream_limit(torch.npu.current_stream(), 12, 24)
>>> torch.npu.set_stream_limit(torch.npu.Stream(), 13, 23)
 ```
