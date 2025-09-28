# torch.npu.set_stream_limit
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

- 该接口只支持使用Ascend C开发的算子控核，如遇到算子不支持控核，可以提Issue。
- 主要使用场景为micro batch多流并行，存在部分非Ascend C开发算子不支持控核，但不影响算子功能使用，可能会影响多流并行效果。
- 该接口需要配合torch_npu.npu.config.allow_internal_format = False来使用（不允许私有格式）。
- 该接口不支持多线程并发设置同一条流上的控核数，无法保证算子执行时的控核生效值。

## 调用示例

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_stream_limit(torch.npu.current_stream(), 12, 24)
>>> torch.npu.set_stream_limit(torch.npu.Stream(), 13, 23)
 ```

## 控核生效案例
1. 使用Ascend PyTorch Profiler接口采集性能数据，主要包括PyTorch层算子信息、CANN层算子信息、底层NPU算子信息以及算子内存占用信息等。
> **说明**：Ascend PyTorch Profiler是CANN针对PyTorch框架开发的性能分析工具，通过在PyTorch脚本中添加Ascend PyTorch Profiler接口（推荐torch_npu.profiler.profile接口）采集指定指标数据，模型执行时同步采集性能数据，详细的使用方法和结果文件介绍请参考[《CANN 性能调优工具用户指南》](https://hiascend.com/document/redirect/CanncommercialToolProfiling)中的“Ascend PyTorch Profiler”章节。
 ```python
>>> import torch
>>> import torch_npu
>>> stream1 = torch.npu.current_stream()
>>> stream2 = torch.npu.Stream()
>>> x1 = torch.randn(1024, 1960).npu()
>>> experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
>>> with torch_npu.profiler.profile(
>>>    with_stack=False,                # 采集算子的函数调用栈开关，默认关闭
>>>    record_shapes=False,             # 采集算子的input shape和input type开关，默认关闭
>>>    profile_memory=False,            # 采集memory相关数据开关，默认关闭
>>>    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1),           # warmup默认为0，老版本torch_npu包该参数为必填项
>>>    experimental_config=experimental_config,                                   
>>>    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir") # 导出tensorboard可视化数据
>>>    ) as prof:
>>>    torch.npu.set_stream_limit(stream1, 12, 22)
>>>    torch.npu.set_stream_limit(stream2, 13, 23)
>>>    output = torch_npu.npu_swiglu(x1, dim=-1)
>>>    with torch.npu.stream(stream2):
>>>       output = torch_npu.npu_swiglu(x1, dim=-1)
 ```
2. 查看采集结果文件（json、csv等格式）。 

单算子控核可以查看**kernel_details.csv**文件，查看算子的**BlockDim**列（算子计算所用核数），Blockdim小于等于用户设置的控核数即可认为控核成功。
这里的SwiGlu算子的Blockdim分别为22和23，符合控核逻辑。