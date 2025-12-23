# torch_npu.npu.pin_memory_stats

## 产品支持情况
不涉及

## 功能说明

查询pin_memory内存使用统计，单位：byte。

## 定义文件
torch_npu/npu/memory.py

## 函数原型

```
torch_npu.npu.pin_memory_stats() -> dict
```

## 参数说明
无

## 返回值说明
pin_memory内存使用统计，包含`已申请的物理内存`和`正在使用的物理内存`的`当前值`和`峰值`，包含以下key，单位：byte。

| key                     |        描述         |
|-------------------------|:-----------------:|
| allocated_bytes.current |    当前正在使用的物理内存    |
| allocated_bytes.peak    | 进程启动以来使用的物理内存总量峰值 |
| reserved_bytes.current  |   当前已申请的物理内存总量    |
| reserved_bytes.peak     | 进程启动以来申请的物理内存总量峰值 |


## 约束说明
无


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty([1024, 1024]).pin_memory()
>>> print(torch_npu.npu.pin_memory_stats())
```