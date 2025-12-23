# torch_npu.npu.pin_memory_reserved

## 产品支持情况
不涉及

## 功能说明

查询已申请的pin_memory物理内存总量，单位：byte。

## 定义文件
torch_npu/npu/memory.py

## 函数原型

```
torch_npu.npu.pin_memory_reserved() -> int
```

## 参数说明
无

## 返回值说明
已申请的pin_memory物理内存总量，单位：byte。

## 约束说明
无


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty([1024, 1024]).pin_memory()
>>> print(torch_npu.npu.pin_memory_reserved())
```