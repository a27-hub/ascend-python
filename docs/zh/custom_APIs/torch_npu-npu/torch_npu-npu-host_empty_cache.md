# torch_npu.npu.host_empty_cache

## 产品支持情况
不涉及

## 功能说明

释放当前无引用的`pin_memory`物理内存，如环境变量`PYTORCH_NPU_ALLOC_CONF`的值设置为`pin_memory_expandable_segments:True`时，同时解除虚拟内存物理内存映射。


## 定义文件
torch_npu/npu/memory.py

## 函数原型

```
torch_npu.npu.host_empty_cache()
```

## 参数说明
无

## 返回值说明
无

## 约束说明

无


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty([1024, 1024]).pin_memory()
>>> del x
>>> torch_npu.npu.host_empty_cache()
>>> print(torch_npu.npu.host_memory_stats())
```