# （beta）torch_npu.npu.restart_device

> [!NOTICE]  
> 本接口为预留接口，暂不支持。

## 功能说明

恢复对应device上的状态，后续在此device上可以继续进行计算执行。

## 函数原型

```python
torch_npu.npu.restart_device(device_id: int, rebuild_all_resources: int = False, disable_tensor_unsafe_check: bool = False) -> None
```

## 参数说明

- **device_id** (`int`)：需要处理的device id。
- **rebuild_all_resources** (`int`)：该参数为一个保留参数，在当前接口中没有实际的功能。
- **disable_tensor_unsafe_check** (`bool`)：是否禁用tensor不安全检查，默认为False。当设置为False时，restart_device会标记所有NPU tensor为不安全状态并开启不安全检查；当设置为True时，跳过该检查。

## 约束说明

要确保是一个有效的device，这个device可以是被stop过的也可以是没有被stop。

## 调用示例

 ```python
>>> import torch
>>> import torch_npu  
>>> torch.npu.set_device(0) 
>>> torch_npu.npu.stop_device(0)
>>> torch_npu.npu.restart_device(0)
 ```
