# （beta）torch.distributed.is_hccl_available
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

参考原生接口`torch.distributed.is_nccl_available`功能，具体请参考[https://pytorch.org/docs/stable/distributed.html\#torch.distributed.is_nccl_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available)。
`torch_npu.distributed.is_hccl_available`接口用于判断HCCL通信后端是否可用。

## 函数原型
```
torch.distributed.is_hccl_available()
```

## 返回值说明
`Bool`

## 调用示例

```python
import torch
import torch_npu

torch.distributed.is_hccl_available()
```

结果示例：

```
True
```