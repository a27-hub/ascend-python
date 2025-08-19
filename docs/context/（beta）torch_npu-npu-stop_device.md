# （beta）torch_npu.npu.stop_device

>**须知：**<br>
>本接口为预留接口，暂不支持。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
<<<<<<< HEAD
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

=======
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
>>>>>>> d1c6186d06d5ab8610b972a0e7bf2ca6bec4c80b

## 功能说明

停止对应device上的计算，对于没有执行的计算进行清除，后续在此device上执行计算会报错。

## 函数原型

```
torch_npu.npu.stop_device(device_id: int) -> int 
```

<<<<<<< HEAD

## 参数说明

**device_id** (`int`)：需要处理的device id。


## 返回值说明
`int`

代表执行结果，0表示执行成功，1表示执行失败。

## 约束说明
=======
## 参数说明

**device_id**（`int`）：需要处理的device id，确保是一个有效地device。

## 返回值说明
>>>>>>> d1c6186d06d5ab8610b972a0e7bf2ca6bec4c80b

`int`

<<<<<<< HEAD
=======
返回值为`int`，代表执行结果，0表示执行成功，1表示执行失败。
>>>>>>> d1c6186d06d5ab8610b972a0e7bf2ca6bec4c80b

## 调用示例

 ```python
>>> import torch
>>> import torch_npu  
>>> torch.npu.set_device(0) 
>>> torch_npu.npu.stop_device(0)
 ```

