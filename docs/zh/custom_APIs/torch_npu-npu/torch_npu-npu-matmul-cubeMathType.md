# torch_npu.npu.matmul.cubeMathType

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

设置或查询matmul类算子的cube计算类型。

`torch_npu.npu.matmul.cubeMathType` 用于控制matmul类算子的计算精度类型，与华为昇腾NPU的Cube计算单元相关。该属性可设置不同的数学计算模式，以在精度和性能之间取得平衡。

## 函数原型

```
torch_npu.npu.matmul.cubeMathType: CubeMathType
```

`CubeMathType` 是枚举类，可通过 `torch_npu.npu.CubeMathType` 访问。

## 参数说明

**CubeMathType**：设置cube计算类型，可选值如下：

| 枚举值                      | 值   | 说明                               |
| --------------------------- | ---- | ---------------------------------- |
| CubeMathType.KEEP_DTYPE     | 0    | 保持原始数据类型，不进行精度转换   |
| CubeMathType.ALLOW_FP32_DOWN_PRECISION | 1    | 允许FP32降精度                     |
| CubeMathType.USE_FP16       | 2    | 使用FP16计算模式                   |
| CubeMathType.USE_HF32       | 3    | 使用HF32计算模式                   |
| CubeMathType.FORCE_GRP_ACC_FOR_FP32 | 4    | 强制使用FP32分组精度               |

## 返回值说明

返回`CubeMathType`枚举类型，返回当前cubeMathType的值。

## 调用示例

```python
>>>import torch
>>>import torch_npu
>>>torch_npu.npu.matmul.cubeMathType
<CubeMathType.KEEP_DTYPE: 0>
>>>torch_npu.npu.CubeMathType.USE_HF32
<CubeMathType.USE_HF32: 3>
>>>torch_npu.npu.matmul.cubeMathType = torch_npu.npu.CubeMathType.USE_HF32
>>>torch_npu.npu.matmul.cubeMathType
<CubeMathType.USE_HF32: 3>
>>>torch_npu.npu.matmul.cubeMathType = torch_npu.npu.CubeMathType.KEEP_DTYPE
>>>torch_npu.npu.matmul.cubeMathType
<CubeMathType.KEEP_DTYPE: 0>
```
