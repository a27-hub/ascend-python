# （beta）torch_npu.npu_softmax_cross_entropy_with_logits

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

计算softmax的交叉熵cost。

## 函数原型

```
torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor
```


## 参数说明

- **features** (`Tensor`)：张量，一个“batch_size \* num_classes”矩阵。
- **labels** (`Tensor`)：与“features”同类型的张量。一个“batch_size \* num_classes”矩阵。

## 返回值说明

`Tensor`
