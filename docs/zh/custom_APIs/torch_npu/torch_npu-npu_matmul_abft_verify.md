# torch_npu.npu_matmul_abft_verify

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

## 功能说明

- API功能：实现基于方差估计自适应门限算法（V-ABFT）的GEMM容错检测算子。接收矩阵A、B以及预先计算的矩阵乘结果C，对C进行分块ABFT校验，检测静默计算错误并输出逐行检测结果。

- 计算公式：

  $$
  C = A \times B, \quad A \in \mathbb{R}^{M \times K},\; B \in \mathbb{R}^{K \times N}
  $$

  $$
  C^r = C\times r, \quad B^r = B\times r
  $$

  其中阈值$Threshold_i$由输入矩阵A、B的局部统计特征（均值、标准差）动态估计，无需依赖C矩阵输出结果。

> [!NOTE]
> - 自适应阈值：根据矩阵大小与值域自动确定用于比对的阈值，在保证检出率的同时避免误检。自适应阈值算法与推导见V-ABFT相关文档。
> - 计算量显著小于基于重新计算的容错方案：在矩阵维度m=n=k=a时，本算子只需8a²次浮点计算，而重算需要2a³次浮点计算。
> - `checksum_weight`全为1时，对应普通的分块行校验和。
> - 输出`comp_row`中每个bit表示一行一段的检测结果，1表示正确，0表示检测到错误。

## 函数原型

```
torch_npu.npu_matmul_abft_verify(a, b, c, checksum_weight, *, e_max=0.001) -> Tensor
```

## 参数说明

- **a** (`Tensor`)：必选参数，矩阵乘法输入A，shape为$[M, K]$，数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$。
- **b** (`Tensor`)：必选参数，矩阵乘法输入B，shape为$[K, N]$，数据类型支持`float16`、`bfloat16`、`float32`（须与`a`一致），数据格式支持$ND$。
- **c** (`Tensor`)：必选参数，预先计算的矩阵乘结果C = A × B，作为容错检测的数据输入，shape为$[M, N]$，数据类型仅支持`float32`，数据格式支持$ND$。
- **checksum_weight** (`Tensor`)：必选参数，列编码值向量，对应ABFT中的行校验和编码向量$r$（加权向量），shape为$[N]$，数据类型须与`a`一致，数据格式支持$ND$。
- <strong>*</strong>：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，需通过键值对方式传入，未赋值时使用默认值。
- **e_max** (`float`)：可选参数，误差阈值系数，控制故障检测的灵敏度，默认值为`0.001`。`a`、`b`为`bfloat16`精度时推荐值为0.001，为`float32`精度时推荐值为0.00002。使用小于推荐值的e_max会增大检出率，但也可能出现误检情况；出现误报时可以将e_max调大。

## 返回值说明

`Tensor`

压缩后的行方向故障检测位流输出`comp_row`，数据类型为`uint8`，shape为$[ceil(M/8) \times splitN]$，其中$splitN = \lceil N / 256\rceil$。每个bit表示一行一段的检测结果，1表示正确，0表示检测到错误。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口不支持图模式。
- 该接口默认确定性实现。
- 输入矩阵`a`、`b`和`c`必须为2维，shape分别为$[M, K]$、$[K, N]$和$[M, N]$，且`a`的第1维（K）与`b`的第0维（K）必须相等。
- 输入向量`checksum_weight`的shape必须为$[N]$。
- `e_max`不得为负数。
- 支持的数据类型组合为：

  | a       | b       | c       | checksum_weight |
  |:-------:|:-------:|:-------:|:-------:|
  | float16 | float16 | float32 | float16 |
  | bfloat16| bfloat16| float32 | bfloat16|
  | float32 | float32 | float32 | float32 |

- 不支持的场景：
  - 不支持$ND$格式以外的数据格式。
  - 不支持非连续的Tensor。
  - 不支持`a`、`b`和`c`的维度不为2的场景。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, n, k = 16, 16, 16
    >>> a = (torch.randn(m, k, dtype=torch.float32) * 0.1).half().npu()
    >>> b = (torch.randn(k, n, dtype=torch.float32) * 0.1).half().npu()
    >>> c = torch.matmul(a.float(), b.float()).npu()
    >>> checksum_weight = torch.ones(n, dtype=torch.float16).npu()
    >>> result = torch_npu.npu_matmul_abft_verify(a, b, c, checksum_weight)
    >>> result
    tensor([255, 255], device='npu:0', dtype=torch.uint8)
    ```

- 检错场景调用（对C注入破坏后，对应行段检测结果为0）

    ```python
    >>> c_corrupted = c.clone()
    >>> c_corrupted[0, :] += 50.0
    >>> result = torch_npu.npu_matmul_abft_verify(a, b, c_corrupted, checksum_weight)
    >>> result
    tensor([0, 255], device='npu:0', dtype=torch.uint8)
    ```
