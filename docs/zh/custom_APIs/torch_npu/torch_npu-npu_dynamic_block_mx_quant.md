# torch\_npu.npu\_dynamic\_block\_mx\_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：对输入张量，以数据块（32*32）为基本块进行MX量化转换为目的数据类型。在每个基本块中，根据`scale_alg`的取值采取不同的scale算法计算出当前块对应的量化参数`scale`（1*1），将其广播为`scale1`（32*1）和`scale2`（1*32）输出。同时对基本块中的每一个数除以`scale`，根据`round_mode`转换到对应的`dst_type`，得到量化结果`y`。

- 计算公式：

  - 将输入x以数据块（32*32）为基本块进行分组，一个数据块的数 $\{\{V_i\}_{i=1}^{32*32}\}$ 量化为 $\{scale, \{P_i\}_{i=1}^{32*32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    scale = 2^{shared\_exp} \\
    P_i = cast\_to\_dst\_type(V_i/scale, round\_mode), \space i\space from\space 1\space to\space 32*32
    $$

  - 同时将scale（1*1）广播为scale1（32*1）和scale2（1*32）作为输出scale1和scale2，量化后的$P_i$按对应的$V_i$的位置组成输出y。

  - emax：对应数据类型的最大正则数的指数位。

    | dst_type | emax |
    | :---: | :---: |
    | float4_e2m1fn_x2 | 2 |
    | float4_e1m2fn_x2 | 0 |
    | float8_e4m3fn | 8 |
    | float8_e5m2 | 15 |

## 函数原型

```python
torch_npu.npu_dynamic_block_mx_quant(input, *, round_mode="rint", dst_type=296, scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，待量化的输入张量，数据类型支持`float16`、`bfloat16`，shape支持2-3维度，支持非连续的Tensor，数据格式支持ND。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，input的最后一维必须是偶数。不支持空Tensor。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round_mode** (`str`)：可选参数，数据转换的模式。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，支持取值`"rint"`、`"floor"`、`"round"`；当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，仅支持取值`"rint"`。默认值为`"rint"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y的数据类型。支持取值291（`float8_e5m2`）、292（`float8_e4m3fn`）、296（`float4_e2m1fn_x2`）、297（`float4_e1m2fn_x2`），默认值为296。
- **scale_alg** (`int`)：可选参数，指定计算`scale`的算法，支持取值0和2。取值为0时采用基于shared_exp的MX量化算法；取值为2时仅适用于`float4_e2m1fn_x2`类型，采用Dynamic Dtype Range实现。默认值为0。
- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大表示值。在`scale_alg`为2且`dst_type`为`float4_e2m1fn_x2`时生效，取值为0.0时使用目标精度能表示的最大值，当前仅支持取值0.0、6.0、7.0。默认值为0.0。

## 返回值说明

- **y** (`Tensor`)：量化后的输出张量。当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，y的数据类型与`dst_type`对应，shape与输入x一致。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，y的实际数据类型为uint8，shape的最后一维为x最后一维的一半。
- **scale1** (`Tensor`)：-1轴每个分组对应的量化尺度，数据类型为`float8_e8m0fnu`，实际返回的数据类型为uint8。shape比输入x多一维，最后一维为2。
- **scale2** (`Tensor`)：-2轴每个分组对应的量化尺度，数据类型为`float8_e8m0fnu`，实际返回的数据类型为uint8。shape比输入x多一维，最后一维为2，且输出需要对每两行数据进行交织处理。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入x的维度必须为2-3。
- 该接口为确定性实现。
- 输入`input`和输出`scale1`、`scale2`的shape约束关系：
  - rank(scale1) = rank(x) + 1
  - rank(scale2) = rank(x) + 1
  - scale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2
  - scale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2
  - scale1.shape[-1] = 2
  - scale2.shape[-1] = 2
  - 其他维度与输入x一致
  - 举例：输入x的shape为[B, M, N]，对应的y的shape为[B, M, N]，scale1的shape为[B, M, (ceil(N/32)+2-1)/2, 2]，scale2的shape为[B, (ceil(M/32)+2-1)/2, N, 2]。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((32, 32), dtype=torch.bfloat16).npu()
    >>> y, scale1, scale2 = torch_npu.npu_dynamic_block_mx_quant(x, round_mode="rint", dst_type=292, scale_alg=0, dst_type_max=0.0)
    >>> print(y)
    tensor([[  0,  18,  -2, ..., -16,   0,  24],
            [  0,  -8,  32, ...,   0, -16,   0],
            ...,
            [  0,  24,  16, ...,  -8,  32,   0],
            [  0, -16,   0, ...,  24,   0,  18]], device='npu:0',
           dtype=torch.float8_e4m3fn)
    >>> print(scale1)
    tensor([[[  0, 128]],
            [[  0, 128]]], device='npu:0', dtype=torch.uint8)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicBlockMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch_npu.npu_dynamic_block_mx_quant(x, round_mode="rint", dst_type=292, scale_alg=0, dst_type_max=0.0)

    x = torch.randn((32, 32), dtype=torch.bfloat16).npu()
    model = DynamicBlockMxQuantModel()
    model.to('npu')

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

    y, scale1, scale2 = model(x)
    print(y)
    print(scale1)
    ```
