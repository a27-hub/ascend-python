# torch_npu.npu_rotate_quant

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    ×     |
| Atlas 推理系列产品                          |    ×     |
| Atlas 训练系列产品                          |    ×     |

## 功能说明

- API功能：对张量x进行旋转变换，然后执行可选的clamp操作，最后执行对称动态量化（INT8/INT4）或MX量化（FLOAT4/FLOAT8）。

- 计算公式：

  1. 旋转变换

    $$
    Y = (x.\text{reshape}(*,k) @ \text{rotation}).\text{reshape}(m, n)
    $$

  2. 当alpha在有效取值范围(0.0, 1.0)内，执行clamp计算

    $$
    groupMaxVal = GroupMax(|Y|) \\
    limit = alpha * groupMaxVal \\
    Y = Y.clamp(min=-limit, max=limit)
    $$

  3. 量化

    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：对称动态量化（pertoken逐行量化）

      $$
      s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
      $$

      其中 $C_{\text{MAX}}$：int8取127，int4取7。

    - <term>Ascend 950PR/Ascend 950DT</term>：MX量化（支持FLOAT4_E2M1/FLOAT8_E4M3FN/FLOAT8_E5M2），scaleAlg=0/1/2 对应三种场景。

## 函数原型

```python
torch_npu.npu_rotate_quant(x, rot, *, alpha=None, y_dtype=None, axis=-1, round_mode="rint", scale_alg=0, dst_type_max=0.0, trans=False) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，待旋转量化的输入张量，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，维度支持1-7，支持非连续的Tensor。不支持空Tensor。
- **rot** (`Tensor`)：必选参数，旋转矩阵，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，维度支持2或3，支持非连续的Tensor。最后两维必须是方阵(K, K)。x和rot的数据类型必须相同。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **alpha** (`Tensor`)：可选参数，clamp的缩放系数，默认值为`None`。shape为(1,)，数据类型为`bfloat16`。有效取值范围(0.0, 1.0)，不在有效取值范围内不做clamp处理。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前版本仅支持传入空指针。
    - <term>Ascend 950PR/Ascend 950DT</term>：需配合scale_alg=2使用。
- **y_dtype** (`ScalarType`)：可选参数，指定量化输出的类型，默认值为`None`（当作`int8`处理）。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持取值`int8`、`quint4x2`。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持取值`float8_e4m3fn`、`float8_e5m2`、`float4_e2m1fn_x2`。
- **axis** (`int`)：可选参数，表示量化发生的轴（reduce轴），默认值为`-1`。目前只支持`-1`或最后一维的正索引。
- **round_mode** (`str`)：可选参数，表示数据转换的模式，默认值为`"rint"`。支持`"rint"`、`"round"`、`"floor"`。当`y_dtype`为`float8_e4m3fn`或`float8_e5m2`时，仅支持`"rint"`。
- **scale_alg** (`int`)：可选参数，表示scale的计算算法，默认值为`0`。支持取值0、1、2。
    - 当`y_dtype`为`float8_e4m3fn`或`float8_e5m2`时，只支持0和1。
    - 当`y_dtype`为`float4_e2m1fn_x2`时，只支持0和2。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：目前只支持0。
- **dst_type_max** (`float`)：可选参数，表示量化目标类型的最大值，默认值为`0.0`。当`scale_alg=2`时`dst_type_max`必须在[6.0, 12.0]范围内，其余场景仅支持`0.0`。
- **trans** (`bool`)：可选参数，表示输出y是否转置，默认值为`False`。目前只支持`False`。

## 返回值说明

- **y** (`Tensor`)：量化后的输出张量，shape与x相同（INT4时最后一维除以8，FLOAT4时最后一维除以2）。
    - 当`y_dtype=int8`时，y数据类型为`int8`，shape与x一致。
    - 当`y_dtype=quint4x2`时，y数据类型为`int32`，最后一维为x最后一维的1/8，每个int32存放8个int4。
    - 当`y_dtype=float4_e2m1fn_x2`时，y数据类型为`uint8`，最后一维为x最后一维的1/2。
    - 当`y_dtype=float8_e4m3fn`或`float8_e5m2`时，y数据类型为对应的float8类型，shape与x一致。
- **scale** (`Tensor`)：动态量化计算出的缩放系数。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型为`float32`，shape为`(M,)`。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型为`uint8`（FLOAT8_E8M0），shape为`(*, CeilDiv(N,64), 2)`。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- x的shape为(*, N)，维度范围[1, 7]；rot的shape为(K, K)或(N/K, K, K)，维度范围[2, 3]。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - N必须在[128, 16000]范围内。
    - K必须在[16, 1024]范围内。
    - x最后一维长度(N)必须是K的整数倍，且必须能整除8。
    - x和rotation的数据类型必须相同。
- <term>Ascend 950PR/Ascend 950DT</term>：
    - K当前版本仅支持取32、64、128。
    - x最后一维长度(N)必须是K的整数倍。

## 调用示例

- 单算子模式调用（A2/A3，INT8量化）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> M, N, K = 32, 256, 64
    >>> x = torch.randn(M, N, dtype=torch.bfloat16, device='npu')
    >>> rot = torch.eye(K, dtype=torch.bfloat16, device='npu')
    >>> y, scale = torch_npu.npu_rotate_quant(x, rot)
    >>> y.shape, y.dtype
    (torch.Size([32, 256]), torch.int8)
    >>> scale.shape, scale.dtype
    (torch.Size([32]), torch.float32)
    ```

- 单算子模式调用（A2/A3，INT4量化）

    ```python
    >>> M, N, K = 32, 256, 64
    >>> x = torch.randn(M, N, dtype=torch.bfloat16, device='npu')
    >>> rot = torch.eye(K, dtype=torch.bfloat16, device='npu')
    >>> y, scale = torch_npu.npu_rotate_quant(x, rot, y_dtype=torch.quint4x2)
    >>> y.shape, y.dtype
    (torch.Size([32, 32]), torch.int32)  # N/8 = 32
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            K = 64
            self.rot = torch.nn.Parameter(torch.eye(K, dtype=torch.bfloat16))

        def forward(self, x):
            return torch_npu.npu_rotate_quant(x, self.rot)

    model = Model().to('npu')
    model = torch.compile(model, backend=npu_backend, dynamic=True)

    x = torch.randn(32, 256, dtype=torch.bfloat16, device='npu')
    y, scale = model(x)
    ```
