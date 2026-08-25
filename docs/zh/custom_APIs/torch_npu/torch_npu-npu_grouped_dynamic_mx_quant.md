# torch\_npu.npu\_grouped\_dynamic\_mx\_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- API功能：根据传入的分组索引起始值（`group_index`），对输入的各个分组以基本块（`blocksize`）的粒度进行目标数据类型为float4/float8的动态MX量化。

- 计算公式：根据传入的分组索引起始值，对输入`x`按第0维切分成不同分组（groups），对每个group按第0维方向以`blocksize`为基本块粒度进行DynamicMxQuant，每个group的具体计算过程如下。

  - 场景1，当`scale_alg`为0时：

    - 对每个group，在第0维上按k = `blocksize`个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 计算出这组数对应的量化尺度mxscale_pre。
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale\_pre = 2^{shared\_exp}
    $$
    - 这组数每一个除以mxscale_pre，根据`round_mode`转换到对应的`dst_type`，得到量化结果y。
    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale\_pre, round\_mode), \space i\space from\space 1\space to\space blocksize
    $$
    - 量化后的$P_i$按对应的$V_i$的位置组成输出y，mxscale_pre按对应的`group_index`分组，分组内第一个维度pad为偶数，组成输出mxscale。

  - 场景2，当`scale_alg`为1时，仅适用于`float8_e4m3fn`、`float8_e5m2`类型：

    - 对每个group按k = `blocksize`个数分块，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值：
    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$
    - 将FP32映射到目标数据类型可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值：
    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$
    - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
    - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
    $$
    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$。

  - 场景3，当`scale_alg`为2时，仅适用于`float4_e2m1fn_x2`、`float4_e1m2fn_x2`类型：

    - 当`dst_type_max`为0.0、6.0或7.0（`float4_e2m1fn_x2`）或1.875（`float4_e1m2fn_x2`）时：
      - 对每个group按k = `blocksize`个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$：
      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果尾数位的高比特前一/两/三位为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases}
      $$
      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize
      $$
    - 当`dst_type_max`不为0.0、6.0或7.0（`float4_e2m1fn_x2`）或1.875（`float4_e1m2fn_x2`）时：
      - 对每个group按k = `blocksize`个数分块，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
      - 找到该块中数值的最大绝对值：
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - 将FP32映射到目标数据类型可表示的范围内，其中当`dst_type_max`为0时$Amax(DType)$为目标精度能表示的最大值，当`dst_type_max`不为0时$Amax(DType)$为`dst_type_max`传入值：
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
      - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
      - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
      - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
      $$
      - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
      - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$。

  - emax：对应数据类型的最大正则数的指数位。

    | dst_type | emax |
    | :---: | :---: |
    | float4_e2m1fn_x2 | 2 |
    | float4_e1m2fn_x2 | 0 |
    | float8_e4m3fn | 8 |
    | float8_e5m2 | 15 |

## 函数原型

```python
torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, *, round_mode="rint", dst_type=torch.float8_e5m2, blocksize=32, scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，待量化的输入张量，数据类型支持`float16`、`bfloat16`，shape仅支持2维，支持非连续的Tensor，数据格式支持$ND$。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，`x`的最后一维必须是偶数。
- **group_index** (`Tensor`)：必选参数，量化分组的起始索引，数据类型支持`int32`，shape仅支持1维，数据格式支持$ND$。索引要求大于等于0，且非递减，并且最后一个数需要与`x`的第一个维度大小相等。
- **\***：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **round_mode** (`str`)：可选参数，数据转换的模式，默认值为`"rint"`。
  - 当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，仅支持取值`"rint"`。
  - 当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，支持取值`"rint"`、`"round"`、`"floor"`。
- **dst_type** (`int`)：可选参数，指定量化后输出y的数据类型。支持的类型为`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`，默认值为`torch.float8_e5m2`。
- **blocksize** (`int`)：可选参数，指定每次量化的元素个数，仅支持取值32，默认值为32。
- **scale_alg** (`int`)：可选参数，指定计算`mxscale`的算法，默认值为0。取值范围：0代表场景1、1代表场景2、2代表场景3。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时支持取值0和2；当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时支持取值0和1。
- **dst_type_max** (`float`)：可选参数，指定目标数据类型的最大表示值，默认值为0.0。支持取值0.0和1.75-3.5（`float4_e1m2fn_x2`）、0.0和6.0-12.0（`float4_e2m1fn_x2`），仅在`scale_alg`为2时支持设置该值；取值为0.0时使用目标精度能表示的最大值，取值为非0.0时使用传入值。

## 返回值说明

- **y** (`Tensor`)：量化后的输出张量。当`dst_type`为`float8_e4m3fn`或`float8_e5m2`时，y的数据类型与`dst_type`对应，shape与输入`input`一致。当`dst_type`为`float4_e2m1fn_x2`或`float4_e1m2fn_x2`时，y的实际数据类型为`uint8`，shape的最后一维为`input`最后一维的一半。
- **mxscale** (`Tensor`)：每个分组对应的量化尺度，数据类型为`float8_e8m0fnu`，实际返回的数据类型为`uint8`。假设`input`的shape为[m, n]，group_index的shape为[g]，则mxscale的shape为[m / (blocksize * 2) + g, n, 2]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入x的维度必须为2，group_index的维度必须为1。
- group_index的元素要求大于等于0，且非递减，并且最后一个元素需要与`x`的第一个维度大小相等。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    x = torch.randn((8, 32), dtype=torch.bfloat16).npu()
    group_index = torch.tensor([4, 8], dtype=torch.int32).npu()
    y, mxscale = torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, round_mode="rint", dst_type=torch.float8_e4m3fn, blocksize=32)
    print(y)
    print(mxscale)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair

    class GroupedDynamicMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, group_index):
            return torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, round_mode="rint", dst_type=torch.float8_e4m3fn, blocksize=32)

    x = torch.randn((8, 32), dtype=torch.bfloat16).npu()
    group_index = torch.tensor([4, 8], dtype=torch.int32).npu()
    model = GroupedDynamicMxQuantModel()
    model.to('npu')

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)

    y, mxscale = model(x, group_index)
    print(y)
    print(mxscale)
    ```
