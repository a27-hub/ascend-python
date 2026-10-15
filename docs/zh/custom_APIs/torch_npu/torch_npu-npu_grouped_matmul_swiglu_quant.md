# torch_npu.npu_grouped_matmul_swiglu_quant

## 产品支持情况

<!-- npu="950" id4 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id4 -->

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

<!-- npu="950,A3,910b" id3 -->
- API功能：融合GroupedMatmul（分组矩阵乘）、dequant（反量化）、swiglu（SwiGLU激活）和quant（量化）四个计算环节，deepseek模型使用，对比小算子做性能优化，weight需以FRACTAL\_NZ格式传入。该API是`npu_grouped_matmul_swiglu_quant_v2`的简化版本，支持<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>，以及<term>Ascend 950PR&950DT系列产品</term>上的INT8 A8W8 per-token × per-channel量化场景。torch_npu在各型号上统一调用`aclnnGroupedMatmulSwigluQuantWeightNZ`，由ACLNN在950 INT8 × INT8 weight NZ场景转发至`aclnnGroupedMatmulSwigluQuantWeightNzV2`。
  
  多算子融合计算顺序（依次执行）：
  1. GroupedMatmul：根据group\_list对token按组切分，各组计算$X_{i}\cdot W_{i}$；
  2. dequant：矩阵乘结果与激活量化因子$x_scale$、权重量化因子$w_scale$逐元素相乘，完成反量化；
  3. swiglu：反量化结果沿N轴对半切分为$C_{i,act}$与$gate_{i}$，计算$S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$；
  4. quant：$Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$，$Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$，得到量化输出Q与量化因子Q\_scale。
<!-- end id3 -->
- 量化场景**A8W8**（A指激活矩阵，W指权重矩阵，8指`torch.int8`数据类型）：
  - 输入：
    - $X∈\mathbb{Z_8}^{M \times K}$为激活矩阵（左矩阵），M是总token数，K是特征维度；
    - $W∈\mathbb{Z_8}^{E \times K \times N}$为分组权重矩阵（右矩阵），E是专家个数，N是输出维度；
    - $w\_scale∈\mathbb{R}^{E \times N}$为权重逐通道缩放因子；
    - $x\_scale∈\mathbb{R}^{M}$为激活逐token缩放因子；
    - $groupList∈\mathbb{N}^{E}$为cumsum的分组索引列表。
  - 输出：
    - $Q∈\mathbb{Z_8}^{M \times N / 2}$为量化后的输出矩阵；
    - $Q\_scale∈\mathbb{R}^{M}$为量化缩放因子。
  - 计算过程：

    $$
    C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\,\text{Broadcast}} \odot w\_scale_{i\,\text{Broadcast}}
    $$

    $$
    C_{i,act}, gate_{i} = split(C_{i})
    $$

    $$
    S_{i}=Swish(C_{i,act})\odot gate_{i}
    $$

  - 量化输出结果：

    $$
    Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
    $$

    $$
    Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
    $$

- A2/A3量化场景**A8W4**（A指激活矩阵，W指权重矩阵，4指`torch_npu.int4`数据类型）：激活为`torch.int8`、权重为`torch_npu.int4`，通过将左矩阵拆分为高低4bit两部分分别与权重做矩阵乘，再结合离线生成的辅助矩阵bias还原结果，其余计算过程与A8W8一致。该场景下`weight_scale`支持2维（perchannel）或3维（pergroup）输入。
  - 输入：
    - $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度；
    - $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度；
    - $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）；
    - $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num是在K轴维度上的分组数，N是输出维度；
    - $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数；
    - $groupList∈\mathbb{N}^{E}$：cumsum的分组索引列表。
  - 输出：
    - $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵；
    - $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
  - 计算过程：
    1. 生成辅助矩阵（weightAssistMatrix）的计算过程：
        - perchannel量化：

          $$
          weightAssistMatrix_{i} = 8 × w\_scale × Σ_{k=0}^{K-1} weight[:,k,:]
          $$

        - pergroup量化（$w\_scale$为3维）:
  
          $$
          weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × w\_scale[:, \lfloor\frac{k}{num\_per\_group}\rfloor, :])
          $$

          其中

          $$ num\_per\_group = K // K\_group\_num $$

    2. 将左矩阵$\mathbb{Z_8}$，转变为高低位两部分的$\mathbb{Z_4}$：

        $$
        X\_high\_4bits_{i} = \left\lfloor \frac{X_{i}}{16} \right\rfloor
        $$

        $$
        X\_low\_4bits_{i} = And(X_{i}, \mathrm{0x0f}) - 8
        $$

    3. 做矩阵乘时，开启perchannel或pergroup量化。
        - perchannel量化：

          $
          C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}
          $

          $
          C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}
          $

        - pergroup量化:

          $
          C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )
          $

          $
          C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )
          $

    4. 将高低位的矩阵乘结果还原为整体的结果：

        $$
        C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}
        $$

        $$
        C_{i,act}, gate_{i} = split(C_{i})
        $$

        $$
        S_{i}=Swish(C_{i,act})\odot gate_{i}
        $$

        其中 $Swish(x)=\frac{x}{1+e^{-x}}$ 。

  - 量化输出结果：

    $$
    Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
    $$

    $$
    Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
    $$

- group\_list分组说明：根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList))$。例如groupList=[3,4,4,6]：第0个右矩阵W[0,:,:]对应token x[0:3]；第1个对应x[3:4]；第2个对应x[4:4]（0个token）；第3个对应x[4:6]。groupList中未指定的部分不会参与更新：例如groupList=[12,14,18]、x的shape为[30,N/2]时，Q[18:, :]与Q\_scale[18:]不会进行更新或初始化，其中数据为显存空间申请时的原数据，即输出的Q[:groupList[-1],:]和Q\_scale[:groupList[-1]]为有效数据部分。

> [!NOTE]
>
> - weight需先通过`torch_npu.npu_format_cast(weight, 29)`转换为FRACTAL\_NZ格式（5维）再传入；该接口会忽略weight的数据格式标志，直接强制视为FRACTAL\_NZ格式，但要求weight的存储shape为NZ打包后的5维。
> - 该接口提供非原地语义：输出output为新建Tensor，不修改输入。

## 函数原型

```python
torch_npu.npu_grouped_matmul_swiglu_quant(x, weight, group_list, weight_scale, x_scale, *, bias=None, offset=None) -> (Tensor, Tensor, Tensor)
```

### ACLNN 调用与转发说明

`torch_npu.npu_grouped_matmul_swiglu_quant`继续使用非V2的`aclnnGroupedMatmulSwigluQuantWeightNZ`两阶段接口。A5（Ascend 950 / DAV_3510）上，INT8 × INT8、激活per-token与权重per-channel量化的转发由ACLNN第一阶段完成，op-plugin不选择硬件转发分支，也不直接调用V2。

| ACLNN旧入口 | A5 INT8 A8W8第一阶段转发目标 | 第二阶段 |
| --- | --- | --- |
| `aclnnGroupedMatmulSwigluQuant`（普通入口） | `aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize` | 仍调用旧普通入口执行返回的executor |
| `aclnnGroupedMatmulSwigluQuantWeightNZ` | `aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize` | 仍调用旧WeightNZ入口执行返回的executor |

上表普通入口转发是CANN ACLNN的兼容能力；本Python旧接口实际使用WeightNZ入口，仍要求真实NZ权重，不因此新增普通ND权重支持。需要ND权重或显式V2参数时，使用[`npu_grouped_matmul_swiglu_quant_v2`](torch_npu-npu_grouped_matmul_swiglu_quant_v2.md)，并遵守其输入及量化模式约束。

两条旧ACLNN转发路径固定`dequantMode=0`、`dequantDtype=0`（GE FLOAT）、`quantMode=0`、`groupListType=0`，要求`bias`、`offset`、`outputOffset`为空；V2第一阶段失败直接返回，不回退。Python旧接口保留三个返回值，第三个是未定义数值的FLOAT标量占位；向ACLNN传入的`outputOffset`为空指针。A2/A3继续旧调用路径。

CANN须包含对应旧入口的转发实现和V2接口。上述说明针对旧入口兼容转发；显式Python V2接口保留自身的参数和调用选择规则。

## 参数说明

- **x**（`Tensor`）：必选参数，左矩阵，公式中的$X$。shape支持2维[M, K]，数据类型支持`torch.int8`，数据格式支持ND，支持非连续Tensor。A2/A3的A8W8场景K必须小于65536，A8W4场景K必须小于20000；950约束见下文。
- **weight**（`Tensor`）：必选参数，权重矩阵，公式中的$W$。需先通过`torch_npu.npu_format_cast(weight, 29)`转换为FRACTAL\_NZ格式（5维存储shape），数据类型支持`torch.int8`、`torch_npu.int4`、`torch.int32`（`torch.int32`为适配用途，实际1个`torch.int32`会被解释为8个`torch_npu.int4`数据），支持非连续Tensor。该接口将已有NZ数据强制视为FRACTAL\_NZ格式；950仅支持torch.int8，不接受普通三维ND权重。
- **group\_list**（`Tensor`）：必选参数，指示每个分组参与计算的Token个数，公式中的$groupList$。shape支持1维，长度需与weight的首轴维度（专家数E）相等，数据类型支持`torch.int64`，数据格式支持ND，支持非连续Tensor。为cumsum形式的分组索引列表，最后一个值约束了输出数据的有效部分。
- **weight\_scale**（`Tensor`）：必选参数，右矩阵的量化因子，公式中的$w\_scale$。shape支持2维（perchannel）或3维（pergroup），首轴长度需与weight的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴（N）相同。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`、`torch.int64`（perchannel），数据格式支持ND，支持非连续Tensor。950仅支持二维perchannel scale，数据类型支持torch.float32、torch.float16、torch.bfloat16。
- **x\_scale**（`Tensor`）：必选参数，左矩阵的量化因子，公式中的$x\_scale$。shape支持1维，长度需与x的首轴维度（M）相等。数据类型支持`torch.float32`，数据格式支持ND，支持非连续Tensor。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **bias**（`Tensor`）：可选参数，计算矩阵乘时的辅助矩阵，对应公式中的$weightAssistMatrix$。shape支持2维[E, N]，数据类型支持`torch.float32`，数据格式支持ND。仅在A8W4场景生效，A8W8场景需传入None。默认值为None。
- **offset**（`Tensor`）：可选参数，perchannel非对称反量化的偏移。预留输入，暂不支持，需传入None。默认值为None。

## 返回值说明

- **output**（`Tensor`）：输出的量化结果，公式中的$Q$。数据类型支持`torch.int8`，shape支持2维[M, N/2]，数据格式支持ND。group\_list指定了输入和输出中的有效值范围，有效数据截至group\_list[-1]，即`output[:group_list[-1], :]`为有效数据，其余部分不会更新。
- **output\_scale**（`Tensor`）：输出的量化因子，公式中的$Q\_scale$。数据类型支持`torch.float32`，shape支持1维[M]，数据格式支持ND。有效数据截至group\_list[-1]，即`output_scale[:group_list[-1]]`为有效数据。
- **output\_offset**（`Tensor`）：输出的非对称量化的偏移。预留输出，暂未使用，返回shape为`[]`的0维`torch.float32`占位Tensor（不是零元素Tensor）。其数值未定义，不应读取或参与计算；950同样保留该返回值以兼容旧接口；向ACLNN传入的outputOffset为空指针。

## 约束说明

### Ascend 950PR&950DT系列产品

- 仅支持INT8 A8W8、激活per-token与权重per-channel量化，输出同样为INT8。不支持INT4、MX量化或多权重TensorList。
- x为ND格式的二维[M, K]，不支持转置；x_scale为一维[M]，数据类型为torch.float32。
- weight为torch.int8，支持三维逻辑view和五维显式NZ数据。非转置权重的逻辑shape为[E, K, N]，NZ存储shape为[E, ceil(N/32), ceil(K/16), 16, 32]；转置权重的stride和物理布局需符合底层WeightNzV2接口约束。
- weight必须已经采用NZ存储。该接口不会把普通三维ND权重自动转换成NZ，也不会修改输入Tensor的格式标记。
- weight_scale必须为二维[E, N]，支持torch.float32、torch.float16、torch.bfloat16；不支持三维per-group scale或torch.int64 scale。
- N必须大于0且为64的整数倍。逻辑N从weight_scale的尾轴获取，不从NZ分块维度获取。
- group_list为一维[E]、torch.int64，使用cumsum形式；E不超过1024。所有输入必须位于同一设备。
- bias和offset均必须为None，即使Tensor内容全为0也不支持传入。
- 支持M为0的空token输入。group_list未覆盖的输出尾部不保证被初始化，不应读取。
- ACLNN转发时固定dequantMode=0、dequantDtype=0（GE FLOAT）、quantMode=0、groupListType=0；参数校验由ACLNN完成。
- CANN需包含旧WeightNZ接口的950 INT8转发实现及WeightNzV2接口。op-plugin不检查V2符号，也不直接调用V2。

### Atlas A2/A3系列产品

- 该接口支持训练、推理场景下使用。
- 确定性计算：默认确定性实现。
- **A8W8**场景：
  - x的尾轴长度（K）不能大于等于65536。
  - N轴长度不能超过10240。
  - bias需传入None。
- **A8W4**场景：
  - x的尾轴长度（K）不能大于等于20000。
  - N轴长度不能超过10240。
  - bias（weightAssistMatrix）为必传参数。
- weight需为FRACTAL\_NZ格式的5维存储，首轴长度为专家数E。
- group\_list的长度需与weight的首轴维度相等；group\_list为cumsum形式，其最后一个值约束了输出的有效数据范围。

## 调用示例

### Ascend 950 INT8 A8W8 weight NZ

```python
import torch
import torch_npu

torch.npu.config.allow_internal_format = True
E, M, K, N = 4, 17, 128, 256
x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="npu")
weight = torch.randint(-128, 128, (E, K, N), dtype=torch.int8, device="npu")
weight_nz = torch_npu.npu_format_cast(weight, 29)
weight_scale = torch.full((E, N), 0.01, dtype=torch.float32, device="npu")
x_scale = torch.full((M,), 0.01, dtype=torch.float32, device="npu")
group_list = torch.tensor([3, 7, 7, 17], dtype=torch.int64, device="npu")
output, output_scale, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(
    x, weight_nz, group_list, weight_scale, x_scale, bias=None, offset=None)
# output: [17, 128], INT8；output_scale: [17], FLOAT32。
# output_offset仅为0维FLOAT32占位Tensor，不读取其数值。
```

### Atlas A2/A3

- 单算子模式调用（A8W8场景）

    ```python
    import torch
    import torch_npu
    import numpy as np

    torch.npu.config.allow_internal_format = True  # 必须开启，否则FRACTAL_NZ格式无法正确传递

    def generate_non_decreasing_sequence(length, upper_limit):
        # 生成随机增量
        random_increments = torch.randint(1, 128, (length,), dtype=torch.int64)  # 避免零增量
        # 累加生成非递减序列
        sequence = torch.cumsum(random_increments, dim=0)
        # 确保最后一个元素不超过上限
        if sequence[-1] > upper_limit:
            # 线性缩放以确保总和不超过上限
            scale_factor = upper_limit / sequence[-1].item()
            sequence = (sequence * scale_factor).to(torch.int64)
            for i in range(1, length):
                if sequence[i] <= sequence[i-1]:
                    sequence[i] = sequence[i-1] + 1
        return sequence

    def gen_input_data(E=16, M=512, K=7168, N=4096):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).npu()
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).npu()
        # 需转成FRACTAL_NZ的5维存储shape（E, N//32, K//16, 16, 32）
        weight_npu = torch_npu.npu_format_cast(weight, 29).view(E, N // 32, K // 16, 16, 32)
        weight_scale = torch.randn(E, N, dtype=torch.float32).npu()
        x_scale = torch.randn(M, dtype=torch.float32).npu()
        group_list = generate_non_decreasing_sequence(E, M).npu()
        output, output_scale, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(x, weight_npu, group_list, weight_scale, x_scale,bias=None, offset=None)
        # 有效数据group_list[-1]
        valid_output = output[:group_list[-1], :]
        valid_output_scale = output_scale[:group_list[-1]]
        return output, output_scale, output_offset

    if __name__ == "__main__":
        output, output_scale, output_offset = gen_input_data()
        print(output.shape, output.dtype)
        print(output_scale.shape, output_scale.dtype)
        print(output_offset.shape) 
    ```

    输出为

    ```text
    torch.Size([512, 2048]) torch.int8
    torch.Size([512]) torch.float32
    torch.Size([])
    ```

- 单算子模式调用（A8W4场景）

    ```python
    import torch
    import torch_npu
    import numpy as np

    torch.npu.config.allow_internal_format = True  # 必须开启，否则FRACTAL_NZ格式无法正确传递

    def generate_non_decreasing_sequence(length, upper_limit):
        # 生成随机增量
        random_increments = torch.randint(1, 128, (length,), dtype=torch.int64)  # 避免零增量
        # 累加生成非递减序列
        sequence = torch.cumsum(random_increments, dim=0)
        # 确保最后一个元素不超过上限
        if sequence[-1] > upper_limit:
            # 线性缩放以确保总和不超过上限
            scale_factor = upper_limit / sequence[-1].item()
            sequence = (sequence * scale_factor).to(torch.int64)
            for i in range(1, length):
                if sequence[i] <= sequence[i-1]:
                    sequence[i] = sequence[i-1] + 1
        return sequence

    def gen_input_data(E=16, M=512, K=7168, N=4096):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).npu()
        # A8W4：权重为int4，沿N轴每2个int4打包进1个int8，故打包后weight的shape为(E, K, N//2)
        weight_pack = torch.randint(-128, 127, (E, K, N // 2), dtype=torch.int8).npu()
        # 转成FRACTAL_NZ的5维存储shape（E, N//64, K//16, 16, 8）
        weight_npu = torch_npu.npu_format_cast(weight_pack, 29).view(torch.int32).view(E, N // 64, K // 16, 16, 8)
        # perchannel场景：weight_scale为int64，shape为(E, N)；pergroup场景shape为(E, k_group_num, N)
        weight_scale = torch.randint(-100, 100, (E, N), dtype=torch.int64).npu()
        x_scale = torch.randn(M, dtype=torch.float32).npu()
        # A8W4场景bias（weightAssistMatrix）为必传参数，shape为(E, N)，数据类型float32
        bias = torch.randn(E, N, dtype=torch.float32).npu()
        group_list = generate_non_decreasing_sequence(E, M).npu()
        output, output_scale, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(
            x, weight_npu, group_list, weight_scale, x_scale, bias=bias, offset=None)
        # 有效数据截至group_list[-1]
        valid_output = output[:group_list[-1], :]
        valid_output_scale = output_scale[:group_list[-1]]
        return output, output_scale, output_offset

    if __name__ == "__main__":
        output, output_scale, output_offset = gen_input_data()
        print(output.shape, output.dtype)
        print(output_scale.shape, output_scale.dtype)
        print(output_offset.shape)
    ```

    输出为

    ```text
    torch.Size([512, 2048]) torch.int8
    torch.Size([512]) torch.float32
    torch.Size([])
    ```
