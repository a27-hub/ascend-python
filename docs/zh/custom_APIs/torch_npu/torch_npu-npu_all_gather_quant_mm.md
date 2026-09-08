# torch_npu.npu_all_gather_quant_mm

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- **API功能**：AllGatherMatmulV2算子是AllGather算子和Matmul算子的融合，实现通信和计算流水并行。

- 计算公式：
    - 情形1：

        当`x1`和`x2`输入为`float16`和`bfloat16`类型时，入参`x1`进行allgather后，对`x1`、`x2`进行matmul计算。
        $$
        output = allgather(x1) \mathbin{@} x2 + bias
        $$
        $$
        gatherOut = allgather(x1)
        $$

    - 情形2：

        当`x1`和`x2`数据类型为`float8_e4m3fn`/`float8_e5m2`/`hifloat8`时，不输出amaxOut，入参`x1`进行allgather后，对`x1`、`x2`进行matmul计算，然后进行dequant操作。
        $$
        output = (x1\_scale * x2\_scale) * (allgather(x1) \mathbin{@} x2 + bias)
        $$
        $$
        gatherOut = allgather(x1)
        $$

    - 情形3：

        当`x1`和`x2`数据类型为`float8_e4m3fn`/`float8_e5m2`/`hifloat8`时，且groupSize取值为有效值，入参`x1`进行allgather后，对`x1`、`x2`进行perblock量化matmul计算，然后进行dequant操作。
        $$
        output[r(i), r(j)] = \sum_{k=1}^{\frac{K}{groupSizeK}} x1\_scale[i, k] * x2\_scale[k, j] * (allgather(x1)[r(i), r(j)] \mathbin{@} x2[r(k), r(j)])
        $$
        $$
        r(z) = (groupSizeK * (z-1) + 1) : (groupSizeK * z)
        $$
        $$
        output =
        \begin{bmatrix}
        output[r(1), r(1)] & \cdots & output[r(1), r(\frac{N}{groupSizeN})] \\
        \vdots & \ddots & \vdots \\
        output[r(\frac{M}{groupSizeM}), r(1)] & \cdots & output[r(\frac{M}{groupSizeM}), r(\frac{N}{groupSizeN})]
        \end{bmatrix}
        $$
        $$
        gatherOut = allgather(x1)
        $$

    - 情形4：

        如果`x1`和`x2`数据类型为`float8_e4m3fn`/`float8_e5m2`/`float4_e2m1`，`x1` shape为\(m, k\)，`x2` shape为\(n, k\)，x1Scale shape为\(m, ceilDiv\(k, 64\), 2\)，x2Scale shape为\(n, ceilDiv\(k, 64\), 2\)，`x1`不转置，`x2`转置，x1Scale不转置，x2Scale转置，入参`x1`进行allgather后，对`x1`、`x2`进行matmul计算，然后进行dequant操作。
        $$
        gatherOut = append(allgather(x1), allgather(x1\_scale))
        $$
        $$
        output = \sum_{0}^{\lfloor k / blockSize=32 \rfloor} (allgather(x1))_{pr} \mathbin{@} x2_{rq} * (allgather(x1\_scale)_{pr} * x2\_scale_{rq})
        $$

## 函数原型

```python
torch_npu.npu_all_gather_quant_mm(x1, x2, hcom, world_size, *, bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, gather_index=0, gather_output=True, comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None, comm_mode=None) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x1**（`Tensor`）：**必选参数**，表示matmul计算中的左矩阵，对应公式中的$x1$，不支持非连续的tensor。数据格式支持$ND$，数据类型支持`float16`、`bfloat16`、`float8_e4m3fn`、`float8_e5m2`、`hifloat8`、`float4_e2m1`。输入shape支持2维，形如\(m, k\)，轴满足matmul算子入参要求，k轴与`x2`的k轴相等，且k轴取值范围为\[256, 65535\)。
- **x2**（`Tensor`）：**必选参数**，表示matmul计算中的右矩阵，数据格式支持$ND$，数据类型支持`float16`、`bfloat16`、`float8_e4m3fn`、`float8_e5m2`、`hifloat8`、`float4_e2m1`，输入shape维度需要和`x1`保持一致。
- **hcom**（`str`）：**必选参数**，表示通信域handle名，通过get\_hccl\_comm\_name接口获取。
- **world\_size**（`int`）：**必选参数**，表示通信域内的rank总数，支持2、4、8、16、32、64卡，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **bias**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`float16`、`bfloat16`、`float32`。`bias`仅支持一维，且维度大小与`output`的第1维大小相同。情形1中`bias`数据类型和输入类型一致，情形2中`bias`数据类型为`float32`，情形3中`bias`输入只能为None，情形4中`bias`数据类型为`float32`。
- **x1\_scale**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`float32`、`float8_e8m0`。当`x1`和`x2`数据类型为`float16`、`bfloat16`时，仅支持输入为None。在pertensor场景下，shape为\[1\]。在perblock场景下，shape为\[ceilDiv\(m, 128\), ceilDiv\(k, 128\)\]。在pertensor和perblock场景下，数据类型支持`float32`。在mx量化场景下，数据类型为`float8_e8m0`，shape为\(m, ceilDiv\(k, 64\), 2\)。
- **x2\_scale**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`float32`、`float8_e8m0`。当`x1`和`x2`数据类型为`float16`、`bfloat16`时，仅支持输入为None。在pertensor场景下，shape为\[1\]。在perblock场景下，shape为\[ceilDiv\(k, 128\), ceilDiv\(n, 128\)\]。在pertensor和perblock场景下，数据类型支持`float32`。在mx场景下，数据类型为`float8_e8m0`，shape为\(n, ceilDiv\(k, 64\), 2\)，仅支持转置场景。
- **quant\_scale**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`float32`。
- **block\_size**（`int`）：**可选参数**，表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化。`block_size`由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为block\_size = blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0。**当前版本仅支持输入0。**
- **gather\_index**（`int`）：**可选参数**，表示gather操作对象，0（默认值）表示对`x1`做gather，1表示对`x2`做gather。**当前版本仅支持输入0。**
- **gather\_output**（`bool`）：**可选参数**，表示是否需要gather输出。默认值True。
- **comm\_turn**（`int`）：**可选参数**，表示rank间通信切分粒度，0（默认值）表示框架内置切分方式。**当前版本仅支持输入0。**
- **group\_sizes**（`List[int]`）：**可选参数**，表示反量化中x1Scale/x2Scale输入的一个数在其所在维度方向上可用于`x1`/`x2`输入的个数的反量化。`group_sizes`的list组成为\[groupSizeM，groupSizeN，groupSizeK\]，其中groupSizeM、groupSizeN、groupSizeK分别表示一个反量化系数在各个维度对应的数的个数。
    - 仅当`x1_scale`和`x2_scale`输入是2维或3维时，`group_sizes`取值有效，其他场景需传入\[0, 0, 0\]。
    - 对于groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据`x1`/`x2`/`x1_scale`/`x2_scale`输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与`x1` shape中的m一致，scaleM与`x1_scale` shape中的m一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与`x1` shape中的k一致，scaleK与`x1_scale` shape中的k一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与`x2` shape中的n一致，scaleN与`x2_scale` shape中的n一致。
    - 计算公式：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。
    - 如果满足重新设置的条件，当`x1_scale`、`x2_scale`输入为2维且数据类型都为`float32`时，\[groupSizeM，groupSizeN，groupSizeK\]取值组合会推导为\[128, 128, 128\]；当`x1_scale`、`x2_scale`输入都是3维，且数据类型都为`float8_e8m0`时，\[groupSizeM, groupSizeN, groupSizeK\]取值组合会推导为\[1, 1, 32\]。

- **amax\_output**（`bool`）：**可选参数**，代表是否输出MM计算的最大值结果，默认为False。**当前版本仅支持False。**
- **y\_dtype**（`int`）：**可选参数**，代表输出数据类型（例如torch.float16），数据类型支持`float16`、`float32`、`bfloat16`。当输入`x1`数据类型为`float16`或`bfloat16`时不需要输入，其他场景需要输入。
- **x1\_dtype**（`int`）：**可选参数**，代表输入的左矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x2\_dtype**（`int`）：**可选参数**，代表输入的右矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x1\_scale\_dtype**（`int`）：**可选参数**，代表输入左矩阵量化系数的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x2\_scale\_dtype**（`int`）：**可选参数**，代表输入右矩阵的量化系数的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **comm\_mode**（`str`）：**可选参数**，表示通信模式，支持`ai_cpu`、`ccu`、`None`（默认值）。当为`None`时，走AI CPU通信。

## 返回值说明

- **y**（`Tensor`）：allgather和matmul计算后输出的结果。如果`x1`数据类型为`float16`、`bfloat16`，输出数据类型和`x1`保持一致；如果`x1`数据类型为`float8_e4m3fn`、`float8_e5m2`、`hifloat8`、`float4_e2m1`，输出数据类型由`y_dtype`指定。shape为\[world\_size \* m, n\]，数据格式支持$ND$。
- **gatherout**（`Tensor`）：allgather的结果，作为matmul计算的左矩阵。数据类型和`x1`保持一致。当`gather_output`为True时，shape为\(world\_size \* m, k\)，数据格式支持$ND$。
- **amaxout**（`Tensor`）：当前只支持输出为None，数据格式支持$ND$。

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- `x1`的M轴在情形3必须满足为128的整数倍。
- 输出`y`为2维，其维度为\(m \* world\_size, n\)，world\_size为卡数。
- 输出`gatherout`为2维，其维度为\(m \* world\_size, k\)，world\_size为卡数。
- 当`x1`、`x2`的数据类型为`float16`、`bfloat16`时，输出数据类型和`x1`、`x2`保持一致，且不支持amaxOut的输出。
- 当`x1`、`x2`的数据类型为`float8_e4m3fn`、`float8_e5m2`、`hifloat8`时，输出数据类型支持`float16`、`float32`、`bfloat16`。支持`bias`输入为`float32`。
- 当`x1`、`x2`的数据类型为`float16`、`bfloat16`、`hifloat8`时，`x1`和`x2`数据类型需要保持一致。
- 当`x1`、`x2`数据类型为`float8_e4m3fn`、`float8_e5m2`时，`x1`和`x2`数据类型可以为其中一种。
- 当`x1`、`x2`数据类型为`float16`、`bfloat16`、`hifloat8`、`float8_e4m3fn`、`float8_e5m2`时，`x2`矩阵支持转置/不转置场景，`x1`矩阵只支持不转置场景。
- 当`x1`、`x2`数据类型为`float4_e2m1`时，`x1`和`x2`数据类型需要保持一致，`x1`矩阵只支持不转置场景，`x2`矩阵只支持转置场景，k轴需要为偶数。
- <term>Ascend 950PR/Ascend 950DT</term>：allgather\(x1\)集合通信数据总量不能超过63\*256MB，集合通信数据总量计算方式为：m \* k \* sizeof\(x1\_dtype\) \* 卡数。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)

        tensor_allgather_shape = x1_shape
        single_shape = [x1_shape[0] // world_size, x1_shape[1]]

        input_ = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output, gather_out, amax_out = torch_npu.npu_all_gather_quant_mm(input_, weight, hcomm_info, world_size)

    if __name__ == "__main__":
        worksize = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(run_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    class ALLGATHER_MM_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, gather_output):
            output, gatherout, amax_out = torch_npu.npu_all_gather_quant_mm(input, weight, hcomm_info, world_size,gather_output=gather_output)
            return output, gatherout
    def define_model(model, graph_type):
        import torchair
        if graph_type == 1:  # 传统入图模式，静态shape+在线编译场景
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN入图模式，动态shape+二进制
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print("Error type")
        return model
    def get_graph(input, weight, hcomm_info, world_size, gather_output):
        model = ALLGATHER_MM_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, gather_output=gather_output)
        output_npu = model_output[0]
        gather_output_npu = model_output[1]
        return output_npu, gather_output_npu
    def run_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        single_shape = [x1_shape[0] // world_size, x1_shape[1]]
        input = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        is_gather_out = True
        output, gather_out = get_graph(input, weight, hcomm_info, world_size, is_gather_out)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```
