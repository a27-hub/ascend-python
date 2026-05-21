# torch_npu.npu_quant_mm_reduce_scatter

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>     | √  |

## 功能说明

- API功能：TP切分场景下，实现matmul和reduce\_scatter的融合，融合算子内部实现计算和通信流水并行。

- 计算公式：

    - 情形1：如果$x_1$和$x_2$数据类型为`float16`、`bfloat16`时，入参$x_1$、$x_2$、`bias`进行matmul计算后，进行reduce\_scatter通信。
        $$
        y = ReduceScatter(x_1 \mathbin{@} x_2 + bias)
        $$

    - 情形2：对于$x_1$和$x_2$数据类型为`float8_e4m3fn`、`float8_e5m2`和`hifloat8`的pertoken场景，且不输出`amax_out`，入参$x_1$、$x_2$进行matmul计算和dequant计算后，进行reduce\_scatter通信。
        $$
        y = ReduceScatter(x1\_scale \times x2\_scale \times (x_1 \mathbin{@} x_2))
        $$

    - 情形3：对于$x_1$和$x_2$数据类型为`float8_e4m3fn`、`float8_e5m2`和`hifloat8`的perblock场景，且不输出`amax_out`，当$x_2$不转置时，假设$x_1$的shape为$(m, k)$，$x_2$的shape为$(k, n)$，则`x1_scale`的shape为$(ceildiv(m, 128), ceildiv(k, 128))$，`x2_scale`的shape为$(ceildiv(k, 128), ceildiv(n, 128))$；当$x_2$转置时，假设$x_1$的shape为$(m, k)$，$x_2$的shape为$(n, k)$，则`x1_scale`的shape为$(ceildiv(m, 128), ceildiv(k, 128))$，`x2_scale`的shape为$(ceildiv(n, 128), ceildiv(k, 128))$。入参$x_1$、$x_2$进行matmul计算和dequant计算后，再进行reduce\_scatter通信。$x_2$和`x2_scale`的转置保持一致。
        $$
        y = ReduceScatter(x1\_scale \times x2\_scale \times (x_1 \mathbin{@} x_2))
        $$

    - 情形4：对于$x_1$和$x_2$数据类型为`float8_e4m3fn`、`float8_e5m2`的pertoken量化和mxfp8量化场景，且不输出`amax_out`，当$x_1$的shape为$(m，k)$，$x_2$的shape为$(n，k)$时，`x1_scale`的shape为$(m，ceildiv(k，64)，2)$，`x2_scale`的shape为$(n，ceildiv(k，64)，2)$，$x_1$不转置，$x_2$转置时，入参$x_1$、$x_2$进行matmul计算和dequant计算后，再进行ReduceScatter通信。mx量化仅支持$x_2$、`x2_scale`转置场景。
        $$
        y = ReduceScatter(x1\_scale \times x2\_scale \times (x_1 \mathbin{@} x_2))
        $$

## 函数原型

```python
torch_npu.npu_quant_mm_reduce_scatter(self, x2, hcom, world_size, *, reduce_op='sum', bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None) -> (Tensor, Tensor)
```

## 参数说明

- **self** (`Tensor`)：必选参数，表示matmul计算中的左矩阵，对应公式中的$x_1$。不支持非连续的tensor，数据格式支持$ND$，数据类型支持`float16`、`bfloat16`、`float8_e4m3fn`、`float8_e5m2`、`hifloat8`，输入shape支持2维，形如$(m, k)$、$(k, n)$，轴满足matmul算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
- **x2** (`Tensor`)：必选参数，表示matmul计算中的右矩阵，数据格式支持$ND$，数据类型支持`float16`、`bfloat16`、`float8_e4m3fn`、`float8_e5m2`、`hifloat8`，输入shape维度需要和`self`保持一致，数据格式支持$ND$。
- **hcom** (`str`)：必选参数，表示通信域handle名，通过get\_hccl\_comm\_name接口获取。
- **world\_size** (`int`)：必选参数，表示通信域内的rank总数，支持2、4、8、16、32、64卡。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **reduce\_op** (`str`)：可选参数，表示reduce操作类型，默认值'sum'。当前仅支持'sum'。
- **bias** (`Tensor`)：可选参数，数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$格式。数据类型需要和`self`保持一致。`bias`仅支持一维，且维度大小与output的第1维大小相同。情形1的`bias`数据类型和输入类型一致，情形2、情形4的`bias`数据类型为`float32`，情形3的`bias`输入只能为None。
- **x1\_scale** (`Tensor`)：可选参数，数据格式支持$ND$，数据类型支持`float32`、`float8_e8m0fnu`。当`self`和`x2`数据类型为`float16`、`bfloat16`时，仅支持输入为nullptr。在pertoken场景下，shape为$(1)$。在perblock场景下，shape为$(ceildiv(m, 128), ceildiv(k, 128))$。在pertoken和perblock场景下，数据类型支持`float32`。在mx量化场景下，数据类型为`float8_e8m0fnu`，shape为$(m, ceilDiv(k, 64), 2)$。
- **x2\_scale** (`Tensor`)：可选参数，数据格式支持$ND$，数据类型支持`float32`、`float8_e8m0fnu`。当`self`和`x2`数据类型为`float16`、`bfloat16`时，仅支持输入为nullptr。在pertoken场景下，shape为$(1)$。在perblock场景下，shape为$(ceildiv(k, 128), ceildiv(n, 128))$。在pertoken和perblock场景下，数据类型支持`float32`。在mx量化场景下，数据类型为`float8_e8m0fnu`，shape为$(n, ceilDiv(k, 64), 2)$。
- **quant\_scale** (`Tensor`)：可选参数，数据格式支持$ND$，数据类型支持`float32`。
- **block\_size** (`int`)：可选参数，表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化，`block_size`由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为block\_size= blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0。当前版本只支持`block_size`输入为0。
- **comm\_turn** (`int`)：可选参数，表示rank间通信切分粒度，默认值0，表示框架内置的切分方式。当前版本仅支持输入0。
- **group\_sizes** (`List[int]`)：可选参数，表示反量化中`x1_scale`/`x2_scale`输入的一个数在其所在的对应维度方向上可以用于该方向`self`/`x2`输入的多少个数的反量化，`group_sizes`为\[groupSizeM，groupSizeN，groupSizeK\]。groupSizeM、groupSizeN、groupSizeK分别表示一个反量化系数在各个维度对应的数的个数。
- **amax\_output** (`bool`)：可选参数，代表是否输出MM计算的最大值结果，默认值为False，当前版本只支持False。
- **y\_dtype** (`int`)：可选参数，代表输出数据类型（例如：torch.float16），支持`float16`、`bfloat16`、`float32`。
- **x1\_dtype** (`int`)：可选参数，代表输入的左矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x2\_dtype** (`int`)：可选参数，代表输入的右矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x1\_scale\_dtype** (`int`)：可选参数，代表输入的左矩阵的量化系数数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x2\_scale\_dtype** (`int`)：可选参数，代表输入的右矩阵的量化系数数据类型，PyTorch原生不支持的数据类型需通过该参数指定。

## 返回值说明

- **y** (`Tensor`)：reduce\_scatter和matmul计算后输出的结果，如果`self`数据类型为`float16`或`bfloat16`，输出数据类型和`self`保持一致；如果`self`类型为`float8_e4m3fn`、`float8_e5m2`、`hifloat8`，输出数据类型由`y_dtype`指定，shape为$(m / world\_size, n)$，数据格式支持$ND$。
- **amax\_out** (`Tensor`)：当前只支持输出为None，数据格式支持$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- **通信引擎约束**：仅支持CCU通信。
- 输出output为2维，其维度为$(m / world\_size, n)$，`world_size`为卡数。
- 当`self`、`x2`的数据类型为`float16`、`bfloat16`时，output计算输出数据类型和`self`、`x2`保持一致，不支持`amax_out`的输出。
- 当`self`、`x2`的数据类型为`float8_e4m3fn`、`float8_e5m2`、`hifloat8`时，output输出数据类型支持`float16`、`float32`、`bfloat16`。支持`bias`输入为`float32`。
- 当`self`、`x2`的数据类型为`float16`、`bfloat16`、`hifloat8`时，`self`和`x2`数据类型需要保持一致。
- 当`self`、`x2`数据类型为`float8_e4m3fn`、`float8_e5m2`时，`self`和`x2`数据类型可以为其中一种。
- 当`self`、`x2`数据类型为`float16`、`bfloat16`、`hifloat8`、`float8_e4m3fn`、`float8_e5m2`时，`x2`矩阵支持转置/不转置场景，`self`矩阵只支持不转置场景。
- <term>Ascend 950PR/Ascend 950DT</term>：ReduceScatter集合通信数据总量不能超过16×256MB，集合通信数据总量计算方式为：m × n × sizeof(output\_dtype)。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。
- **group\_sizes**：
    - 仅当`x1_scale`和`x2_scale`输入都是2维或3维时，`group_sizes`取值有效，其他场景需传入\[0, 0, 0\]。
    - groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据`self`/`x2`/`x1_scale`/`x2_scale`输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。
    - 设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与`self` shape中的m一致，scaleM与`x1_scale` shape中的m一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与`self` shape中的k一致，scaleK与`x1_scale` shape中的k一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与`x2` shape中的n一致，scaleN与`x2_scale` shape中的n一致。
    - 如果满足重新设置条件，一般情况下，当`x1_scale`、`x2_scale`输入都是2维，且数据类型都为`float32`时，\[groupSizeM，groupSizeN，groupSizeK\]取值组合会推导为\[128, 128, 128\]；当`x1_scale`、`x2_scale`输入都是3维，且数据类型都为`float8_e8m0fnu`时，\[groupSizeM, groupSizeN, groupSizeK\]取值组合会推导为\[1, 1, 32\]。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)

        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output, amax_out = torch_npu.npu_quant_mm_reduce_scatter(input_, weight, hcomm_info, world_size)

    if __name__ == "__main__":
        worksize = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(run_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    class MM_REDUCESCATTER_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, reduce_op):
            output, amax_out = torch_npu.npu_quant_mm_reduce_scatter(input, weight, hcomm_info, world_size, reduce_op=reduce_op)
            return output

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

    def get_graph(input, weight, hcomm_info, world_size):
        model = MM_REDUCESCATTER_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, reduce_op="sum")
        return model_output

    def run_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        input = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = get_graph(input, weight, hcomm_info, world_size)
        print("output:", output)

    if __name__ == "__main__":
        worksize = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```