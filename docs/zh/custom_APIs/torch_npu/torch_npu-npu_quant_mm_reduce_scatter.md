# torch_npu.npu_quant_mm_reduce_scatter

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- **API功能**：TP切分场景下，实现matmul和reduce\_scatter的融合，融合算子内部实现计算和通信流水并行。

- 计算公式：
    - 情形1：

        如果`x1`和`x2`数据类型为`torch.float16`/`torch.bfloat16`时，入参`x1`、`x2`、`bias`进行matmul计算后，进行reducescatter通信。
        $$
        output = ReduceScatter(x1 \mathbin{@} x2 + bias_{optional})
        $$

    - 情形2：

        对于`x1`和`x2`数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`和`torch_npu.hifloat8`的pertensor场景，且不输出amaxOut，入参`x1`、`x2`进行matmul计算和dequant计算后，进行reduce\_scatter通信。
        $$
        output = ReduceScatter((x1\_scale * x2\_scale) * (x1 \mathbin{@} x2 + bias_{optional}))
        $$

    - 情形3：

        对于`x1`和`x2`数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`和`torch_npu.hifloat8`的perblock场景，且不输出amaxOut，当`x2`不转置时，假设`x1`的shape为\(m, k\)，`x2`的shape为\(k, n\)，则`x1_scale`的shape为\(ceilDiv\(m, 128\), ceilDiv\(k, 128\)\)，`x2_scale`的shape为\(ceilDiv\(k, 128\), ceilDiv\(n, 128\)\)；当`x2`转置时，假设`x1`的shape为\(m, k\)，`x2`的shape为\(n, k\)，则`x1_scale`的shape为\(ceilDiv\(m, 128\), ceilDiv\(k, 128\)\)，`x2_scale`的shape为\(ceilDiv\(n, 128\), ceilDiv\(k, 128\)\)。入参`x1`、`x2`进行matmul计算和dequant计算后，再进行reduce\_scatter通信。`x2`和x2Scale的转置保持一致。
        $$
        output = ReduceScatter(\sum_{0}^{\lfloor k / blockSize=128 \rfloor} ((x1_{pr} \mathbin{@} x2_{rq}) * (x1\_scale_{pr} * x2\_scale_{rq})))
        $$

    - 情形4：

        对于`x1`和`x2`数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.float4_e2m1`的pertensor量化和mx量化场景，且不输出amaxOut，当`x1`的shape为\(m, k\)，`x2`的shape为\(n, k\)时，x1Scale的shape为\(m, ceilDiv\(k, 64\), 2\)，x2Scale的shape为\(n, ceilDiv\(k, 64\), 2\)，`x1`不转置，`x2`转置时，入参`x1`、`x2`进行matmul计算和dequant计算后，再进行ReduceScatter通信。mx量化仅支持`x2`、x2Scale转置场景。
        $$
        output = ReduceScatter(\sum_{0}^{\lfloor k / blockSize=32 \rfloor} (x1_{pr} \mathbin{@} x2_{rq} * (x1\_scale_{pr} * x2\_scale_{rq})))
        $$

## 函数原型

```python
torch_npu.npu_quant_mm_reduce_scatter(x1, x2, hcom, world_size, *, reduce_op='sum', bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## 参数说明

- **x1**（`Tensor`）：**必选参数**，表示matmul计算中的左矩阵，对应公式中的$x1$，不支持非连续的tensor。数据格式支持$ND$，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1`。输入shape支持2维，形如\(m, k\)，轴满足matmul算子入参要求，k轴与`x2`的k轴相等，且k轴取值范围为\[256, 65535\)。
- **x2**（`Tensor`）：**必选参数**，表示matmul计算中的右矩阵，数据格式支持$ND$，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1`，输入shape维度需要和`x1`保持一致。MX量化场景下`x2`仅支持转置输入，且k必须为偶数。
- **hcom**（`str`）：**必选参数**，表示通信域handle名，通过get\_hccl\_comm\_name接口获取。
- **world\_size**（`int`）：**必选参数**，表示通信域内的rank总数，支持2、4、8、16、32、64卡，CCU仅支持单机UB域内互联，AI CPU可支持跨机UB域内互联。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **reduce\_op**（`str`）：**可选参数**，表示reduce操作类型，默认值'sum'。当前仅支持'sum'。
- **bias**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`。`bias`仅支持一维，且维度大小与`y`的第1维大小相同。**情形1的`bias`数据类型和输入类型一致，情形2、情形4的`bias`数据类型为`torch.float32`，情形3的`bias`输入只能为None。**
- **x1\_scale**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`torch.float32`、`torch_npu.float8_e8m0`。当`x1`和`x2`数据类型为`torch.float16`、`torch.bfloat16`时，仅支持输入为None。在pertensor场景下，shape为\[1\]。在perblock场景下，shape为\[ceilDiv\(m, 128\), ceilDiv\(k, 128\)\]。在pertensor和perblock场景下，数据类型支持`torch.float32`。在mx量化场景下，数据类型为`torch_npu.float8_e8m0`，shape为\(m, ceilDiv\(k, 64\), 2\)。
- **x2\_scale**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`torch.float32`、`torch_npu.float8_e8m0`。当`x1`和`x2`数据类型为`torch.float16`、`torch.bfloat16`时，仅支持输入为None。在pertensor场景下，shape为\[1\]。在perblock场景下，shape为\[ceilDiv\(k, 128\), ceilDiv\(n, 128\)\]。在pertensor和perblock场景下，数据类型支持`torch.float32`。在mx场景下，数据类型为`torch_npu.float8_e8m0`，shape为\(n, ceilDiv\(k, 64\), 2\)，仅支持转置输入。
- **quant\_scale**（`Tensor`）：**可选参数**，数据格式支持$ND$，数据类型支持`torch.float32`。
- **block\_size**（`int`）：**可选参数**，表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化。`block_size`由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为block\_size = blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0。**当前版本仅支持输入0。**
- **comm\_turn**（`int`）：**可选参数**，表示rank间通信切分粒度，默认值0，表示框架内置的切分方式。**当前版本仅支持输入0。**
- **group\_sizes**（`List[int]`）：**可选参数**，表示反量化中`x1_scale`/`x2_scale`输入的一个数在其所在的对应维度方向上可以用于该方向`x1`/`x2`输入的多少个数的反量化。`group_sizes`为\[groupSizeM，groupSizeN，groupSizeK\]。groupSizeM、groupSizeN、groupSizeK分别表示一个反量化系数在各个维度对应的数的个数。
- **amax\_output**（`bool`）：**可选参数**，代表是否输出MM计算的最大值结果，默认值为False。**当前版本仅支持False。**
- **y\_dtype**（`int`）：**可选参数**，代表输出数据类型（例如torch.float16），支持`torch.float16`、`torch.bfloat16`、`torch.float32`。
- **x1\_dtype**（`int`）：**可选参数**，代表输入的左矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x2\_dtype**（`int`）：**可选参数**，代表输入的右矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x1\_scale\_dtype**（`int`）：**可选参数**，代表输入左矩阵量化系数的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **x2\_scale\_dtype**（`int`）：**可选参数**，代表输入右矩阵的量化系数的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
- **comm\_mode**（`str`）：**可选参数**，表示通信模式，支持`ai_cpu`、`ccu`。当输入为`None`时，默认选择AI CPU通信。

## 返回值说明

- **y**（`Tensor`）：matmul计算和reducescatter通信后输出的结果。如果`x1`数据类型为`torch.float16`或`torch.bfloat16`，输出数据类型和`x1`保持一致；如果`x1`数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`，输出数据类型由`y_dtype`指定。shape为\[m / world\_size, n\]，数据格式支持$ND$。
- **amaxout**（`Tensor`）：当前只支持输出为None，数据格式支持$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- 输出`y`为2维，其维度为\(m / world\_size, n\)，world\_size为卡数。
- 当`x1`、`x2`的数据类型为`torch.float16`、`torch.bfloat16`时，`y`计算输出数据类型和`x1`、`x2`保持一致，不支持amaxOut的输出。
- 当`x1`、`x2`的数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch_npu.hifloat8`时，`y`输出数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`，必须指定`y_dtype`。支持`bias`输入为`torch.float32`。
- 当`x1`、`x2`的数据类型为`torch.float16`、`torch.bfloat16`、`torch_npu.hifloat8`、`torch_npu.float4_e2m1`时，`x1`和`x2`数据类型需要保持一致。
- 当`x1`、`x2`数据类型为`torch.float8_e4m3fn`、`torch.float8_e5m2`时，`x1`和`x2`数据类型可以为其中一种。
- 当`x1`、`x2`数据类型为`torch.float16`、`torch.bfloat16`、`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`时，`x2`矩阵支持转置/不转置场景，`x1`矩阵只支持不转置场景。
- mx量化场景下，`x2`/x2Scale仅支持转置输入。
- mx量化场景下，且`x1`和`x2`输入为`torch_npu.float4_e2m1`（MXFP4量化）时，k必须是偶数。
- `x1_dtype`、`x2_dtype`、`x1_scale_dtype`、`x2_scale_dtype`用于指定PyTorch原生不支持的数据类型（例如float8\_e8m0、float4\_e2m1），当该系列参数输入时，优先生效该系列参数指定的数据类型，`x1`、`x2`、`x1_scale`、`x2_scale`的原生数据类型被覆盖。但需要注意，原生数据类型和指定数据类型需要有相同的比特位长（float4\_e2m1等效于8bit长度），否则指定数据类型失败且报错。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - ReduceScatter集合通信数据总量不能超过16\*256MB，集合通信数据总量计算方式为：m \* n \* sizeof\(y\_dtype\)。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。
  - `group_sizes`：
      - 仅当`x1_scale`和`x2_scale`输入都是2维或3维时，`group_sizes`取值有效，其他场景需传入\[0, 0, 0\]。
      - groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据`x1`/`x2`/`x1_scale`/`x2_scale`输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。

          设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与`x1` shape中的m一致，scaleM与`x1_scale` shape中的m一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与`x1` shape中的k一致，scaleK与`x1_scale` shape中的k一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与`x2` shape中的n一致，scaleN与`x2_scale` shape中的n一致。

      - 如果满足重新设置条件，一般情况下，当`x1_scale`、`x2_scale`输入都是2维，且数据类型都为`torch.float32`时，\[groupSizeM，groupSizeN，groupSizeK\]取值组合会推导为\[128, 128, 128\]；当`x1_scale`、`x2_scale`输入都是3维，且数据类型都为`torch_npu.float8_e8m0`时，\[groupSizeM, groupSizeN, groupSizeK\]取值组合会推导为\[1, 1, 32\]。

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
            output, amax_out = torch_npu.npu_quant_mm_reduce_scatter(input, weight, hcomm_info, world_size,reduce_op=reduce_op)
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
