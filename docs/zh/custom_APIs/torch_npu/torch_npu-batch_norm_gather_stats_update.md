# torch_npu.batch_norm_gather_stats_update

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

## 功能说明

- API功能：在跨 device 的同步 Batch Normalization（SyncBatchNorm）场景下，收集所有 device 的通道特征和、特征平方和与样本计数，计算全局的批均值（batch mean）与批标准差倒数（batch invstd），并以指数平滑方式更新运行均值/运行方差（running mean / running variance）。
- 计算公式：

  $$
  batchMean = \frac{\sum^N_{i=0}{totalSum[i]}}{\sum^N_{i=0}{sampleCount[i]}}
  $$

  $$
  batchVar = \frac{\sum^N_{i=0}{totalSquareSum[i]}}{\sum^N_{i=0}{sampleCount[i]}} - batchMean^2
  $$

  $$
  batchInvstd = \frac{1}{\sqrt{batchVar + \epsilon}}
  $$

  $$
  runningMean = runningMean \cdot (1 - momentum) + momentum \cdot batchMean
  $$

  $$
  runningVar = runningVar \cdot (1 - momentum) + momentum \cdot \left(batchVar \cdot \frac{\sum^N_{i=0}{sampleCount[i]}}{\sum^N_{i=0}{sampleCount[i]} - 1}\right)
  $$

> [!NOTE]
> `running_mean`、`running_var` 作为入参传入后由底层算子**原地更新**；本接口仅返回 `(batch_mean, batch_invstd)`。当两者为 `None`（未定义）时不进行 running 统计的更新。

## 函数原型

```
torch_npu.batch_norm_gather_stats_update(input, sum_all, square_sum_all, running_mean, running_var, momentum, eps, counts) -> (Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选参数，原始输入张量，数据类型支持`float16`、`bfloat16`、`float32`。仅用于确定通道数 `C = input.size(1)` 与输出 shape/dtype 推导，不参与 aclnn 计算。
- **sum_all** (`Tensor`)：必选参数，各 device 的通道特征和（对应公式中的 `totalSum`），数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持$ND$，支持非连续的Tensor。shape 为 `(N, C)`，其中 `N` 为 device 数。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`float32`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`float16`、`bfloat16`、`float32`。
- **square_sum_all** (`Tensor`)：必选参数，各 device 的通道特征平方和（对应公式中的 `totalSquareSum`），数据类型与格式要求同 `sum_all`，shape 必须与 `sum_all` 完全一致。
- **running_mean** (`Tensor`, 可选)：运行均值（对应公式中的 `runningMean`），1D，shape 为 `(C,)`，数据类型支持`float16`、`bfloat16`、`float32`。由算子**原地更新**；传 `None` 时不更新。支持空 Tensor。
- **running_var** (`Tensor`, 可选)：运行方差（对应公式中的 `runningVar`），1D，shape 为 `(C,)`，数据类型支持`float16`、`bfloat16`、`float32`。由算子**原地更新**；传 `None` 时不更新。支持空 Tensor。
- **momentum** (`float`)：必选参数，`running_mean`/`running_var` 的指数平滑参数。
- **eps** (`float`)：必选参数，用于防止产生除 0 的偏移。
- **counts** (`Tensor`)：必选参数，各 device 的样本计数（对应公式中的 `sampleCount`），1D，shape 为 `(N,)`（与 `sum_all` 第一维一致），数据类型仅支持`int32`，数据格式支持$ND$。

## 返回值说明

返回值为 `(batch_mean, batch_invstd)` 二元组：

- **batch_mean** (`Tensor`)：全局批均值，1D，shape 为 `(C,)`，数据类型与 `sum_all` 保持一致。
- **batch_invstd** (`Tensor`)：全局批标准差倒数，1D，shape 为 `(C,)`，数据类型与 `sum_all` 保持一致。

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持图模式。
- 该算子为默认确定性实现。
- `sum_all` 第一维（device 数）必须大于 0，且为 2D。
- `square_sum_all` 的 shape 必须与 `sum_all` 完全相同。
- `counts` 的第一维必须与 `sum_all` 第一维一致，数据类型必须为 `int32`。
- `running_mean`、`running_var` 为 1D，shape 与 `sum_all` 第二维（通道数 `C`）一致；支持空 Tensor。
- `bfloat16` 仅 <term>Ascend 950PR/Ascend 950DT</term> 支持；<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 的 `sum_all`、`square_sum_all`、`running_mean`、`running_var`、`batch_mean`、`batch_invstd` 不支持 `bfloat16`。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> # input 仅用于确定通道数 C = input.size(1)，取值不参与计算
    >>> input = torch.randn((4, 8), dtype=torch.float32, device='npu')
    >>> sum_all = torch.randn((2, 8), dtype=torch.float32, device='npu')        # 2 个 device, C=8
    >>> square_sum_all = torch.randn((2, 8), dtype=torch.float32, device='npu')
    >>> running_mean = torch.zeros(8, dtype=torch.float32, device='npu')
    >>> running_var = torch.ones(8, dtype=torch.float32, device='npu')
    >>> counts = torch.tensor([16, 16], dtype=torch.int32, device='npu')
    >>> batch_mean, batch_invstd = torch_npu.batch_norm_gather_stats_update(
    ...     input, sum_all, square_sum_all, running_mean, running_var, 0.1, 1e-5, counts)
    >>> batch_mean.shape
    torch.Size([8])
    >>> batch_invstd.shape
    torch.Size([8])
    >>> # running_mean / running_var 已被原地更新
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
        def forward(self, input, sum_all, square_sum_all, running_mean, running_var, counts):
            return torch_npu.batch_norm_gather_stats_update(
                input, sum_all, square_sum_all, running_mean, running_var, 0.1, 1e-5, counts)

    input = torch.randn((4, 8), dtype=torch.float32, device='npu')
    sum_all = torch.randn((2, 8), dtype=torch.float32, device='npu')
    square_sum_all = torch.randn((2, 8), dtype=torch.float32, device='npu')
    running_mean = torch.zeros(8, dtype=torch.float32, device='npu')
    running_var = torch.ones(8, dtype=torch.float32, device='npu')
    counts = torch.tensor([16, 16], dtype=torch.int32, device='npu')

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    batch_mean, batch_invstd = model(input, sum_all, square_sum_all, running_mean, running_var, counts)
    print(batch_mean, batch_invstd)
    ```
