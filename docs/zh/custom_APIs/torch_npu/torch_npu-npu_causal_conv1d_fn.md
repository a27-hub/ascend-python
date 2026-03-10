# torch_npu.npu_causal_conv1d_fn

## 产品支持情况

| 产品                                    | 是否支持 |
| --------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                            |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

## 功能说明

- API功能：对序列执行因果一维卷积（prefill场景）。沿序列维度，使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列尾部的数据（长度为卷积核宽减1）原地更新到缓存。

- 支持场景：

  | 场景类型 | 情况概要 |
  | --- | --- |
  | npu_causal_conv1d_fn（prefill） | 1. `x`支持CuSeqLen格式，shape为`[cu_seq_len, dim]`，其中`cu_seq_len`为batch内所有变长序列拼接后的总长度<br>2. `weight`的shape为`[K, dim]`，K固定为3<br>3. 每个序列卷积前，使用长度为K-1的缓存数据对序列头部进行padding，保证因果性 |

- 计算公式：

  K是卷积核宽度，L是原始序列长度，dim是特征维度。
  
  - 缓存拼接：

    $$
    x'[i, dim] =
    \begin{cases}
    cacheState[i, dim], & 0 \leq i < K-1 \\
    x[i - (K-1), dim], & K-1 \leq i < L + K - 1
    \end{cases}
    $$

  - 因果1维卷积：

    $$
    y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim]
    $$

  - 缓存更新：

    $$
    cacheState[i, dim] = x'[L + i, dim], \quad i = 0, 1, \dots, K-2
    $$

## 函数原型

```
torch_npu.npu_causal_conv1d_fn(x, weight, conv_states, *, bias=None, query_start_loc=None, cache_indices=None, initial_state_mode=None, activation="None", pad_slot_id=-1) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，输入序列。数据类型支持`float16`、`bfloat16`，不支持空Tensor。shape为2维`[cu_seq_len, dim]`，其中`cu_seq_len`为batch内所有变长序列拼接后的总长度，取值范围为`[1, 65536]`；`dim`为特征维度，取值范围为`[64, 16384]`。
- **weight** (`Tensor`)：必选参数，输入张量，因果1维卷积核。数据类型与`x`相同，不支持空Tensor。shape为2维`[K, dim]`，其中K为卷积核宽度，固定为3；`dim`为特征维度，取值范围为`[64, 16384]`。
- **conv_states** (`Tensor`)：必选参数，输入/输出张量，缓存状态张量，存储各序列的历史token数据，各序列计算完成后原地更新。数据类型与`x`相同，不支持空Tensor。shape为3维`[..., K-1, dim]`，第0维大小不固定。
- **bias** (`Tensor`)：可选参数，输入张量，卷积的偏置。数据类型与`x`相同。shape为1维`[dim,]`，默认值为`None`。
- **query_start_loc** (`Tensor`)：可选参数，输入张量，序列起始位置索引，记录各序列在拼接张量`x`中的起始位置，`query_start_loc[i]`表示第i个序列的起始偏移。数据类型支持`int32`。shape为1维`[batch+1,]`，`batch`取值范围为`[1, 256]`。当`x`的shape为2维时，不可省略。
- **cache_indices** (`Tensor`)：可选参数，输入张量，缓存索引，指定每个序列对应的缓存状态在`conv_states`中的索引。数据类型支持`int32`。shape为1维`[batch,]`。当`x`的shape为2维时，不可省略。
- **initial_state_mode** (`Tensor`)：可选参数，输入张量，初始状态标志，表示各序列是否使用缓存数据。数据类型支持`int32`。shape为1维`[batch,]`，默认值为`None`。
- **activation** (`str`)：可选参数，激活函数类型，支持`"None"`、`"silu"`、`"swish"`，默认值为`"None"`。
- **pad_slot_id** (`int`)：可选参数，用于跳过不需要参与计算的batch，默认值为`-1`。

## 返回值说明

- **y** (`Tensor`)：输出张量，`x`经过因果1维卷积计算后的结果。数据类型与`x`相同，shape与`x`保持一致，为`[cu_seq_len, dim]`。

此外，`conv_states`将被原地更新为各序列最新的缓存状态。

## 约束说明

- 该接口仅支持prefill场景使用。
- `x`的shape为2维时，`query_start_loc`和`cache_indices`不可省略。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 接口入参需在参数范围内。

## 调用示例

```python
import torch
import torch_npu

K = 3           # 卷积核宽度（状态长度 = K - 1 = 2）
dim = 64        # 特征维度
batch = 4       # 批次大小
dtype = torch.bfloat16

weight = torch.randn(K, dim, dtype=dtype).npu()

# 变长拼接格式（prefill场景）
seq_lens = [5, 3, 7, 4]               # 每个序列的 token 数
cu_seq_len = sum(seq_lens)            # 总 token 数 = 19

x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()

# 序列起始位置索引，共 batch+1 个元素
query_start_loc = torch.tensor([0, 5, 8, 15, 19], dtype=torch.int32).npu()

# 缓存状态池
num_slots = 8
conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()  # shape: (8, 2, 64)
cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()

# 初始状态模式：1=使用缓存，0=零填充
initial_state_mode = torch.tensor([1, 1, 0, 0], dtype=torch.int32).npu()

out = torch_npu.npu_causal_conv1d_fn(
    x,
    weight,
    conv_states,
    query_start_loc=query_start_loc,
    cache_indices=cache_indices,
    initial_state_mode=initial_state_mode,
    activation="None",
    pad_slot_id=-1,
)
print(f"output shape: {out.shape}")  # 预期 (19, 64)
# conv_states 已被原地更新为各序列最新的缓存状态
```
