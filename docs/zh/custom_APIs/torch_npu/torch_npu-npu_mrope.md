# torch_npu.npu_mrope

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

> 说明：本接口在 PyTorch 侧通过 [aclnnRopeWithSinCosCacheV2](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/rope_with_sin_cos_cache) 等底层算子实现；产品支持范围与上述 CANN 算子一致。若运行环境不支持 `aclnnRopeWithSinCosCacheV2`，使用 `cache_mode='interleave'` 时会报错，需升级 CANN 或改用 `cache_mode='default'`（将回退至 `aclnnRopeWithSinCosCache`）。

## 功能说明

- **API 功能**：在推理等场景中，将正弦、余弦位置编码以 cache 形式传入，对 `query`、`key` 执行旋转位置编码（RoPE）。支持普通 **RoPE** 与多模态 **MRoPE**（Multimodal RoPE）两种模式。
- **与底层算子的关系**：相较 `aclnnRopeWithSinCosCache`，`aclnnRopeWithSinCosCacheV2` 增加 `cacheMode` 属性，用于指定 cos、sin 的拼接方式：
  - `cacheMode = 0`：分段式拼接，与 V1 行为一致。
  - `cacheMode = 1`：交错式拼接 cos、sin。
- **Torch 参数映射**：`torch_npu.npu_mrope` 中 `cache_mode='default'` 对应 `cacheMode=0`，`cache_mode='interleave'` 对应 `cacheMode=1`。
- **模式区分**：
  - **RoPE 模式**：`positions` 为一维，shape 为 `(numTokens)`；`mrope_section` 使用默认值（如不使能 MRoPE，可传 `[0, 0, 0]`）。
  - **MRoPE 模式**：`positions` 为二维，shape 为 `(m, numTokens)`，其中 `m` 为 `mrope_section` 分段数，支持 **3** 或 **4**（与 CANN 文档一致）；需传入非空的 `mrope_section` 以整合各段上的位置编码。
- **旋转方式**（与 CANN 文档一致）：
  - `rotary_mode='half'`：GPT-NeoX 风格（`rotate_half`），将 `query`/`key` 在 `rotary_dim` 部分沿最后一维二等分后旋转。
  - `rotary_mode='interleave'`：GPT-J 风格（`rotate_interleaved`），按奇偶维交错取数后旋转。
- **计算要点摘要**（详见 CANN 文档中的公式）：
  - 从 `cos_sin_cache` 中按 `positions` 索引取出 `cosSin`，再 `chunk(2, dim=-1)` 得到 `cos`、`sin`。
  - 仅对 `query[..., :rotary_dim]`（及 key 的对应部分）做旋转，其余维度原样拼接回去。

## 函数原型

```
torch_npu.npu_mrope(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    *,
    mrope_section=None,
    rotary_mode='half',
    cache_mode='default',
) -> (Tensor, Tensor)
```

## 参数说明

- **positions** (`Tensor`)：位置索引，用于从 cache 中选取位置编码。  
  - **RoPE**：1D，shape `(numTokens)`，类型 `INT32` 或 `INT64`，格式 `ND`，可非连续。  
  - **MRoPE**：2D，shape `(3, numTokens)` 或 `(4, numTokens)`，要求与 `mrope_section` 分段数一致。  
  - 各索引取值应小于 `cos_sin_cache` 第 0 维长度 `max_seq_len`。
- **query** (`Tensor`)：待施加 RoPE 的 Query，2D，shape `(num_tokens, num_q_heads * head_size)`，类型 `FLOAT16`、`BFLOAT16` 或 `FLOAT32`，格式 `ND`，可非连续。
- **key** (`Tensor`)：待施加 RoPE 的 Key，2D，shape `(num_tokens, num_k_heads * head_size)`，类型与 `query` 一致，格式 `ND`，可非连续。
- **cos_sin_cache** (`Tensor`)：预计算的位置编码 cache，2D，shape `(max_seq_len, rotary_dim)`。最后一维经 `chunk(2, dim=-1)` 拆成 cos、sin 两半。`query`、`key`、`cos_sin_cache` 的浮点类型须一致。
- **head_size** (`int`)：单头维度大小，即每个注意力头的特征维长度。
- **mrope_section** (`int[]`，可选)：MRoPE 各段在 half 旋转维度上的长度配置；不传或 `[0, 0, 0]` 表示 **RoPE 模式**。使能 MRoPE 时需与 `positions` 的行数及 `rotary_dim` 约束一致（见约束说明）。
- **rotary_mode** (`str`，可选)：`half` 表示 NeoX 风格，`interleave` 表示 GPT-J 风格。默认 `half`。
- **cache_mode** (`str`，可选)：`default` 对应分段式 cos/sin 拼接（`cacheMode=0`）；`interleave` 对应交错式拼接（`cacheMode=1`）。默认 `default`。

## 返回值说明

返回二元组 `(query_out, key_out)`：

- 分别为对 `query`、`key` 施加 RoPE/MRoPE 后的张量，shape 与 dtype 与输入一致。

## 约束说明

- **维度**：`query`、`key`、`cos_sin_cache` 均为 2 维；`rotary_dim` 为 `cos_sin_cache` 最后一维大小，且满足 `rotary_dim <= head_size`。
- **对齐与倍数**（与 CANN 一致）：
  - `head_size`：`FLOAT16`/`BFLOAT16` 时为 **32** 的倍数；`FLOAT32` 时为 **16** 的倍数。
  - `rotary_dim`：`FLOAT16`/`BFLOAT16` 时为 **32** 的倍数；`FLOAT32` 时为 **16** 的倍数。
- **MRoPE**：`mrope_section` 各元素之和应等于 **`rotary_dim / 2`**（与 half 维度的 cos/sin 分段一致）。
- **MRoPE 支持的 `mrope_section` 取值**（与 CANN 一致）：当前支持 `[16, 24, 24]`、`[24, 20, 20]`、`[8, 12, 12]`、`[16, 16, 16, 16]`。
- **cache_mode**：MRoPE 下仅支持 `default` 与 `interleave`。当 `mrope_section` 为 **`[16, 16, 16, 16]`** 时，**仅支持 `cache_mode='default'`**（即不支持交错拼接）。
- **环境**：若需 `cache_mode='interleave'`，运行环境必须提供 **`aclnnRopeWithSinCosCacheV2`**；否则会报错并提示升级 CANN 或使用 `cache_mode='default'`。
- **确定性**：底层 `aclnnRopeWithSinCosCacheV2` 默认为确定性实现。

## 调用示例

### MRoPE 示例（三段 `mrope_section`）

`positions` 为二维 `(3, num_tokens)`，与 `[16, 24, 24]` 等三段配置对应；`mrope_section` 三数之和须等于 `rotary_dim / 2`。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope = torch.arange(num_tokens, dtype=torch.int64).repeat(3, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_3 = [16, 24, 24]
query_out, key_out = torch_npu.npu_mrope(
    positions_mrope,
    query,
    key,
    cos_sin_cache,
    head_size,
    mrope_section=mrope_section_3,
    rotary_mode="half",
    cache_mode="default",
)
```

### MRoPE 示例（四段 `mrope_section`）

`positions` 为 `(4, num_tokens)`，与 `[16, 16, 16, 16]` 等四段配置对应；该配置下 **`cache_mode` 仅支持 `'default'`**。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope4 = torch.arange(num_tokens, dtype=torch.int64).repeat(4, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_4 = [16, 16, 16, 16]
query_out, key_out = torch_npu.npu_mrope(
    positions_mrope4,
    query,
    key,
    cos_sin_cache,
    head_size,
    mrope_section=mrope_section_4,
    rotary_mode="half",
    cache_mode="default",
)
```

### RoPE（非 MRoPE）示例

使用一维 `positions`，shape 为 `(num_tokens,)`；`mrope_section` 置为 `[0, 0, 0]` 表示不使能 MRoPE（与当前 `torch_npu` 默认行为一致）。

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_1d = torch.arange(num_tokens, dtype=torch.int64).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

query_out, key_out = torch_npu.npu_mrope(
    positions_1d,
    query,
    key,
    cos_sin_cache,
    head_size,
    mrope_section=[0, 0, 0],
    rotary_mode="half",
    cache_mode="default",
)
```

## 参考

- CANN 算子说明：[aclnnRopeWithSinCosCacheV2.md](../../../../../ops-transformer/posembedding/rope_with_sin_cos_cache/docs/aclnnRopeWithSinCosCacheV2.md)（源码与公式以 ops-transformer 仓库为准）。
